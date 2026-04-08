"""Train and compare learned metadata + URL phishing models properly.

This script follows a clean protocol:
1. Create reproducible train / validation / test splits.
2. Train each candidate model on train only.
3. Tune the phishing threshold on validation only.
4. Select the winning model using validation metrics only.
5. Report untouched test metrics for the selected threshold.
6. Refit the winner on train+validation for deployment and save it.

  - NEW (default): data/processed/features/eml_training_features.csv
    Produced by scripts/extract_eml_features.py from EPVME + SpamAssassin
    .eml files. Contains direct feature columns + 'label' (0/1).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    METADATA_URL_MODEL_METRICS_PATH,
    METADATA_URL_MODEL_PATH,
    METADATA_URL_SPLIT_MANIFEST_PATH,
)
from src.pipeline.metadata_url_model import (
    DEFAULT_FEATURE_NAMES,
    MetadataURLModel,
)

# New default: .eml-based features (EPVME + SpamAssassin with augmentation)
DEFAULT_RESULTS_CSV = PROJECT_ROOT / "data" / "processed" / "features" / "eml_training_features.csv"
DEFAULT_COMPARISON_CSV = PROJECT_ROOT / "results" / "metadata_url_model_comparison.csv"
COMPARISON_MODELS = ("xgboost", "catboost")


def parse_args() -> argparse.Namespace:
    """Parse training arguments."""
    parser = argparse.ArgumentParser(
        description="Train a metadata + URL phishing classifier from the feature table."
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=DEFAULT_RESULTS_CSV,
        help="Path to the feature table CSV. Defaults to the new .eml-based features.",
    )

    parser.add_argument(
        "--model",
        choices=("xgboost", "catboost", "random_forest", "logreg"),
        default="xgboost",
        help="Estimator to train when not using --compare.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Train and compare XGBoost vs CatBoost on the same split, then save the validation-selected winner.",
    )
    parser.add_argument(
        "--comparison-csv",
        type=Path,
        default=DEFAULT_COMPARISON_CSV,
        help="Path to save model comparison metrics when using --compare.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("f1", "roc_auc", "average_precision", "balanced_accuracy", "accuracy"),
        default="f1",
        help="Validation metric used to choose the saved winner in --compare mode.",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="Fraction of labeled rows reserved for validation.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of labeled rows reserved for final testing.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for the splits and estimators.",
    )
    parser.add_argument(
        "--threshold-objective",
        choices=("balanced", "f1"),
        default="balanced",
        help="How to choose the phishing threshold from the validation split.",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=METADATA_URL_MODEL_PATH,
        help="Path to save the trained metadata + URL model bundle.",
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=METADATA_URL_MODEL_METRICS_PATH,
        help="Path to save training/evaluation metrics as JSON.",
    )
    parser.add_argument(
        "--balance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Randomly downsample the majority class to reach a 1:1 ratio for balanced training.",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=METADATA_URL_SPLIT_MANIFEST_PATH,
        help="Path to save the train/validation/test split manifest.",
    )
    return parser.parse_args()


def main() -> None:
    """Train one or more metadata + URL models with proper evaluation."""
    args = parse_args()
    validate_split_sizes(args.val_size, args.test_size)

    dataset = load_dataset(args.results_csv)

    if args.balance:
        # Perform random downsampling for class balance
        neg_indices = dataset[dataset["label"] == 0].index
        pos_indices = dataset[dataset["label"] == 1].index
        
        n_neg = len(neg_indices)
        n_pos = len(pos_indices)
        target_size = min(n_neg, n_pos)
        
        if n_neg > target_size:
            neg_indices = dataset[dataset["label"] == 0].sample(n=target_size, random_state=args.random_state).index
        if n_pos > target_size:
            pos_indices = dataset[dataset["label"] == 1].sample(n=target_size, random_state=args.random_state).index
            
        dataset = dataset.loc[neg_indices.union(pos_indices)].copy()
        print("=" * 60)
        print(f"   CLASS BALANCING ENABLED")
        print(f"   Downsampling majority class to match minority class.")
        print(f"   Samples per class: {target_size}")
        print(f"   Total Training Size: {len(dataset)}")
        print("=" * 60)

    split_manifest = create_split_manifest(
        dataset,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    save_split_manifest(split_manifest, args.split_manifest)

    x_train, y_train = frame_to_xy(select_split_rows(dataset, split_manifest, "train"))
    x_val, y_val = frame_to_xy(select_split_rows(dataset, split_manifest, "val"))
    x_test, y_test = frame_to_xy(select_split_rows(dataset, split_manifest, "test"))
    x_train_val = pd.concat([x_train, x_val], axis=0)
    y_train_val = pd.concat([y_train, y_val], axis=0)

    # Compute class imbalance ratio for estimators that support it
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    class_weight_ratio = n_neg / max(n_pos, 1)
    print(f"Class distribution — train: {n_pos} phishing, {n_neg} legitimate (ratio {n_pos/max(n_neg,1):.1f}:1)")

    model_names = list(COMPARISON_MODELS) if args.compare else [args.model]
    runs: list[dict[str, Any]] = []

    for model_name in model_names:
        try:
            estimator = build_estimator(
                model_name, random_state=args.random_state, scale_pos_weight=class_weight_ratio,
            )
        except ImportError as exc:
            if args.compare:
                print(f"Skipping {model_name}: {exc}")
                continue
            raise

        run = train_and_evaluate(
            model_name=model_name,
            estimator=estimator,
            x_train=x_train,
            x_val=x_val,
            x_test=x_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            threshold_objective=args.threshold_objective,
        )
        runs.append(run)

    if not runs:
        raise RuntimeError("No models were trained successfully.")

    winning_run = select_best_run(runs, args.selection_metric)
    final_estimator = build_estimator(
        winning_run["model_name"], random_state=args.random_state, scale_pos_weight=class_weight_ratio,
    )
    final_estimator.fit(x_train_val, y_train_val)

    metrics_payload = build_metrics_payload(
        winning_run=winning_run,
        all_runs=runs,
        selection_metric=args.selection_metric,
        split_manifest_path=args.split_manifest,
        split_sizes={
            "train_rows": int(len(x_train)),
            "validation_rows": int(len(x_val)),
            "test_rows": int(len(x_test)),
            "deployment_train_rows": int(len(x_train_val)),
        },
    )

    bundle = MetadataURLModel(
        model=final_estimator,
        feature_names=list(DEFAULT_FEATURE_NAMES),
        threshold=float(winning_run["selected_threshold"]),
        model_name=f"{winning_run['model_name']}_metadata_url",
        training_metrics=metrics_payload,
    )
    model_path = bundle.save(args.output_model)

    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_json.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    if args.compare:
        save_comparison_csv(runs, args.comparison_csv)

    print_summary(
        input_csv=args.results_csv,
        runs=runs,
        winning_run=winning_run,
        output_model=model_path,
        metrics_json=args.metrics_json,
        split_manifest=args.split_manifest,
        comparison_csv=args.comparison_csv if args.compare else None,
        selection_metric=args.selection_metric,
    )


def validate_split_sizes(val_size: float, test_size: float) -> None:
    """Validate the requested split fractions."""
    if not 0.0 < val_size < 1.0:
        raise ValueError("--val-size must be between 0 and 1")
    if not 0.0 < test_size < 1.0:
        raise ValueError("--test-size must be between 0 and 1")
    if val_size + test_size >= 1.0:
        raise ValueError("--val-size + --test-size must be less than 1")


def load_dataset(results_csv: Path) -> pd.DataFrame:
    """Load and validate the feature CSV.
    Uses the new .eml-based format (label=0/1).
    """
    if not results_csv.exists():
        raise FileNotFoundError(
            f"Could not find feature table at {results_csv}. "
            "Generate it first with: scripts/extract_eml_features.py"
        )
    frame = pd.read_csv(results_csv, low_memory=False)

    # New format has integer labels
    frame = frame[frame["label"].isin([0, 1])].copy()

    if frame.empty:
        raise ValueError("No labeled rows were found in the supplied feature CSV.")
    if "row_number" not in frame.columns:
        raise ValueError("The feature CSV must contain a row_number column.")
    return frame


def create_split_manifest(
    dataset: pd.DataFrame,
    *,
    val_size: float,
    test_size: float,
    random_state: int,
) -> pd.DataFrame:
    """Create a reproducible train/validation/test split manifest."""
    label_col = "label"
    rows = dataset[["row_number", label_col]].copy()
    train_rows, holdout_rows = train_test_split(
        rows,
        test_size=val_size + test_size,
        random_state=random_state,
        stratify=rows[label_col],
    )
    relative_test_size = test_size / (val_size + test_size)
    val_rows, test_rows = train_test_split(
        holdout_rows,
        test_size=relative_test_size,
        random_state=random_state,
        stratify=holdout_rows[label_col],
    )

    manifest = pd.concat(
        [
            train_rows.assign(split="train"),
            val_rows.assign(split="val"),
            test_rows.assign(split="test"),
        ],
        ignore_index=True,
    )
    return manifest.sort_values("row_number").reset_index(drop=True)


def save_split_manifest(split_manifest: pd.DataFrame, output_path: Path) -> None:
    """Persist the split assignments used for training/evaluation."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split_manifest.to_csv(output_path, index=False)


def select_split_rows(
    dataset: pd.DataFrame,
    split_manifest: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    """Return the dataset rows assigned to one named split."""
    selected_row_numbers = split_manifest.loc[
        split_manifest["split"] == split_name,
        "row_number",
    ].astype(int)
    frame = dataset[dataset["row_number"].isin(selected_row_numbers)].copy()
    return frame.sort_values("row_number").reset_index(drop=True)


def frame_to_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Convert a results-frame slice into features and labels.

    For the new .eml-based format, features are already stored as direct columns.
    """
    features = frame[list(DEFAULT_FEATURE_NAMES)].copy()
    # Fill NaN with 0.0 for robustness
    features = features.fillna(0.0)
    targets = frame["label"].astype(int)

    return features, targets


def train_and_evaluate(
    *,
    model_name: str,
    estimator: Any,
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    threshold_objective: str,
) -> dict[str, Any]:
    """Fit one estimator, tune threshold on validation, and score untouched test."""
    estimator.fit(x_train, y_train)
    val_probabilities = estimator.predict_proba(x_val)[:, 1]
    val_threshold_analysis = sweep_probability_thresholds(y_val.tolist(), val_probabilities.tolist())
    selected_threshold = (
        val_threshold_analysis["best_balanced_threshold"]["threshold"]
        if threshold_objective == "balanced"
        else val_threshold_analysis["best_f1_threshold"]["threshold"]
    )
    val_predictions = [1 if probability >= selected_threshold else 0 for probability in val_probabilities]
    test_probabilities = estimator.predict_proba(x_test)[:, 1]
    test_predictions = [1 if probability >= selected_threshold else 0 for probability in test_probabilities]

    validation_metrics = build_metric_block(
        y_true=y_val.tolist(),
        probabilities=val_probabilities.tolist(),
        predictions=val_predictions,
    )
    test_metrics = build_metric_block(
        y_true=y_test.tolist(),
        probabilities=test_probabilities.tolist(),
        predictions=test_predictions,
    )

    return {
        "model_name": model_name,
        "estimator": estimator,
        "selected_threshold": float(selected_threshold),
        "threshold_objective": threshold_objective,
        "validation": validation_metrics,
        "test": test_metrics,
        "best_balanced_threshold": val_threshold_analysis["best_balanced_threshold"],
        "best_f1_threshold": val_threshold_analysis["best_f1_threshold"],
    }


def build_metric_block(
    *,
    y_true: list[int],
    probabilities: list[float],
    predictions: list[int],
) -> dict[str, float | int]:
    """Build a metric block for a probability/prediction set."""
    tp, fp, tn, fn = confusion_counts(y_true, predictions)
    precision = precision_score(y_true, predictions, zero_division=0)
    recall = recall_score(y_true, predictions, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision,
        "recall": recall,
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "average_precision": average_precision_score(y_true, probabilities),
        "balanced_accuracy": (recall + specificity) / 2,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def build_estimator(model_name: str, *, random_state: int, scale_pos_weight: float = 1.0):
    """Return the requested phishing estimator.

    Args:
        model_name: One of 'xgboost', 'catboost', 'random_forest', 'logreg'.
        random_state: Seed for reproducibility.
        scale_pos_weight: Ratio of negative/positive samples for class imbalance.
    """
    if model_name == "logreg":
        return Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        )

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=500,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )

    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except Exception as exc:
            raise ImportError(
                "xgboost is required for --model xgboost. Install project dependencies first."
            ) from exc

        return XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.5,
            reg_alpha=0.1,
            min_child_weight=3,
            gamma=0.1,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )

    if model_name == "catboost":
        try:
            from catboost import CatBoostClassifier
        except Exception as exc:
            raise ImportError(
                "catboost is required for --model catboost. Install project dependencies first."
            ) from exc

        return CatBoostClassifier(
            iterations=600,
            depth=6,
            learning_rate=0.03,
            loss_function="Logloss",
            eval_metric="AUC",
            auto_class_weights="Balanced",
            random_seed=random_state,
            verbose=False,
        )

    raise ValueError(f"Unsupported model: {model_name}")


def sweep_probability_thresholds(
    y_true: list[int],
    probabilities: list[float],
) -> dict[str, dict[str, float]]:
    """Find useful operating thresholds on validation probabilities only."""
    rows: list[dict[str, float]] = []
    for threshold in [step / 100 for step in range(5, 96)]:
        predictions = [1 if probability >= threshold else 0 for probability in probabilities]
        tp, fp, tn, fn = confusion_counts(y_true, predictions)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        balanced_accuracy = (recall + specificity) / 2
        rows.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "balanced_accuracy": balanced_accuracy,
            }
        )

    best_balanced_threshold = max(
        rows,
        key=lambda row: (row["balanced_accuracy"], row["f1"], row["precision"], -row["threshold"]),
    )
    best_f1_threshold = max(
        rows,
        key=lambda row: (row["f1"], row["balanced_accuracy"], row["precision"], -row["threshold"]),
    )
    return {
        "best_balanced_threshold": best_balanced_threshold,
        "best_f1_threshold": best_f1_threshold,
    }


def select_best_run(runs: list[dict[str, Any]], selection_metric: str) -> dict[str, Any]:
    """Choose the winning run using validation metrics only."""
    return max(
        runs,
        key=lambda run: (
            float(run["validation"][selection_metric]),
            float(run["validation"]["f1"]),
            float(run["validation"]["roc_auc"]),
            float(run["validation"]["accuracy"]),
        ),
    )


def build_metrics_payload(
    *,
    winning_run: dict[str, Any],
    all_runs: list[dict[str, Any]],
    selection_metric: str,
    split_manifest_path: Path,
    split_sizes: dict[str, int],
) -> dict[str, Any]:
    """Build the JSON payload saved next to the selected model."""

    def _strip_estimator(run: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in run.items() if key != "estimator"}

    return {
        "protocol": "train_val_test",
        "model_name": winning_run["model_name"],
        "selected_threshold": winning_run["selected_threshold"],
        "threshold_objective": winning_run["threshold_objective"],
        "selection_metric": selection_metric,
        "split_manifest_path": str(split_manifest_path),
        **split_sizes,
        "feature_count": len(DEFAULT_FEATURE_NAMES),
        "validation_metrics": winning_run["validation"],
        "test_metrics": winning_run["test"],
        "best_balanced_threshold": winning_run["best_balanced_threshold"],
        "best_f1_threshold": winning_run["best_f1_threshold"],
        "saved_model_training_data": "train_plus_validation",
        "winner_selected_on": "validation_only",
        "final_reported_metrics_from": "untouched_test_split",
        "model_comparison": [_strip_estimator(run) for run in all_runs],
    }


def save_comparison_csv(runs: list[dict[str, Any]], output_csv: Path) -> None:
    """Save validation/test comparison metrics for all trained models."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in runs:
        rows.append(
            {
                "model_name": run["model_name"],
                "selected_threshold": run["selected_threshold"],
                "threshold_objective": run["threshold_objective"],
                "val_accuracy": run["validation"]["accuracy"],
                "val_precision": run["validation"]["precision"],
                "val_recall": run["validation"]["recall"],
                "val_f1": run["validation"]["f1"],
                "val_roc_auc": run["validation"]["roc_auc"],
                "val_average_precision": run["validation"]["average_precision"],
                "val_balanced_accuracy": run["validation"]["balanced_accuracy"],
                "test_accuracy": run["test"]["accuracy"],
                "test_precision": run["test"]["precision"],
                "test_recall": run["test"]["recall"],
                "test_f1": run["test"]["f1"],
                "test_roc_auc": run["test"]["roc_auc"],
                "test_average_precision": run["test"]["average_precision"],
                "test_balanced_accuracy": run["test"]["balanced_accuracy"],
            }
        )
    pd.DataFrame(rows).to_csv(output_csv, index=False)


def print_summary(
    *,
    input_csv: Path,
    runs: list[dict[str, Any]],
    winning_run: dict[str, Any],
    output_model: Path,
    metrics_json: Path,
    split_manifest: Path,
    comparison_csv: Path | None,
    selection_metric: str,
) -> None:
    """Print a concise training/comparison summary."""
    print("=== Metadata + URL Model Training ===")
    print(f"Input results CSV: {input_csv}")
    print(f"Split manifest: {split_manifest}")
    print(f"Selection metric: validation {selection_metric}")
    print()

    if len(runs) > 1:
        print("=== Model Comparison ===")
        for run in runs:
            print(
                f"{run['model_name']}: "
                f"val_f1={run['validation']['f1'] * 100:.1f}% | "
                f"val_roc_auc={run['validation']['roc_auc'] * 100:.1f}% | "
                f"test_f1={run['test']['f1'] * 100:.1f}% | "
                f"test_roc_auc={run['test']['roc_auc'] * 100:.1f}% | "
                f"threshold={run['selected_threshold']:.3f}"
            )
        print()

    print("=== Selected Model ===")
    print(f"Winner: {winning_run['model_name']}")
    print(f"Threshold chosen on validation: {winning_run['selected_threshold']:.3f}")
    print(f"Validation F1: {winning_run['validation']['f1'] * 100:.1f}%")
    print(f"Validation ROC-AUC: {winning_run['validation']['roc_auc'] * 100:.1f}%")
    print(f"Untouched test accuracy: {winning_run['test']['accuracy'] * 100:.1f}%")
    print(f"Untouched test precision: {winning_run['test']['precision'] * 100:.1f}%")
    print(f"Untouched test recall: {winning_run['test']['recall'] * 100:.1f}%")
    print(f"Untouched test F1: {winning_run['test']['f1'] * 100:.1f}%")
    print(f"Untouched test ROC-AUC: {winning_run['test']['roc_auc'] * 100:.1f}%")
    print(f"Model saved to: {output_model}")
    print(f"Metrics saved to: {metrics_json}")
    if comparison_csv is not None:
        print(f"Comparison CSV saved to: {comparison_csv}")


def confusion_counts(y_true: list[int], predictions: list[int]) -> tuple[int, int, int, int]:
    """Return TP, FP, TN, FN counts."""
    tp = sum(1 for truth, pred in zip(y_true, predictions) if truth == 1 and pred == 1)
    fp = sum(1 for truth, pred in zip(y_true, predictions) if truth == 0 and pred == 1)
    tn = sum(1 for truth, pred in zip(y_true, predictions) if truth == 0 and pred == 0)
    fn = sum(1 for truth, pred in zip(y_true, predictions) if truth == 1 and pred == 0)
    return tp, fp, tn, fn


if __name__ == "__main__":
    main()
