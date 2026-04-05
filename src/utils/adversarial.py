"""
Utilities — Adversarial Example Generator

Generates adversarial phishing email variants for training set hardening.
Applies common evasion techniques used by real-world phishing authors:
  - Random character substitution (homoglyphs)
  - Zero-width space insertion
  - URL obfuscation (hex encoding, extra subdomains)
  - Subject line perturbation

These augmented samples help the model generalize to obfuscated phishing.

Dependencies: random, re
"""

import logging
import random
import re

logger = logging.getLogger(__name__)

# Cyrillic / lookalike character substitutions (Latin → homoglyph)
_HOMOGLYPH_MAP: dict[str, str] = {
    "a": "а",  # Cyrillic а
    "e": "е",  # Cyrillic е
    "o": "о",  # Cyrillic о
    "p": "р",  # Cyrillic р
    "c": "с",  # Cyrillic с
    "i": "і",  # Cyrillic і
}

_ZERO_WIDTH_SPACE = "\u200b"


def insert_homoglyphs(text: str, rate: float = 0.05) -> str:
    """Randomly substitute Latin characters with visual homoglyphs.

    Args:
        text: Input text string.
        rate: Fraction of eligible characters to substitute.

    Returns:
        Text with random homoglyph substitutions.
    """
    result = []
    for char in text:
        lower = char.lower()
        if lower in _HOMOGLYPH_MAP and random.random() < rate:
            sub = _HOMOGLYPH_MAP[lower]
            result.append(sub if char.islower() else sub.upper())
        else:
            result.append(char)
    return "".join(result)


def insert_zero_width_spaces(text: str, rate: float = 0.03) -> str:
    """Insert zero-width spaces at random positions to confuse tokenizers.

    Args:
        text: Input text string.
        rate: Probability of inserting a ZWS after each character.

    Returns:
        Text with randomly inserted zero-width spaces.
    """
    result = []
    for char in text:
        result.append(char)
        if random.random() < rate:
            result.append(_ZERO_WIDTH_SPACE)
    return "".join(result)


def obfuscate_url(url: str) -> str:
    """Apply random URL obfuscation technique.

    Techniques:
        - Hex-encode path characters
        - Add a benign-looking prefix subdomain

    Args:
        url: Original URL string.

    Returns:
        Obfuscated URL string.
    """
    technique = random.choice(["hex", "subdomain"])

    if technique == "hex":
        # Hex-encode a random character in the path
        path_match = re.search(r"(https?://[^/]+)(/.+)", url)
        if path_match:
            base, path = path_match.groups()
            chars = list(path)
            idx = random.randint(0, len(chars) - 1)
            chars[idx] = f"%{ord(chars[idx]):02X}"
            return base + "".join(chars)

    elif technique == "subdomain":
        # Insert a convincing-looking subdomain
        prefixes = ["secure", "login", "account", "verify", "update"]
        prefix = random.choice(prefixes)
        url = re.sub(r"(https?://)", rf"\1{prefix}.", url)

    return url


def generate_adversarial_sample(text: str, seed: int | None = None) -> str:
    """Apply a random combination of adversarial perturbations to a phishing email.

    Args:
        text: Original phishing email body text.
        seed: Optional random seed for reproducibility.

    Returns:
        Perturbed email text.
    """
    if seed is not None:
        random.seed(seed)

    text = insert_homoglyphs(text)
    text = insert_zero_width_spaces(text)

    # Obfuscate any URLs found in the text
    url_pattern = re.compile(r"https?://\S+")
    text = url_pattern.sub(lambda m: obfuscate_url(m.group()), text)

    return text
