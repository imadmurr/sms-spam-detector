"""Regex fingerprint features, shared by the notebook and the Streamlit app.

Always run on the RAW message: cleaning deletes the very fingerprints these detect.
"""
import re

# One count column per pattern (spec section 4, regex feature catalogue).
FEATURES = {
    "n_urls": r"https?://\S+|www\.\S+|\b\w+\.(?:com|net|org|co\.uk)\b",
    "n_phones": r"\+?\d[\d\s-]{8,}\d",
    "n_shortcodes": r"\b\d{5}\b",
    "n_money": r"[£$€]\s?\d+|\b\d+\s?(?:p|pounds?)\b",
    "n_caps": r"\b[A-Z]{2,}\b",
    "n_urgent": r"(?i)\b(?:free|win|winner|won|claim|prize|urgent|txt|reply|stop|call now)\b",
    "n_exclaim": r"!",
}

# Compile once instead of on every call.
PATTERNS = {name: re.compile(pattern) for name, pattern in FEATURES.items()}

FINGERPRINTS = list(FEATURES)  # the 7 regex fingerprints (used by Model 0 and the app)
FEATURE_NAMES = FINGERPRINTS + ["n_chars", "n_words", "digit_ratio"]


def regex_features(text: str) -> dict:
    """Count each fingerprint in a raw SMS, plus simple length and digit statistics."""
    feats = {name: len(pattern.findall(text)) for name, pattern in PATTERNS.items()}
    feats["n_chars"] = len(text)
    feats["n_words"] = len(text.split())
    feats["digit_ratio"] = sum(ch.isdigit() for ch in text) / max(len(text), 1)
    return feats
