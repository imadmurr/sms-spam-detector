"""Regex fingerprint features and text cleaning, shared by the notebook and the Streamlit app.

regex_features() always runs on the RAW message: cleaning deletes the very fingerprints it detects.
"""
import html
import re

from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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


# --- Text cleaning (FR-3) and slang normalization (FR-4) ---
# Order matters: each rule only sees what the earlier rules left behind.

# Specific placeholders first, so "£2" becomes "money" before "2" is touched.
PLACEHOLDERS = [
    ("url", PATTERNS["n_urls"]),
    ("money", PATTERNS["n_money"]),
    ("phone", PATTERNS["n_phones"]),
]

# SMS slang map, applied while digits still exist ("2", "fr33").
SLANG = [
    (re.compile(r"\bu\b"), "you"),
    (re.compile(r"\bur\b"), "your"),
    (re.compile(r"\btxt\b"), "text"),
    (re.compile(r"\b2\b"), "to"),
    (re.compile(r"fr[e3]{2}"), "free"),
]

NUMBER = re.compile(r"\d+")
NON_LETTER = re.compile(r"[^a-z\s]")
REPEATED = re.compile(r"(\w)\1{2,}")
SPACES = re.compile(r"\s+")


def clean_text(text: str, slang: bool = True) -> str:
    """Turn a raw SMS into lowercase words and placeholder tokens for bag-of-words."""
    text = html.unescape(text.replace("å£", "£"))  # 1. fix encoding and HTML entities
    text = text.lower()  # CAPS were already counted by regex_features
    for token, pattern in PLACEHOLDERS:  # 2. url / money / phone placeholders
        text = pattern.sub(f" {token} ", text)
    if slang:  # 3. slang map (switchable for the ablation)
        for pattern, replacement in SLANG:
            text = pattern.sub(replacement, text)
    text = NUMBER.sub(" num ", text)  # 4. remaining numbers
    text = NON_LETTER.sub(" ", text)  # 5. punctuation, symbols and other non-ASCII
    text = REPEATED.sub(r"\1\1", text)  # 6. "sooo" -> "soo"
    return SPACES.sub(" ", text).strip()  #    collapse whitespace


# --- Tokens, stopwords, lemmas (FR-5 to FR-7) ---

NEGATIONS = {"not", "no", "nor", "never"}
STOPWORDS = set(stopwords.words("english")) - NEGATIONS  # keep negations: "not free" != "free"
LEMMATIZER = WordNetLemmatizer()


def wordnet_pos(tag: str) -> str:
    """Map a Penn Treebank tag (VBD, NNS, ...) to one of WordNet's four word types."""
    return {"J": "a", "V": "v", "R": "r"}.get(tag[0], "n")


def process_tokens(clean: str, remove_stopwords: bool = True, lemmatize: bool = True) -> list:
    """Tokenize cleaned text, POS-tag it (before stopwords, so the tagger sees context),
    drop stopwords and lemmatize with POS. The switches serve the ablation runs."""
    tagged = pos_tag(word_tokenize(clean))
    if remove_stopwords:
        tagged = [(word, tag) for word, tag in tagged if word not in STOPWORDS]
    if lemmatize:
        return [LEMMATIZER.lemmatize(word, wordnet_pos(tag)) for word, tag in tagged]
    return [word for word, _ in tagged]


# --- POS ratio features (FR-8), computed on RAW text: the tagger needs capitals and digits ---

POS_FEATURES = ["pos_vb", "pos_cd", "pos_nnp"]


def pos_features(text: str) -> dict:
    """Share of tokens that are verbs (VB*), numbers (CD) and proper nouns (NNP*)."""
    tags = [tag for _, tag in pos_tag(word_tokenize(text))]
    n = max(len(tags), 1)
    return {
        "pos_vb": sum(t.startswith("VB") for t in tags) / n,
        "pos_cd": sum(t == "CD" for t in tags) / n,
        "pos_nnp": sum(t.startswith("NNP") for t in tags) / n,
    }
