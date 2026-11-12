# 4. Technical Design

*How the pipeline is built · based on an ML design doc*

Raw text splits into two branches: regex and POS features are extracted from the untouched text, while a cleaned copy feeds the bag-of-words. The key rule: these features come first, because cleaning deletes the very fingerprints they detect.

```
raw SMS ──┬──► Branch A: regex fingerprints + POS ratios   (raw text)
          │
          └──► Branch B: clean → tokenize → stopwords → lemmatize → TF-IDF

Model 0     = Branch A only   (hand-written rule)
Models 1–2  = Branch B only   (Naive Bayes, Linear SVM)
Model 3     = Branch A + B    (Logistic Regression, final model)
```

Model 0 uses only the regex branch, Models 1–2 only the processed-text branch, and Model 3 combines both.

## Pipeline steps

1. Load, deduplicate, split train/test
2. Extract regex features from raw `text`; Model 0 rules run on these
3. Clean `text` into `clean_text` with regex, including the slang map
4. Tokenize, POS-tag, remove stopwords (keep negations), lemmatize with POS
5. Compute POS ratio features from the raw text (the tagger needs capitals and digits)
6. Vectorize with TF-IDF, unigrams + bigrams (fitted on train only, inside each pipeline)
7. Train Models 0–3, then run the ablation on Model 3
8. Evaluate, plot, save the final model, serve it in the Streamlit app

All feature code lives in `features.py`, shared by the notebook and the app, so the demo builds exactly the same input as training.

## Regex feature catalogue

Each pattern produces one count column.

```python
FEATURES = {
    "n_urls":       r"https?://\S+|www\.\S+|\b\w+\.(?:com|net|org|co\.uk)\b",
    "n_phones":     r"\+?\d[\d\s-]{8,}\d",
    "n_shortcodes": r"\b\d{5}\b",
    "n_money":      r"[£$€]\s?\d+|\b\d+\s?(?:p|pounds?)\b",
    "n_caps":       r"\b[A-Z]{2,}\b",
    "n_urgent":     r"(?i)\b(?:free|win|winner|won|claim|prize|urgent|txt|reply|stop|call now)\b",
    "n_exclaim":    r"!",
}
# plus: n_chars, n_words, digit_ratio
```

## Cleaning rules, in order

Order matters: each rule only sees what earlier rules left behind. Rules that read digits (the slang map) run before the rule that removes them, and specific patterns (money) run before general ones (any number).

1. Fix encoding: `å£` → `£`; `html.unescape()` for `&lt;` and similar; then lowercase (CAPS were already counted)
2. Replace URLs with `url`, money with `money`, phone numbers with `phone`
3. Slang map with word-boundary regex: `\bu\b` → you, `\bur\b` → your, `\btxt\b` → text, `\b2\b` → to, `fr[e3]{2}` → free
4. Replace remaining numbers with `num`
5. Remove punctuation and other non-ASCII: `re.sub(r"[^a-z\s]", " ", text)`
6. Collapse letters repeated 3+ times: `re.sub(r"(\w)\1{2,}", r"\1\1", text)`, then collapse whitespace

## Models

| Model | Input | Classifier | Shows |
| --- | --- | --- | --- |
| 0 | Regex fingerprints | Rule: spam if ≥ k distinct fingerprints fire (k = 3, tuned on train) | How far pure rules go, no ML |
| 1 | Bag-of-words counts | Multinomial Naive Bayes (also reported with TF-IDF as 1b) | The classic text baseline |
| 2 | TF-IDF words | Linear SVM | Usually the strongest classic text model |
| 3 | TF-IDF + scaled regex + POS features | Logistic Regression, `class_weight='balanced'`, `C=100` (tuned on train) | Final model: NLP features on top of words; probabilities and readable weights for the app |

Not used: GaussianNB (assumes normally distributed features, poor fit for word counts); LSTM and BERT (out of scope, listed as next steps).

## Tech stack

| Purpose | Library |
| --- | --- |
| Data | pandas, numpy |
| Regex and cleaning | `re`, `html` |
| Text processing | NLTK (`word_tokenize`, `stopwords`, `WordNetLemmatizer`, `PorterStemmer`, `LancasterStemmer`, `pos_tag`, `ngrams`) |
| Features and models | scikit-learn, joblib |
| Plots | matplotlib, wordcloud |
| Demo app (UI only) | Streamlit |

## Repository layout

```
├── app.py                    # Streamlit demo
├── features.py               # regex features, cleaning, tokens, POS, model input (shared)
├── sms_spam_detector.ipynb   # full pipeline, training and evaluation
├── assets/style.css          # app styling
├── data/spam.csv             # dataset
├── docs/                     # these specs
├── figures/                  # exported charts for slides and report
├── models/model3.joblib      # saved final pipeline
└── requirements.txt          # pinned dependencies
```
