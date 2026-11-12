# SMS Spam & Phishing Detector — Project Specs

Nov 9, 2026

| # | Document | Answers | Based on |
| --- | --- | --- | --- |
| 1 | Project Charter | Why, what, who, by when | PMI project charter |
| 2 | Requirements Spec | What the system must do | IEEE 830 SRS, trimmed |
| 3 | Data Spec | What data, how it looks, how it is split | Datasheets for Datasets / Data Cards |
| 4 | Technical Design | How the pipeline is built | ML design doc |
| 5 | Evaluation Plan | How we prove it works | ML experiment plan |
| 6 | Delivery Plan | Who does what, when, and what we hand in | Project plan + risk log |

## 1. Project Charter

We build an SMS spam and phishing detector that shows how far regex-based text features go compared with a standard bag-of-words model. Final hand-in is Sunday 15 November 2026.

**Problem.** SMS scams (fake prizes, parcel fees, bank alerts) reach phones daily. Spammers leave fingerprints in the text: links, short codes, money amounts, CAPS and urgency words. Regex can catch these cheaply.

**Goal.** Classify an SMS as `ham` or `spam`, and measure the value of regex features versus plain word counts.

**In scope**

- English SMS from the UCI SMS Spam Collection
- Regex feature extraction, text cleaning, tokenization, stopword removal, stemming and lemmatization
- Classic classifiers from scikit-learn (Naive Bayes, Linear SVM, Logistic Regression) plus a regex rule baseline
- Word clouds, n-gram analysis, a small prediction demo
- A deployed app with a clean user interface built with Streamlit (app.py), used as the live demo

**Out of scope**

- Arabic, French or Arabizi messages
- Deep learning or transformer models

**Deliverables**

| Deliverable | Format | Limit |
| --- | --- | --- |
| Code | One Jupyter notebook, Streamlit app, `requirements.txt` | Runs top to bottom without errors |
| Report | Document exported to PDF | 1 page |
| Presentation | PowerPoint | 5 slides |

**Success criteria**

- Every core course requirement (regex, text processing, text cleaning) is visibly used in the code
- The best model beats the always-ham baseline on spam F1 (target in section 5)
- All three deliverables submitted on time

## 2. Requirements Spec

Priority: \
**Must** = needed to pass.\
**Should** = strong added value.\
**Could** = only if time allows.

**Functional requirements**

| ID | Requirement | &#32;Topic | Priority |
| --- | --- | --- | --- |
| FR-1 | Load the dataset, fix encoding, drop empty columns and duplicates | Data | Must |
| FR-2 | Extract at least 7 regex features from raw text: URLs, phone numbers, short codes, money, CAPS words, urgency words, exclamations | Regex | Must |
| FR-3 | Clean text: fix encoding/Unicode noise, unescape HTML, replace URLs/numbers/money with placeholder tokens, lowercase (after CAPS are counted), remove punctuation, collapse repeated letters | Regex + noise removal | Must |
| FR-4 | Normalize SMS slang with a regex/dictionary map (u → you, txt → text, 2 → to, fr33 → free) | Normalization | Should |
| FR-5 | Tokenize with NLTK `word_tokenize` | Tokenization | Must |
| FR-6 | Remove NLTK stopwords, keeping negations (not, no, never) | Stopwords | Must |
| FR-7 | Compare Porter, Lancaster and WordNet lemmatizer (with and without POS); pick one for the pipeline | Stemming / lemmatization | Must |
| FR-8 | POS-tag messages; derive 2–3 POS ratio features (verbs VB, numbers CD, proper nouns NNP) | POS tagging | Should |
| FR-9 | Show top unigrams and bigrams per class | N-grams | Must |
| FR-10 | Word frequencies and word clouds per class, before and after cleaning | Text exploration | Must |
| FR-11 | Represent text with bag-of-words and TF-IDF (unigrams + bigrams) | Representation | Must |
| FR-12 | Train Model 0 (regex rules), Multinomial NB, Linear SVM, and Logistic Regression on TF-IDF + regex/POS features | Classification | Must |
| FR-13 | Ablation: retrain the final model with each NLP step removed, one at a time | Evaluation | Must |
| FR-14 | Report precision, recall, F1 and a confusion matrix per model | Evaluation | Must |
| FR-15 | `predict(text)` served through the Streamlit app, showing spam probability and which fingerprints fired | Demo | Should |
| FR-16 | Test on a few hand-written local scam-style messages | Demo | Could |
| FR-17 | LDA topics of spam messages (prizes, loans, ringtones…) for an insight slide | Topic modelling | Could |
| FR-18 | Levenshtein feature: words within 1 edit of a spam lexicon (catches "fr3e") | Text similarity | Could |

**Non-functional requirements**

| ID | Requirement |
| --- | --- |
| NFR-1 | Reproducible: fixed `random_state=42`, pinned library versions in `requirements.txt` |
| NFR-2 | Runs top to bottom in under 5 minutes on a normal laptop |
| NFR-3 | Readable: every step has a one-line explanation |
| NFR-4 | Only libraries covered in the course materials (see section 4), plus Streamlit for the demo app UI only |
| NFR-5 | No data leakage: anything fitted on data (vectorizer, scaler) is fitted on the training set only |

## 3. Data Spec

We use the UCI SMS Spam Collection: about 5,570 English SMS, roughly 13% spam, already labelled. Counts below are approximate and get confirmed in the EDA step.

| Field | Value |
| --- | --- |
| Source | [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection); Kaggle mirror `spam.csv` |
| License | CC BY 4.0 (credit Almeida & Hidalgo, 2011) |
| Size | \~5,572 messages: \~4,825 ham, \~747 spam |
| Language | English, mostly UK, collected around 2011 |
| Raw columns | `v1` = label, `v2` = text, plus 3 mostly empty `Unnamed` columns |
| Encoding | `latin-1` |

**Working schema after loading**

| Column | Type | Meaning |
| --- | --- | --- |
| `label` | str | `ham` or `spam` |
| `y` | int | 0 = ham, 1 = spam |
| `text` | str | Raw message, never modified |
| `clean_text` | str | Output of the cleaning step |
| `tokens` | list | Tokenized, stopwords removed, lemmatized |
| regex feature columns | int | One count per regex feature (section 4) |

**Known noise to handle**

- Encoding artefacts such as `Â£` instead of `£`
- HTML entities such as `&lt;` and `&amp;`
- SMS slang (`u`, `ur`, `txt`, `2morrow`) and repeated letters (`sooo`)
- About 400 duplicate messages

**Decisions to lock**

- [x] Drop exact duplicates before splitting, so the same message never lands in both train and test
- [x] Split 80/20 train/test, stratified on `y`, `random_state=42`
- [x] The test set is touched only once, for the final results table

## 4. Technical Design

Raw text splits into two branches: regex features are extracted from the untouched text, while a cleaned copy feeds the bag-of-words. The key rule: regex features come first, because cleaning deletes the very fingerprints they detect.

&#91;embedded content: pipeline · 2 branches, 4 models\]

Model 0 uses only the regex branch, Models 1–2 only the processed-text branch, and Model 3 combines both.

**Pipeline steps**

1. Load, deduplicate, split train/test
2. Extract regex features from raw `text`; Model 0 rules run on these
3. Clean `text` into `clean_text` with regex, then apply the slang map
4. Tokenize, remove stopwords (keep negations), POS-tag, lemmatize with POS
5. Compute POS ratio features from the tags
6. Vectorize with TF-IDF, unigrams + bigrams (fitted on train only)
7. Train Models 0–3, then run the ablation on Model 3
8. Evaluate, plot, save the final model, serve it in the Streamlit app

**Regex feature catalogue** (each pattern produces one count column)

```python
FEATURES = {
    "n_urls":    r"https?://\S+|www\.\S+|\b\w+\.(?:com|net|org|co\.uk)\b",
    "n_phones":  r"\+?\d[\d\s-]{8,}\d",
    "n_shortcodes": r"\b\d{5}\b",
    "n_money":   r"[£$€]\s?\d+|\b\d+\s?(?:p|pounds?)\b",
    "n_caps":    r"\b[A-Z]{2,}\b",
    "n_urgent":  r"(?i)\b(?:free|win|winner|won|claim|prize|urgent|txt|reply|stop|call now)\b",
    "n_exclaim": r"!",
}
# plus: n_chars, n_words, digit_ratio
```

**Cleaning rules, in order**

1. Fix encoding: `Â£` → `£`; `html.unescape()` for `&lt;` and similar; strip leftover non-ASCII symbols
2. Replace URLs with `URL`, money with `MONEY`, phone numbers with `PHONE`, other numbers with `NUM`
3. Lowercase (regex CAPS feature is already counted)
4. Remove punctuation: `re.sub(r"[^a-z\s]", " ", text)`
5. Collapse letters repeated 3+ times: `re.sub(r"(\w)\1{2,}", r"\1\1", text)`
6. Slang map with word-boundary regex: `\bu\b` → you, `\bur\b` → your, `\btxt\b` → text, `\b2\b` → to, `fr[e3]{2}` → free
7. Collapse whitespace

**Models**

| Model | Input | Classifier | Shows |
| --- | --- | --- | --- |
| 0 | Regex fingerprints | Hand-written rule: spam if ≥ k fingerprints fire (k tuned on train) | How far pure rules go, no ML |
| 1 | TF-IDF words | Multinomial Naive Bayes | The classic text baseline |
| 2 | TF-IDF words | Linear SVM | Usually the strongest classic text model |
| 3 | TF-IDF + regex + POS features | Logistic Regression, `class_weight='balanced'` | Final model: NLP features on top of words; probabilities and readable weights for the app |

Not used: GaussianNB (assumes normally distributed features, poor fit for word counts); LSTM and BERT (out of scope, listed as next steps).

**Tech stack**

| Purpose | Library |
| --- | --- |
| Data | pandas, numpy |
| Regex and cleaning | `re`, `html` |
| Text processing | NLTK (`word_tokenize`, `stopwords`, `WordNetLemmatizer`, `PorterStemmer`, `pos_tag`, `ngrams`) |
| Features and models | scikit-learn, scipy |
| Plots | matplotlib, seaborn, wordcloud |

Demo app: Streamlit, UI only. `app.py` loads the trained model and pipeline saved from the notebook (`joblib`), and shows the prediction plus which regex fingerprints fired.

**Repository layout**

```
sms-spam-nlp/
├── data/spam.csv
├── docs/             # project specs (.md)
├── sms_spam_detector.ipynb
├── app.py            # Streamlit demo
├── models/           # saved model + vectorizer (joblib)
├── requirements.txt
├── figures/          # exported charts for slides and report
└── README.md         # how to run
```

## 5. Evaluation Plan

Spam F1 is the headline metric, because accuracy is misleading here: predicting "ham" for everything already scores about 87% accuracy while catching zero spam.

**Metrics**

| Metric | Why it matters |
| --- | --- |
| Spam recall | Share of spam we catch; missed scams reach the user |
| Spam precision | Share of flagged messages that really are spam; false alarms block real messages |
| Spam F1 | Balance of the two; used to rank models |
| Confusion matrix | Shows the actual counts of each kind of mistake |
| Accuracy | Reported for completeness, never used alone |

**Protocol**

1. Baseline: always predict ham (`DummyClassifier`)
2. Compare Models 0–3 with stratified 5-fold cross-validation on the training set
3. Ablation on Model 3: remove one NLP step at a time and re-run cross-validation (table below)
4. Evaluate each model once on the held-out test set for the final results table
5. Error analysis: read 10 false positives and 10 false negatives from the best model, and note patterns for the report

**Acceptance targets** (proposed; adjust before locking)

| Model | Target on test set |
| --- | --- |
| Best of Models 1–3 | Spam F1 ≥ 0.90, spam precision ≥ 0.95 |
| Model 0 (regex rules) | Clearly above baseline, spam F1 ≥ 0.70 |
| Ablation | Report every difference, whether a step helps or not; both outcomes are valid findings |

**Results table to fill**

| Model | Precision | Recall | F1 | Accuracy |
| --- | --- | --- | --- | --- |
| Baseline (always ham) |  |  |  |  |
| 0: regex rules |  |  |  |  |
| 1: TF-IDF + NB |  |  |  |  |
| 2: TF-IDF + SVM |  |  |  |  |
| 3: TF-IDF + regex + POS, LogReg |  |  |  |  |

**Ablation table to fill** (Model 3, cross-validated spam F1)

| Run | Spam F1 | Change vs full |
| --- | --- | --- |
| Full pipeline |  | — |
| Without regex features |  |  |
| Without POS features |  |  |
| Without slang map |  |  |
| Without stopword removal |  |  |
| Without lemmatization |  |  |
| Without any cleaning (raw text into TF-IDF) |  |  |

## 6. Delivery Plan

Specs are locked; coding runs Thursday to Saturday and deliverables close on Sunday 15 November, with no buffer day left.

**Schedule**

| Day | Data & regex lead | Modelling lead |
| --- | --- | --- |
| Tue 10 Nov | Specs locked | Specs locked |
| Thu 12 Nov | Load, dedupe, EDA, regex features, Model 0 (FR-1, FR-2) | Notebook setup, tokenizer, stopwords, stemmer/lemmatizer comparison (FR-5 to FR-7) |
| Fri 13 Nov | Cleaning, slang map, word clouds, n-grams (FR-3, FR-4, FR-9, FR-10) | POS features, TF-IDF, Models 1–3, metrics (FR-8, FR-11, FR-12, FR-14) |
| Sat 14 Nov | Error analysis, report draft | Ablation, save model, Streamlit app (FR-13, FR-15) |
| Sun 15 Nov | Slides, rehearsal, submit | Slides, local scam tests (FR-16), submit |

**Slide outline (5 slides)**

1. Problem: SMS scams and why regex suits them
2. Data and cleaning: dataset facts plus one message before and after cleaning
3. Regex fingerprints: feature table and a spam-vs-ham bar chart
4. Results: model comparison table and the best confusion matrix
5. Insights: word clouds, live Streamlit demo, limitations and next steps

**One-page report outline**

1. Problem and goal (2 lines)
2. Data (2 lines)
3. Method: regex features, cleaning, processing, models (5 lines)
4. Results table (from section 5)
5. Insights from error analysis (3 bullets)
6. Limitations: English only, older UK data, classic models (2 lines)

**Risks**

| Risk | Fallback |
| --- | --- |
| NLTK data downloads fail on a machine | Download `punkt`, `stopwords`, `wordnet`, `averaged_perceptron_tagger` once at the notebook top; share the folder |
| Library version clashes between teammates | One shared `requirements.txt`; same Python version |
| Regex patterns over-match (e.g. short codes vs years) | Test each pattern on 5 spam and 5 ham samples before using it |
| Work runs late | Drop Could items first (FR-16 to FR-18), then Should items (FR-4, FR-8, FR-15) |

**Sign-off**

- [x] All six specs reviewed by every team member
- [x] Team names filled in section 1
- [x] Acceptance targets agreed in section 5
