# 2. Requirements Spec

*What the system must do · based on IEEE 830 SRS, trimmed*

**Priority:** **Must** = needed to pass · **Should** = strong added value · **Could** = only if time allows.

## Functional requirements

| ID | Requirement | Topic | Priority | Status |
| --- | --- | --- | --- | --- |
| FR-1 | Load the dataset, fix encoding, drop empty columns and duplicates | Data | Must | Done |
| FR-2 | Extract at least 7 regex features from raw text: URLs, phone numbers, short codes, money, CAPS words, urgency words, exclamations | Regex | Must | Done |
| FR-3 | Clean text: fix encoding/Unicode noise, unescape HTML, replace URLs/numbers/money with placeholder tokens, lowercase (after CAPS are counted), remove punctuation, collapse repeated letters | Regex + noise removal | Must | Done |
| FR-4 | Normalize SMS slang with a regex/dictionary map (u → you, txt → text, 2 → to, fr33 → free) | Normalization | Should | Done |
| FR-5 | Tokenize with NLTK `word_tokenize` | Tokenization | Must | Done |
| FR-6 | Remove NLTK stopwords, keeping negations (not, no, nor, never) | Stopwords | Must | Done |
| FR-7 | Compare Porter, Lancaster and WordNet lemmatizer (with and without POS); pick one for the pipeline | Stemming / lemmatization | Must | Done |
| FR-8 | POS-tag messages; derive 3 POS ratio features (verbs VB, numbers CD, proper nouns NNP) | POS tagging | Should | Done |
| FR-9 | Show top unigrams and bigrams per class | N-grams | Must | Done |
| FR-10 | Word frequencies and word clouds per class, before and after cleaning | Text exploration | Must | Done |
| FR-11 | Represent text with bag-of-words and TF-IDF (unigrams + bigrams) | Representation | Must | Done |
| FR-12 | Train Model 0 (regex rules), Multinomial NB, Linear SVM, and Logistic Regression on TF-IDF + regex/POS features | Classification | Must | Done |
| FR-13 | Ablation: retrain the final model with each NLP step removed, one at a time | Evaluation | Must | Done |
| FR-14 | Report precision, recall, F1 and a confusion matrix per model | Evaluation | Must | Done |
| FR-15 | `predict(text)` served through the Streamlit app, showing spam probability and which fingerprints fired | Demo | Should | Done |
| FR-16 | Test on a few hand-written local scam-style messages | Demo | Could | Done (app examples) |
| FR-17 | LDA topics of spam messages (prizes, loans, ringtones…) for an insight slide | Topic modelling | Could | Dropped |
| FR-18 | Levenshtein feature: words within 1 edit of a spam lexicon (catches "fr3e") | Text similarity | Could | Dropped |

## Non-functional requirements

| ID | Requirement |
| --- | --- |
| NFR-1 | Reproducible: fixed `random_state=42`, pinned library versions in `requirements.txt` |
| NFR-2 | Runs top to bottom in under 5 minutes on a normal laptop |
| NFR-3 | Readable: every step has a one-line explanation |
| NFR-4 | Only libraries covered in the course materials (see the [technical design](04-technical-design.md)), plus Streamlit for the demo app UI only |
| NFR-5 | No data leakage: anything fitted on data (vectorizer, scaler) is fitted on the training set only |
