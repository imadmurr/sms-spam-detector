# SMS Spam Detector

Classify English SMS messages as **spam** or **ham** (legitimate) with classic NLP, and measure how much hand-written regex "fingerprints" add on top of a bag-of-words model.

**Live demo:** https://sms-spam-detector-2.streamlit.app/

NLP course capstone · Python 3.11 · NLTK · scikit-learn · Streamlit

## Results

Test set of 1,032 messages, never used during training or tuning:

| Model | Input | Precision | Recall | Spam F1 |
| --- | --- | --- | --- | --- |
| Baseline (always ham) | — | 0.000 | 0.000 | 0.000 |
| 0: Regex rules | 7 regex fingerprints only | 0.990 | 0.742 | 0.848 |
| 1: Naive Bayes | Bag-of-words | 0.929 | 0.922 | 0.925 |
| 2: Linear SVM | TF-IDF | 0.945 | 0.938 | 0.941 |
| **3: Logistic Regression** | **TF-IDF + regex + POS features** | **0.983** | **0.922** | **0.952** |

- **Regex fingerprints alone** catch three quarters of the spam with almost no false alarms.
- **Combined with words**, they give the best model. Removing them from Model 3 lowers cross-validated F1 by 0.016.
- **Cleaning, stopword removal and lemmatization** change F1 by less than ±0.004, within the noise between folds.

Full tables, the ablation and the targets: [docs/05-evaluation-plan.md](docs/05-evaluation-plan.md).

## How it works

```
raw SMS ──┬──► regex fingerprints + POS ratios        (computed on the raw text)
          └──► clean → tokenize → stopwords → lemmatize → TF-IDF
                                    │
                     both feed Logistic Regression → spam probability
```

1. **Regex fingerprints** are counted on the raw text: links, phone numbers, 5-digit short codes, money amounts, CAPS words, urgency words and exclamation marks. They come first, because cleaning would delete them.
2. **Cleaning:**
   - fix encoding and HTML entities;
   - replace URLs, money, phone numbers and other numbers with placeholder tokens;
   - map SMS slang (`u` → you, `fr33` → free);
   - lowercase, and strip punctuation.
3. **Token processing:** NLTK tokenization, POS tagging, stopword removal (negations are kept), and WordNet lemmatization guided by the POS tags.
4. **Modelling:**
   - TF-IDF on unigrams and bigrams, plus the scaled regex and POS features;
   - one scikit-learn pipeline, fitted on the training set only.

The feature code lives in [`features.py`](features.py) and is shared by the notebook and the app, so the demo builds exactly the same input the model was trained on.

## Quick start

```bash
git clone https://github.com/imadmurr/sms-spam-detector.git
cd sms-spam-detector
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Run the demo app:**

```bash
streamlit run app.py
```

**Reproduce the analysis:** open `sms_spam_detector.ipynb` and run all cells, or run it headless:

```bash
jupyter nbconvert --to notebook --execute --inplace sms_spam_detector.ipynb
```

The notebook takes about 30 seconds and:
- downloads the NLTK data it needs;
- regenerates the charts in `figures/`;
- re-saves the model to `models/model3.joblib`.

## Project structure

```
├── app.py                    # Streamlit demo app
├── features.py               # regex features, cleaning, tokens, POS, model input (shared)
├── sms_spam_detector.ipynb   # full pipeline: EDA, features, models, evaluation, ablation
├── assets/style.css          # app styling
├── .streamlit/config.toml    # app theme
├── data/spam.csv             # UCI SMS Spam Collection
├── docs/                     # project specs (charter, requirements, data, design, evaluation, delivery)
├── figures/                  # charts used in the report and slides
├── models/model3.joblib      # trained final pipeline
└── requirements.txt          # pinned dependencies
```

## Deployment

The demo runs on Streamlit Community Cloud from the `main` branch (main file `app.py`, Python 3.11) and redeploys on every push.

## Limitations

- English only, trained on UK messages from around 2011.
- Modern scams with no links, numbers or money can slip through, e.g. "Hi mum, this is my new number…" and job offers. The app shows both as examples.
- Classic models only; transformers are out of scope.

## Data and credits

UCI SMS Spam Collection by Tiago A. Almeida and José María Gómez Hidalgo (2011), licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Source: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
