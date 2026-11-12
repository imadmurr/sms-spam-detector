# SMS Spam & Phishing Detector

NLP course capstone: classify English SMS as `ham` or `spam`, and measure how much regex "fingerprint" features add on top of a bag-of-words model. Full specs: [docs/project-specs.md](docs/project-specs.md).

## Setup

Requires Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

- Notebook: open `sms_spam_detector.ipynb` and run all cells top to bottom.
- Demo app: `streamlit run app.py`

## Layout

```
data/spam.csv              # UCI SMS Spam Collection (Kaggle mirror)
docs/                      # project specs
sms_spam_detector.ipynb    # full pipeline, training and evaluation
app.py                     # Streamlit demo
models/                    # saved model + vectorizer (joblib)
figures/                   # exported charts for slides and report
requirements.txt           # pinned dependencies
```

## Data credit

UCI SMS Spam Collection by Tiago A. Almeida and José María Gómez Hidalgo (2011), licensed CC BY 4.0.
https://archive.ics.uci.edu/dataset/228/sms+spam+collection
