# 1. Project Charter

*Why, what, who, by when · based on the PMI project charter*

We build an SMS spam and phishing detector that shows how far regex-based text features go compared with a standard bag-of-words model. Final hand-in: **Sunday 15 November 2026**.

## Problem

SMS scams (fake prizes, parcel fees, bank alerts) reach phones daily. Spammers leave fingerprints in the text: links, short codes, money amounts, CAPS and urgency words. Regex can catch these cheaply.

## Goal

Classify an SMS as `ham` or `spam`, and measure the value of regex features versus plain word counts.

## Scope

**In scope**

- English SMS from the UCI SMS Spam Collection
- Regex feature extraction, text cleaning, tokenization, stopword removal, stemming and lemmatization
- Classic scikit-learn classifiers (Naive Bayes, Linear SVM, Logistic Regression) plus a regex rule baseline
- Word clouds, n-gram analysis and a prediction demo
- A deployed Streamlit app (`app.py`) used as the live demo

**Out of scope**

- Arabic, French or Arabizi messages
- Deep learning or transformer models

## Deliverables

| Deliverable | Format | Limit |
| --- | --- | --- |
| Code | One Jupyter notebook, Streamlit app, `requirements.txt` | Runs top to bottom without errors |
| Report | Document exported to PDF | 1 page |
| Presentation | PowerPoint | 5 slides |

## Success criteria

- Every core course requirement (regex, text processing, text cleaning) is visibly used in the code
- The best model beats the always-ham baseline on spam F1 (targets in the [evaluation plan](05-evaluation-plan.md))
- All three deliverables submitted on time
