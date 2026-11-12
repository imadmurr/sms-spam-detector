# 6. Delivery Plan

*Who does what, when, and what we hand in · based on a project plan + risk log*

Coding runs Thursday to Saturday; deliverables close on Sunday 15 November.

## Schedule

| Day | Data & regex lead | Modelling lead |
| --- | --- | --- |
| Tue 10 Nov | Specs locked | Specs locked |
| Thu 12 Nov | Load, dedupe, EDA, regex features, Model 0 (FR-1, FR-2) | Notebook setup, tokenizer, stopwords, stemmer/lemmatizer comparison (FR-5 to FR-7) |
| Fri 13 Nov | Cleaning, slang map, word clouds, n-grams (FR-3, FR-4, FR-9, FR-10) | POS features, TF-IDF, Models 1–3, metrics (FR-8, FR-11, FR-12, FR-14) |
| Sat 14 Nov | Error analysis, report draft | Ablation, save model, Streamlit app (FR-13, FR-15) |
| Sun 15 Nov | Slides, rehearsal, submit | Slides, local scam tests (FR-16), submit |

## Slide outline (5 slides)

1. Problem: SMS scams and why regex suits them
2. Data and cleaning: dataset facts plus one message before and after cleaning
3. Regex fingerprints: feature table and a spam-vs-ham bar chart
4. Results: model comparison table and the best confusion matrix
5. Insights: word clouds, live Streamlit demo, limitations and next steps

## One-page report outline

1. Problem and goal (2 lines)
2. Data (2 lines)
3. Method: regex features, cleaning, processing, models (5 lines)
4. Results table (from the [evaluation plan](05-evaluation-plan.md))
5. Insights from error analysis (3 bullets)
6. Limitations: English only, older UK data, classic models (2 lines)

## Risks

| Risk | Fallback |
| --- | --- |
| NLTK data downloads fail on a machine | Download `punkt_tab`, `stopwords`, `wordnet`, `averaged_perceptron_tagger_eng` once at the notebook top |
| Library version clashes between teammates | One shared `requirements.txt`; Python 3.11 |
| Regex patterns over-match (e.g. short codes vs years) | Test each pattern on spam and ham samples before using it |
| Work runs late | Drop Could items first (FR-16 to FR-18), then Should items (FR-4, FR-8, FR-15) |
