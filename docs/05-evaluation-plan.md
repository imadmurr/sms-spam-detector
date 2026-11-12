# 5. Evaluation Plan

*How we prove it works · based on an ML experiment plan*

Spam F1 is the headline metric, because accuracy is misleading here: predicting "ham" for everything already scores about 87% accuracy while catching zero spam.

## Metrics

| Metric | Why it matters |
| --- | --- |
| Spam recall | Share of spam we catch; missed scams reach the user |
| Spam precision | Share of flagged messages that really are spam; false alarms block real messages |
| Spam F1 | Balance of the two; used to rank models |
| Confusion matrix | Shows the actual counts of each kind of mistake |
| Accuracy | Reported for completeness, never used alone |

## Protocol

1. Baseline: always predict ham (`DummyClassifier`)
2. Compare Models 0–3 with stratified 5-fold cross-validation on the training set
3. Ablation on Model 3: remove one NLP step at a time and re-run cross-validation
4. Evaluate each model once on the held-out test set for the final results table
5. Error analysis: read the false positives and false negatives of the best model and note patterns for the report

## Acceptance targets

| Model | Target on test set | Result |
| --- | --- | --- |
| Best of Models 1–3 | Spam F1 ≥ 0.90, spam precision ≥ 0.95 | Met: Model 3, F1 0.952, precision 0.983 |
| Model 0 (regex rules) | Clearly above baseline, spam F1 ≥ 0.70 | Met: F1 0.848 |
| Ablation | Report every difference, whether a step helps or not | Reported below |

## Results (test set, 1,032 messages)

| Model | Precision | Recall | F1 | Accuracy |
| --- | --- | --- | --- | --- |
| Baseline (always ham) | 0.000 | 0.000 | 0.000 | 0.876 |
| 0: regex rules | 0.990 | 0.742 | 0.848 | 0.967 |
| 1: BoW + NB | 0.929 | 0.922 | 0.925 | 0.982 |
| 1b: TF-IDF + NB | 0.989 | 0.734 | 0.843 | 0.966 |
| 2: TF-IDF + SVM | 0.945 | 0.938 | 0.941 | 0.985 |
| **3: TF-IDF + regex + POS, LogReg** | **0.983** | **0.922** | **0.952** | **0.988** |

Model 3 makes 2 false alarms and misses 10 spam messages out of 1,032.

## Ablation (Model 3, cross-validated spam F1 on train)

| Run | Spam F1 | Change vs full |
| --- | --- | --- |
| Full pipeline | 0.941 | — |
| Without regex features | 0.925 | −0.016 |
| Without regex + POS (words only) | 0.924 | −0.017 |
| Without POS features | 0.941 | 0.000 |
| Without slang map | 0.940 | −0.001 |
| Without stopword removal | 0.944 | +0.003 |
| Without lemmatization | 0.939 | −0.002 |
| Without any cleaning (raw text into TF-IDF) | 0.945 | +0.004 |

The regex fingerprints are the step that clearly helps. The text-processing steps change F1 by less than the fold-to-fold variation (±0.018), so they neither help nor hurt this classifier noticeably.
