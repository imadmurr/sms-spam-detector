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
| Best of Models 1–3 | Spam F1 ≥ 0.90, spam precision ≥ 0.95 | Met: Model 3, F1 0.956, precision 0.983 |
| Model 0 (regex rules) | Clearly above baseline, spam F1 ≥ 0.70 | Met: F1 0.848 |
| Ablation | Report every difference, whether a step helps or not | Reported below |

## Results (test set, 1,032 messages)

| Model | Precision | Recall | F1 | Accuracy |
| --- | --- | --- | --- | --- |
| Baseline (always ham) | 0.000 | 0.000 | 0.000 | 0.876 |
| 0: regex rules | 0.990 | 0.742 | 0.848 | 0.967 |
| 1: BoW + NB | 0.937 | 0.922 | 0.929 | 0.983 |
| 1b: TF-IDF + NB | 1.000 | 0.688 | 0.815 | 0.961 |
| 2: TF-IDF + SVM | 0.953 | 0.945 | 0.949 | 0.987 |
| **3: TF-IDF + regex + POS, LogReg** | **0.983** | **0.930** | **0.956** | **0.989** |

Model 3 makes 2 false alarms and misses 9 spam messages out of 1,032.

Training set: 4,126 original messages + 300 synthetic ham (see note below).

## Ablation (Model 3, cross-validated spam F1 on train)

| Run | Spam F1 | Change vs full |
| --- | --- | --- |
| Full pipeline | 0.937 | — |
| Without regex features | 0.926 | −0.011 |
| Without regex + POS (words only) | 0.927 | −0.010 |
| Without POS features | 0.937 | 0.000 |
| Without slang map | 0.938 | +0.001 |
| Without stopword removal | 0.947 | +0.010 |
| Without lemmatization | 0.931 | −0.006 |
| Without any cleaning (raw text into TF-IDF) | 0.947 | +0.010 |

The regex patterns are the step that clearly helps. The text-processing steps change F1 by less than the fold-to-fold variation (±0.019), so they neither help nor hurt this classifier noticeably.

## Synthetic ham augmentation

The 2011 UCI dataset contains no ham messages with phone numbers (n_phones hit rate: 0.000 for ham, 0.576 for spam). This caused the original model to score ordinary messages like "My new number is 07700900123, save it" near 100% spam. Additionally, the short-code regex (`\b\d{5}\b`) matched the first half of a spaced phone number (e.g. "07700" in "07700 900123"), double-counting one phone as two spam signals.

Fixes applied:
1. `regex_features()` now counts `n_shortcodes` on the text with phone-number matches blanked out first, eliminating the overlap.
2. 300 real ham messages from the training set were augmented with a randomly generated UK mobile number appended via five natural templates (e.g. "call me on {n}", "my new number is {n}"). Numbers follow the 07XXXXXXXXX format; half are written with a space after the fifth digit. The test set was not touched.

After these fixes the three example messages score: "call me if late" 0.5% spam, "call me on 07700 900123 if late" 17.3% spam, "My new number is 07700900123" 0.0% spam, while "URGENT! You won £1000. Call 09061701461 to claim" and "Text WIN to 87121 for your FREE prize" both score 100% spam.
