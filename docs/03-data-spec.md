# 3. Data Spec

*What data, how it looks, how it is split · based on Datasheets for Datasets / Data Cards*

We use the UCI SMS Spam Collection: English SMS messages, already labelled `ham` or `spam`.

| Field | Value |
| --- | --- |
| Source | [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection); Kaggle mirror `spam.csv` |
| License | CC BY 4.0 (credit Almeida & Hidalgo, 2011) |
| Raw size | 5,572 messages: 4,825 ham, 747 spam |
| After deduplication | 5,158 messages: 4,516 ham, 642 spam (12.4% spam) |
| Language | English, mostly UK, collected around 2011 |
| Raw columns | `v1` = label, `v2` = text, plus 3 mostly empty `Unnamed` columns |
| Encoding | `latin-1` |

## Working schema after loading

| Column | Type | Meaning |
| --- | --- | --- |
| `label` | str | `ham` or `spam` |
| `y` | int | 0 = ham, 1 = spam |
| `text` | str | Raw message, never modified |
| `clean_text` | str | Output of the cleaning step |
| `tokens` | list | Tokenized, stopwords removed, lemmatized |
| `doc` | str | `tokens` joined back into one string, the input to the vectorizers |
| regex feature columns | int / float | 7 fingerprint counts plus `n_chars`, `n_words`, `digit_ratio` |
| POS feature columns | float | `pos_vb`, `pos_cd`, `pos_nnp` ratios |

## Known noise

- Encoding artefacts such as `å£` instead of `£` (when read as `latin-1`)
- HTML entities such as `&lt;` and `&amp;`
- SMS slang (`u`, `ur`, `txt`, `2morrow`) and repeated letters (`sooo`)
- 414 duplicate messages (403 exact, 11 more after trimming whitespace); no message appears with both labels

## Decisions

- Drop duplicates (compared on trimmed text) before splitting, so the same message never lands in both train and test
- Split 80/20 train/test, stratified on `y`, `random_state=42`: 4,126 train / 1,032 test
- The test set is touched only once, for the final results table
