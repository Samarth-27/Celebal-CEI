# Preprocessing Report — Module 2
## AI-Powered Intelligent Hiring Assistant (Celebal Technologies CEI Program)

**Input:** `data/processed/resumes_eda_pass.csv` (Module 1 output, 3,500 rows)
**Output:** `data/processed/{clean_resumes,train,test}.csv`, `artifacts/*.pkl`, `artifacts/feature_metadata.json`
**Notebook:** `notebooks/02_Data_Preprocessing.ipynb`

---

## 1. Deduplication

Module 1 flagged 196 rows with duplicate `Text`. Investigating further:

| Group type | Groups | Rows |
|---|---|---|
| Same text, same `Category` (pure redundancy) | 108 | 227 |
| Same text, **conflicting** `Category` (label noise) | 67 | 144 |

**Policy applied:**
- Same-label duplicates → keep first occurrence, drop the rest (119 rows dropped).
- Conflicting-label duplicates → drop **all** copies, since the label can't be trusted for either
  (144 rows dropped).

**Result:** 3,500 → 3,237 rows (263 dropped, 7.5% of the dataset).

## 2. Placeholder Handling

ResumeAtlas rows (67% of data) have non-informative `Skills`/`Education`/`Email`/`Location`
placeholders (Module 1 finding). Rather than delete these columns:
- Added `structured_fields_reliable` flag (`True` only for Synthetic rows).
- Extracted skills directly from resume `Text` via regex against a curated technical-skill
  vocabulary (`feature_engineering.TECH_SKILLS`, 90+ terms) — this becomes the reliable skill
  signal for **all** rows, including ResumeAtlas ones where the `Skills` column is useless.

## 3. Text Cleaning (`text_cleaner.py`)

Pipeline: unicode normalization → HTML removal → URL removal → email masking (`[EMAIL]`) →
non-printable character removal → punctuation cleanup (preserving technical characters like
`C++`, `.NET`, `Node.js`) → whitespace normalization → lowercasing.

## 4. NLP Processing (`feature_engineering.py`, spaCy `en_core_web_sm`)

Tokenization → stopword removal → lemmatization → `lemmatized_text` column (the TF-IDF input).
NER extraction utility (`extract_entities`) also built for reuse in Module 6 (Resume Parsing).

## 5. Feature Engineering

10 numeric features added per resume:

| Feature | Description |
|---|---|
| `char_count`, `word_count`, `sentence_count`, `unique_word_count`, `avg_word_length` | Length-based |
| `technical_skill_count`, `extracted_skills` | Regex-extracted skills from text |
| `education_keyword_count` | Education-related keyword matches |
| `experience_keyword_count` | Experience/seniority keyword matches |
| `certification_keyword_count` | Certification keyword matches |
| `project_keyword_count` | Project-related keyword matches |

## 6. TF-IDF Pipeline

- `max_features=5000`, `ngram_range=(1,2)`, `min_df=3`, `max_df=0.9`, `sublinear_tf=True`
- **Fit on the training set only** to avoid test-set leakage
- Saved to `artifacts/tfidf_vectorizer.pkl`

## 7. Label Encoding

`LabelEncoder` fit on the **full** category set (all 36 classes) before splitting, so no
unseen-label errors occur at test/inference time. Saved to `artifacts/label_encoder.pkl`.

## 8. Stratified Train/Test Split

80/20 split, stratified by `category_encoded`. Max class-proportion drift between train and test:
**0.0011** (essentially none).

**Why stratification matters here:** the dataset has up to a 10x class imbalance (200 vs 20
resumes) across 36 categories. A random split risks under-representing — or in the worst case,
zeroing out — small classes in the test set, which would make Module 3's evaluation metrics
unreliable. Stratification guarantees every category keeps its proportion in both splits.

## 9. Files Created

```
data/processed/clean_resumes.csv   (3,237 rows, 31 columns)
data/processed/train.csv           (2,589 rows)
data/processed/test.csv            (648 rows)
artifacts/tfidf_vectorizer.pkl
artifacts/label_encoder.pkl
artifacts/feature_metadata.json
src/preprocessing/preprocess.py
src/preprocessing/text_cleaner.py
src/preprocessing/feature_engineering.py
src/preprocessing/tfidf_pipeline.py
src/preprocessing/encoding.py
reports/figures/08_length_before_after_cleaning.png
reports/figures/09_wordcount_distribution_cleaned.png
reports/figures/10_top_cleaned_vocabulary.png
reports/figures/11_category_balance_train_test.png
reports/figures/12_tfidf_feature_importance.png
```

## 10. How This Module Improves the Dataset

- Removes 263 unreliable/duplicate/conflicting-label rows.
- Adds a trustworthy, text-derived skill signal for every resume (not just Synthetic ones).
- Adds 10 structured numeric features usable alongside or instead of TF-IDF.
- Produces leakage-safe, stratified, ML-ready train/test splits with saved, reusable artifacts.

## 11. Potential Risks

- **Fixed skill vocabulary:** `TECH_SKILLS` is a curated list (~90 terms) — it will miss skills
  outside the list and doesn't yet resolve synonyms (e.g. "Node" vs "Node.js" vs "NodeJS").
- **Source/Category confound persists:** cleaning and dedup do not remove the fact that every
  category belongs entirely to one source (Module 1 finding). Module 3 must still account for this
  in evaluation.
- **spaCy lemmatization runtime:** ~2–3 minutes for 3,237 resumes on CPU; will need `nlp.pipe`
  batching if the dataset scales up significantly.

## 12. Recommendations for Module 3

1. Use `category_encoded` as target; load `train.csv`/`test.csv` and the saved
   `tfidf_vectorizer.pkl`/`label_encoder.pkl` directly rather than refitting.
2. Report **macro-F1** alongside accuracy, given the class imbalance.
3. Check whether misclassifications correlate with `Source` to confirm the model is learning
   category content rather than source style.
4. Consider models that can combine sparse TF-IDF features with the 10 dense numeric engineered
   features (e.g. Random Forest, XGBoost) in addition to pure linear models like Logistic
   Regression/SVM on TF-IDF alone.
