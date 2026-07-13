# EDA Report — Module 1
## AI-Powered Intelligent Hiring Assistant (Celebal Technologies CEI Program)

**Dataset:** `data/raw/resumes_dataset.jsonl` — Kaggle Resume Dataset
**Records:** 3,500 | **Columns:** 12 | **Categories:** 36
**Notebook:** `notebooks/01_EDA.ipynb`

---

## 1. Schema

| Column | Type | Notes |
|---|---|---|
| ResumeID | string | unique, no duplicates |
| Category | string | 36 job-role labels (target for Module 3) |
| Name | string | synthetic/placeholder names |
| Email | string | mostly placeholder (`contact@email.com`) |
| Phone | string | garbled — appears to be concatenated dates/numbers from source scrape, not a usable phone field |
| Location | string | mostly placeholder (`City, State`) |
| Summary | string | free text, overlaps with `Text` |
| Skills | string | comma-separated, only reliable for Synthetic source |
| Experience | string | free text, overlaps with `Text` |
| Education | string | mostly placeholder for ResumeAtlas source |
| Text | string | full resume text — the one field reliable across both sources |
| Source | string | `ResumeAtlas` (2,337) or `Synthetic` (1,163) |

## 2. Data Quality

- **No null values, no empty strings, no duplicate ResumeIDs.**
- **196 rows have identical `Text`** to another row — real duplication that nulls checks miss.
- **No fully duplicate rows.**

## 3. The Central Finding: Two Merged, Structurally Different Sources

| | ResumeAtlas (n=2,337) | Synthetic (n=1,163) |
|---|---|---|
| Unique `Skills` values | 1 | 918 |
| Unique `Education` values | 1 | 1,159 |
| Unique `Email` values | 1 | 1,160 |
| Unique `Location` values | 1 | 1,163 |
| Mean text length (words) | ~540 (std 392, max 6,554) | ~190 (std 37) |
| Categories | 16, unique to this source | 20, unique to this source |

**ResumeAtlas** rows only carry genuine per-candidate signal in the `Text` field — every other
structured column is a copy-pasted placeholder. **Synthetic** rows have realistic, diverse structured
fields but are short and template-generated.

**Category and Source have zero overlap** — every category belongs entirely to one source or the
other. This is the single most important fact for later modules: a classifier can reach high accuracy
by learning source style (text length, phrasing patterns) rather than actual job-category content.
Module 3 evaluation needs to explicitly check for this rather than trusting raw accuracy.

## 4. Category Distribution

36 categories, ranging from 200 resumes (Java Developer, Python Developer, Data Science) down to 20
(Technical Writer) — roughly a 10x class imbalance. See `figures/01_category_distribution.png`.

## 5. Resume Length

- Overall: median 257 words, mean 424, max 6,554 (heavy right skew driven by ResumeAtlas).
- ResumeAtlas and Synthetic have very different length profiles (see above) — this needs to be
  neutralized before it becomes a spurious classification signal.
- See `figures/03_text_length_distribution.png` and `figures/04_median_length_by_category.png`.

## 6. Skills

Top skills in the reliable (Synthetic) subset: **Git, Python, Linux, Docker, AWS, Agile, SQL,
JavaScript, REST API, Scrum, Microservices, PostgreSQL, Node.js, React, Java, MongoDB.**
See `figures/05_top_skills_synthetic.png`.

## 7. Word Frequency

See `figures/06_wordcloud.png` for the most common terms across sampled resume text after stopword
removal.

## 8. Education

Degree levels for the Synthetic subset (Bachelor/Master/PhD/Associate) — see
`figures/07_degree_level.png`. Not computed for ResumeAtlas since its `Education` field is a
placeholder.

## 9. Recommendations Carried Into Module 2

1. Deduplicate the 196 identical-`Text` rows (decide on drop vs. flag-and-keep).
2. Do not trust `Skills`/`Education`/`Location`/`Email` for ResumeAtlas rows — extract structured data
   from `Text` directly, or engineer a `skills_field_reliable` flag (already added in the EDA-pass CSV).
3. Address the length gap between sources before vectorizing, so length isn't a proxy for category.
4. Use stratified splitting for the 36-class, imbalanced target, and prefer macro-F1 over raw accuracy
   as the primary Module 3 metric given the source/category confound.

## Files Produced By This Module

- `notebooks/01_EDA.ipynb` — full executed analysis
- `reports/EDA_Report.md` — this report
- `reports/figures/*.png` — 7 charts
- `data/processed/resumes_eda_pass.csv` — original data + `is_duplicate_text` and
  `skills_field_reliable` flags (no destructive cleaning yet)
