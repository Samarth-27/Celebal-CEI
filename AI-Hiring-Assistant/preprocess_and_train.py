import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Load Data
logger.info("Loading JSONL dataset...")
data = []
with open(PROJECT_ROOT / "resumes_dataset.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))
df = pd.DataFrame(data)

# Text column and Label column
text_col = "Text"
label_col = "Category"

# Engineer basic numeric features
logger.info("Engineering numeric features...")
df["char_count"] = df[text_col].fillna("").apply(len)
df["word_count"] = df[text_col].fillna("").apply(lambda x: len(x.split()))
df["sentence_count"] = df[text_col].fillna("").apply(lambda x: len(x.split(".")))
df["unique_word_count"] = df[text_col].fillna("").apply(lambda x: len(set(x.split())))
df["avg_word_length"] = df["char_count"] / (df["word_count"] + 1)
df["technical_skill_count"] = df["Skills"].fillna("").apply(lambda x: len(x.split(",")))
df["education_keyword_count"] = df["Education"].fillna("").apply(lambda x: len(x.split()))
df["experience_keyword_count"] = df["Experience"].fillna("").apply(lambda x: len(x.split()))
df["certification_keyword_count"] = 0 # Dummy
df["project_keyword_count"] = 0 # Dummy

num_cols = ["char_count", "word_count", "sentence_count", "unique_word_count", "avg_word_length", 
            "technical_skill_count", "education_keyword_count", "experience_keyword_count", 
            "certification_keyword_count", "project_keyword_count"]

# Add extracted skills
df["extracted_skills"] = df["Skills"]

# Label Encoding
logger.info("Encoding labels...")
le = LabelEncoder()
df["category_encoded"] = le.fit_transform(df[label_col])
joblib.dump(le, ARTIFACTS_DIR / "label_encoder.pkl")

# Train Test Split
logger.info("Splitting data...")
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["category_encoded"])

# TF IDF
logger.info("Fitting TF-IDF...")
vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 1), min_df=5, max_df=0.8, stop_words='english')
vectorizer.fit(train_df[text_col].fillna(""))
joblib.dump(vectorizer, ARTIFACTS_DIR / "tfidf_vectorizer.pkl")

# Save metadata
metadata = {
    "text_column_for_tfidf": text_col,
    "numeric_feature_columns": num_cols,
    "label_column_encoded": "category_encoded"
}
with open(ARTIFACTS_DIR / "feature_metadata.json", "w") as f:
    json.dump(metadata, f)

# Save Data
logger.info("Saving processed datasets...")
train_df.to_csv(DATA_DIR / "train.csv", index=False)
test_df.to_csv(DATA_DIR / "test.csv", index=False)
df.to_csv(DATA_DIR / "resumes_eda_pass.csv", index=False)

logger.info("Done preprocessing. Ready for training.")

# Now we run the actual training logic
import sys
sys.path.append(str(PROJECT_ROOT / "src"))
# pyrefly: ignore [missing-import]
from train_models import load_data, prepare_features, train_and_evaluate

logger.info("Starting model training on the new dataset...")
train_df, test_df, feature_metadata, tfidf_vectorizer, label_encoder = load_data()
X_train, X_test, y_train, y_test = prepare_features(train_df, test_df, feature_metadata, tfidf_vectorizer)
best_model, best_y_pred, best_y_proba, best_model_name, results = train_and_evaluate(X_train, X_test, y_train, y_test, list(label_encoder.classes_))
logger.info(f"Training complete. Best model: {best_model_name}")

