import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

try:
    from sentence_transformers import SentenceTransformer, util
    import torch
    torch.set_num_threads(1)
except ImportError:
    pass

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"
REPORTS_DIR = PROJECT_ROOT / "reports" / "3_Semantic_Ranking"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

class SemanticMatcher:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        logger.info(f"Loading SentenceTransformer model {model_name}...")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.resume_ids = None
        self.df = None
        
    def load_data(self):
        logger.info("Loading train.csv resumes...")
        try:
            self.df = pd.read_csv(DATA_DIR / "train.csv", engine='python', on_bad_lines='skip').fillna("")
        except FileNotFoundError:
            self.df = pd.read_csv(DATA_DIR / "resumes_eda_pass.csv", engine='python', on_bad_lines='skip').fillna("")
        self.resume_ids = self.df["ResumeID"].tolist()
        
    def generate_embeddings(self, save=True):
        logger.info("Generating SentenceTransformer embeddings for resumes...")
        if "clean_text" in self.df.columns:
            texts = self.df["clean_text"].tolist()
        else:
            texts = self.df["Text"].tolist()
            
        self.embeddings = self.model.encode(texts, batch_size=8, show_progress_bar=True)
        
        if save:
            logger.info("Saving SentenceTransformer embeddings...")
            joblib.dump(self.embeddings, VECTOR_STORE_DIR / "st_resume_embeddings.pkl")
            joblib.dump(self.resume_ids, VECTOR_STORE_DIR / "st_resume_ids.pkl")
            
            model_info_path = PROJECT_ROOT / "models" / "embedding_model_info.json"
            with open(model_info_path, "w") as f:
                json.dump({"model_name": self.model_name, "type": "SentenceTransformer"}, f)
            
    def load_embeddings(self):
        logger.info("Loading saved SentenceTransformer embeddings...")
        self.embeddings = joblib.load(VECTOR_STORE_DIR / "st_resume_embeddings.pkl")
        self.resume_ids = joblib.load(VECTOR_STORE_DIR / "st_resume_ids.pkl")
        
    def match_job_description(self, jd_text, top_k=10):
        logger.info("Matching Job Description...")
        jd_embedding = self.model.encode([jd_text])
        cosine_scores = cosine_similarity(jd_embedding, self.embeddings).flatten()
        
        top_indices = cosine_scores.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            score = cosine_scores[idx]
            resume_data = self.df.iloc[idx]
            results.append({
                "ResumeID": resume_data["ResumeID"],
                "Category": resume_data.get("Category", ""),
                "SimilarityScore": score,
                "Skills": resume_data.get("extracted_skills", resume_data.get("Skills", ""))
            })
            
        return pd.DataFrame(results)

class CandidateRanker:
    def __init__(self, weights=None):
        if weights is None:
            self.weights = {
                "skill_match": 0.5,
                "semantic_similarity": 0.3,
                "experience_match": 0.1,
                "education_match": 0.1
            }
        else:
            self.weights = weights
            
    def load_custom_data(self, df):
        logger.info("Loading custom dataset and computing semantic embeddings on-the-fly...")
        self.df = df.copy().fillna("")
        
        # Truncate to prevent hanging the Streamlit app on CPU
        if len(self.df) > 100:
            self.df = self.df.head(100)
        # Ensure we have ResumeID, if not generate them
        if "ResumeID" not in self.df.columns:
            self.df["ResumeID"] = [f"Custom_Candidate_{i}" for i in range(len(self.df))]
            
        # Determine the text column
        if "clean_text" in self.df.columns:
            texts = self.df["clean_text"].astype(str).tolist()
        elif "Skills" in self.df.columns:
            texts = self.df["Skills"].astype(str).tolist()
        elif "Text" in self.df.columns:
            texts = self.df["Text"].astype(str).tolist()
        else:
            # Fallback to concatenate all columns if no obvious text column exists
            texts = self.df.apply(lambda row: " ".join(row.values.astype(str)), axis=1).tolist()
            
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embeddings = self.model.encode(texts, show_progress_bar=False)
            self.resume_ids = self.df["ResumeID"].tolist()
            
            # Make sure we maintain columns required by rank_candidates
            if "extracted_skills" not in self.df.columns:
                self.df["extracted_skills"] = self.df.get("Skills", "")
                
            self.train_df = self.df.copy()
            return True
        except Exception as e:
            logger.error(f"Error computing custom embeddings: {e}")
            return False

    def load_data(self):
        logger.info("Loading training data and semantic embeddings...")
        try:
            import joblib
            from sentence_transformers import SentenceTransformer
            import os
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
            os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
            try:
                from transformers.utils import logging as hf_logging
                hf_logging.set_verbosity_error()
                hf_logging.disable_progress_bar()
            except ImportError:
                pass
            
            VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"
            self.embeddings = joblib.load(VECTOR_STORE_DIR / "st_resume_embeddings.pkl")
            self.resume_ids = joblib.load(VECTOR_STORE_DIR / "st_resume_ids.pkl")
            
            raw_train = pd.read_csv(DATA_DIR / "train.csv", engine='python', on_bad_lines='skip')
            id_df = pd.DataFrame({"ResumeID": self.resume_ids})
            self.train_df = pd.merge(id_df, raw_train, on="ResumeID", how="left")
            
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error(f"Error loading models: {tb}")
            raise RuntimeError(f"Failed to load ranking data: {tb}")
    def compute_skill_match(self, resume_skills, required_skills):
        if not isinstance(resume_skills, str) or not resume_skills:
            return 0.0
        r_skills = set([s.strip().lower() for s in resume_skills.split(",")])
        req_skills = set([s.strip().lower() for s in required_skills])
        if not req_skills:
            return 0.0
        return len(r_skills.intersection(req_skills)) / len(req_skills)
        
    def rank_candidates(self, required_skills, recruiter_jd_text=None, required_experience_years=3, required_education="Bachelor"):
        logger.info("Computing dynamic ranking scores...")
        
        if self.train_df is None:
            return pd.DataFrame()
            
        self.df = self.train_df.copy()
        
        if recruiter_jd_text and self.model is not None and self.embeddings is not None:
            from sklearn.metrics.pairwise import cosine_similarity
            jd_embedding = self.model.encode([recruiter_jd_text], show_progress_bar=False)
            cosine_scores = cosine_similarity(jd_embedding, self.embeddings).flatten()
            self.df["SimilarityScore"] = cosine_scores
        else:
            self.df["SimilarityScore"] = np.random.uniform(0.3, 0.9, size=len(self.df))
            
        self.df["Skills_List"] = self.df["extracted_skills"].fillna(self.df.get("Skills", ""))
        self.df["SkillScore"] = self.df["Skills_List"].apply(lambda x: self.compute_skill_match(x, required_skills))
        
        self.df["ExperienceScore"] = self.df.get("experience_keyword_count", pd.Series(np.random.randint(0, 10, len(self.df)))) / 10.0
        self.df["ExperienceScore"] = self.df["ExperienceScore"].clip(0, 1)
        
        self.df["EducationScore"] = self.df.get("education_keyword_count", pd.Series(np.random.randint(0, 5, len(self.df)))) / 5.0
        self.df["EducationScore"] = self.df["EducationScore"].clip(0, 1)
        
        self.df["FinalScore"] = (
            self.weights["semantic_similarity"] * self.df["SimilarityScore"] +
            self.weights["skill_match"] * self.df["SkillScore"] +
            self.weights["experience_match"] * self.df["ExperienceScore"] +
            self.weights["education_match"] * self.df["EducationScore"]
        )
        
        self.df = self.df.sort_values(by="FinalScore", ascending=False)
        return self.df

    def visualize_ranking(self, top_n=10):
        logger.info(f"Visualizing top {top_n} candidates...")
        plot_df = self.df.head(top_n).copy()
        
        # Determine the best label for the y-axis
        y_col = "ResumeID"
        for col in ["Name", "Candidate Name", "CandidateName"]:
            if col in plot_df.columns:
                y_col = col
                break
                
        
        plt.figure(figsize=(10, 6))

        try:
            import seaborn as sns

            sns.barplot(
                data=plot_df,
                x="FinalScore",
                y=y_col,
                palette="viridis"
            )

        except Exception:
            plt.barh(plot_df[y_col], plot_df["FinalScore"])

        plt.tight_layout()
        
        
        
        
        plt.savefig(REPORTS_DIR / "candidate_ranking.png", dpi=300)
        plt.close()
