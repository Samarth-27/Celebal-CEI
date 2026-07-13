# Project Progress — AI-Powered Intelligent Hiring Assistant

## Status Overview

| Module | Description | Status |
|---|---|---|
| **1. Data Engineering & Preprocessing** | Dataset EDA, feature engineering, and data cleaning | Complete |
| **2. Machine Learning Classification & Matching** | TF-IDF classification and Semantic Embeddings | Complete |
| **3. Resume Parsing & Dynamic Ranking** | Parsing real resumes and ranking them against JDs | Complete |
| **4. Generative AI & Conversational Mentorship** | Explainable AI (XAI), RAG pipeline, and AI Chatbot | Complete |
| **5. Full-Stack Application (Streamlit)** | Final UI for Candidate Portal and Recruiter Dashboard | Complete |

---

## Module 1 — Data Engineering & Preprocessing

**Completed:** 2026-07-11

**Key achievements:**
1. Loaded and processed the core `resumes_dataset.jsonl` (3,501 records). **Note: This is the single source of truth for the entire project. All subsequent datasets (`train.csv`, `test.csv`, `resumes_eda_pass.csv`) used to train and test the models are strictly derived from this file.**
2. Performed Exploratory Data Analysis (EDA) on text lengths, skill frequencies, and job categories.
3. Engineered structural numeric features (e.g., word count, sentence count, skill count).
4. Conducted Train-Test splitting to prepare data for Machine Learning models, preserving strict data isolation.

---

## Module 2 — Machine Learning Classification & Matching

**Completed:** 2026-07-11

**Key achievements:**
1. **Classification:** Trained Logistic Regression, Random Forest, SVM, and XGBoost models. XGBoost performed best with an Accuracy of 92.4% and Macro F1 of 0.9478.
2. Saved best classification models (`models/best_model.pkl`) to identify resume job categories.
3. **Semantic Matching:** Integrated `SentenceTransformers` (`all-MiniLM-L6-v2`) to capture deep context and meaning of resumes.
4. Embedded all candidate resumes and computed cosine similarities to evaluate semantic match against Job Descriptions.

---

## Module 3 — Resume Parsing & Dynamic Ranking

**Completed:** 2026-07-11

**Key achievements:**
1. **Parsing:** Built a robust document extraction engine utilizing `PyMuPDF` (`fitz`) and `pdfplumber` for PDFs, and `python-docx` for DOCX.
2. Implemented Named Entity Recognition (`spaCy`) for data extraction and Regex for contact details, effectively segmenting resumes into Skills, Experience, and Education.
3. **Ranking:** Created a weighted scoring model incorporating Skill Match (50%), Semantic Similarity (30%), Experience Match (10%), and Education Match (10%).
4. Dynamically ranked candidates against Job Descriptions to generate a top-candidates leaderboard.

---

## Module 4 — Generative AI & Conversational Mentorship

**Completed:** 2026-07-11

**Key achievements:**
1. **Explainable AI (XAI):** Developed an XAI module that unpacks the numerical ranking score by outlining matched vs. missing skills using set operations.
2. Used the Gemini API to automatically generate a concise, 3-step learning roadmap tailored to the exact skills the candidate is missing.
3. **RAG Pipeline:** Built a complete Retrieval-Augmented Generation pipeline using FAISS vector store and `RecursiveCharacterTextSplitter` on a local HR Knowledge Base.
4. **Conversational AI:** Integrated a Chatbot that grounds its advice on the RAG context, acting as a hiring mentor that can answer specific questions (e.g., "Why is my score only 75%?").

---

## Module 5 — Full-Stack Application (Streamlit)

**Completed:** 2026-07-11

**Key achievements:**
1. Built a modular, interactive UI integrating the output of all prior modules (Resume Parsing, Classification, FAISS semantic matching, XAI, and Conversational AI).
2. Segmented the UI into two separate portals: 
   - **Candidate Portal:** For resume upload, personalized feedback viewing, and chatting with the AI mentor.
   - **Recruiter Dashboard:** For batch processing Job Descriptions and viewing ranked candidates.
3. Resolved final environment dependencies (OpenBLAS/OMP thread management).

---

## Project Complete

All 5 core modules have been successfully implemented, streamlining the end-to-end "AI-Powered Intelligent Hiring Assistant" data science project pipeline!
