<div align="center">

# 🚀 Celebal-CEI

### Data Science Internship — Celebal Excellence Internship Program 2026

**Samarth Jain · JECRC University, Jaipur**

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)

</div>

---

<div align="center">

## 🏆 MAJOR PROJECT
### AI-Powered Intelligent Hiring Assistant
**The final capstone project of this internship**

[![Live Demo](https://img.shields.io/badge/🌐_LIVE_DEMO-ai--hiring--assistant01.streamlit.app-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://ai-hiring-assistant01.streamlit.app/)

### 👉 [**Click here to view the Major Project**](AI-Hiring-Assistant/README.md)

</div>

An end-to-end AI system that evaluates candidate resumes against job requirements using ML and deep learning, then generates personalized, explainable feedback through a retrieval-based generation pipeline. Built across 5 modules — data engineering, an XGBoost classifier for 40+ job categories, semantic resume ranking, a Groq + FAISS-backed RAG mentor chatbot, and a full Streamlit app.

| Candidate Portal | Recruiter Dashboard |
|:---:|:---:|
| ![Candidate Portal](AI-Hiring-Assistant/assets/candidate_portal_1.png) | ![Recruiter Dashboard](AI-Hiring-Assistant/assets/recruiter_dashboard.png) |

---

## 📓 Weekly Assignments

The 8 notebooks below are the week-by-week practice work completed during the internship, building up from ML fundamentals to the skills used in the capstone above. Each has an **"Open in Colab"** badge — click to run live, no setup needed.

| # | Project | Key Concepts |
|---|---------|---------------|
| [Week 1](week1_SamarthJain_Jecrc.ipynb) | ML Foundations | Python · NumPy · Pandas · Linear Algebra · Statistics · Probability |
| [Week 2](week2_SamarthJain_Jecrc.ipynb) | Tesla EV Production Forecasting | Feature Engineering · sklearn Pipelines · GridSearchCV · Gradient Boosting |
| [Week 3](week3_SamarthJain_Jecrc.ipynb) | Unsupervised Learning on Country Data | K-Means · DBSCAN · PCA · Cluster Profiling |
| [Week 4](week4_SamarthJain_Jecrc.ipynb) | CIFAR-10: ANN vs CNN | CNNs · Data Augmentation · EarlyStopping · Confusion Matrix |
| [Week 5](week5_SamarthJain_Jecrc.ipynb) | Text Generation | Vanilla RNN · LSTM · GRU · Sequence Modeling |
| [Week 6](week6_SamarthJain_Jecrc.ipynb) | Denoising Autoencoder | PyTorch · Convolutional Autoencoders · Skip Connections |
| [Week 7](week7_SamarthJain_Jecrc.ipynb) | RAG Document Q&A | Embeddings · Vector DB · Retrieval · Mistral-7B |
| [Week 8](week8_SamarthJain_Jecrc.ipynb) | Keyword & Calculator Agent | Rule-based Routing · Tool-use Pattern · Error Handling |

<details>
<summary><b>📘 Week 1 — ML Foundations</b></summary>
<br>

Core building blocks for everything that follows:
- **Python:** control flow, data structures, exceptions, lambdas
- **NumPy:** array creation, indexing/slicing, matrix operations, dot products
- **Pandas:** DataFrames vs Series, `loc`/`iloc`, filtering, group-by, missing data
- **Linear Algebra:** vectors, eigenvalues/eigenvectors, SVD & its link to PCA
- **Statistics:** descriptive vs inferential stats, hypothesis testing, error metrics (MAE/MSE/RMSE/R²), stationarity testing, model monitoring (PSI, concept vs covariate drift)
- **Probability:** joint/conditional probability, common distributions, Bayes' theorem, Central Limit Theorem

</details>

<details>
<summary><b>🚗 Week 2 — Tesla EV Production Forecasting</b></summary>
<br>

End-to-end regression pipeline on Tesla EV delivery/production data (2015–2025).

- EDA on yearly deliveries, model-wise share, and regional pricing
- Feature engineering: price/km, delivery rate, rolling averages, lag features
- Leakage fix (dropped a near-duplicate target column) + one-hot encoding
- Chronological (time-aware) train/test split
- `sklearn` Pipeline: `StandardScaler` + `GradientBoostingRegressor`
- Hyperparameter tuning via `GridSearchCV` (5-fold CV)
- Evaluation: R², RMSE, cross-validation consistency, feature importance

**Result: ~99.65% R² on held-out data**

</details>

<details>
<summary><b>🌍 Week 3 — Unsupervised Learning on Country Data</b></summary>
<br>

Clustering countries by socio-economic indicators (child mortality, income, GDP, health/education spend, life expectancy).

- Cleaning + scaling of 9 numeric indicators
- Elbow method → optimal k = 3 for K-Means
- DBSCAN as a density-based alternative (flags outlier nations)
- PCA for 2D cluster visualization
- Cluster profiling → underdeveloped / developing / developed country groups

</details>

<details>
<summary><b>🖼️ Week 4 — CIFAR-10: ANN vs CNN</b></summary>
<br>

Head-to-head comparison of a plain ANN and a CNN on CIFAR-10 (60,000 32×32 color images, 10 classes).

- Baseline ANN on flattened pixel vectors
- CNN with 32→64→128 filter progression
- EarlyStopping (patience=5) to prevent overfitting
- Data augmentation (flip/rotate/zoom) to close the train/val gap
- Learning curve comparison + confusion matrix on the best model
- Write-up on *why* CNNs beat ANNs on image data (spatial feature learning)

</details>

<details>
<summary><b>✍️ Week 5 — Text Generation: RNN vs LSTM vs GRU</b></summary>
<br>

Sequence modeling for next-word prediction and text generation on a shared corpus.

- Tokenization + n-gram sequence creation
- Three architectures trained side by side: **Vanilla RNN**, **LSTM**, **GRU**
- Training loss comparison across all three
- Discussion of vanishing gradients (RNN) vs gated memory (LSTM/GRU)
- Sample text generation from each model

</details>

<details>
<summary><b>🧹 Week 6 — Denoising Autoencoder (MNIST)</b></summary>
<br>

Convolutional autoencoder that learns to remove Gaussian noise from MNIST digits — built with PyTorch.

- Encoder-decoder using conv + maxpool / upsampling (avoids transposed-conv checkerboarding)
- **Debugging story:** hit dying ReLU at `lr=1e-3` → fixed with LeakyReLU + `lr=1e-4`
- Evaluated across unseen noise levels using MSE/PSNR
- **v2 improvements:** skip connection, BatchNorm, randomized noise schedule (0.1–0.6)
- v2 outperformed the baseline at every tested noise level

</details>

<details>
<summary><b>📄 Week 7 — Retrieval-Augmented Generation (RAG) Document Q&A</b></summary>
<br>

Full end-to-end RAG pipeline for answering questions over a document (a resume):

1. Document ingestion
2. Text chunking
3. Embedding creation
4. Vector database storage
5. Query processing
6. Context retrieval
7. Answer generation via **Mistral-7B**

Demonstrated hallucination-free, grounded answers compared to a plain LLM.

</details>

<details>
<summary><b>🤖 Week 8 — Keyword & Calculator Agent Pipeline</b></summary>
<br>

A small single-agent system that routes a query to the right tool based on keyword matching:

- `"calculate"` → safe calculator (`eval`-based, with error handling)
- `"keyword"` → keyword extractor (word-length heuristic)
- anything else → general fallback response

Handles malformed input gracefully and includes an interactive manual-testing loop. A simple, clear illustration of the tool-routing pattern behind most agent frameworks.

</details>

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Languages** | Python |
| **Data Handling** | NumPy, Pandas |
| **Visualization** | Matplotlib, Seaborn |
| **Classical ML** | Scikit-learn |
| **Deep Learning** | TensorFlow/Keras, PyTorch |
| **Stats** | Statsmodels, SciPy |
| **NLP / GenAI** | Mistral-7B, Groq API, FAISS, SentenceTransformers, spaCy |
| **Web App** | Streamlit |
| **Environment** | Google Colab |

## 📂 Repository Structure

```
Celebal-CEI/
├── AI-Hiring-Assistant/            # 🏆 Major Project — AI-Powered Intelligent Hiring Assistant
├── week1_SamarthJain_Jecrc.ipynb   # ML Foundations
├── week2_SamarthJain_Jecrc.ipynb   # Regression Pipeline (Tesla EV)
├── week3_SamarthJain_Jecrc.ipynb   # Unsupervised Learning (Clustering)
├── week4_SamarthJain_Jecrc.ipynb   # CNN vs ANN (CIFAR-10)
├── week5_SamarthJain_Jecrc.ipynb   # RNN / LSTM / GRU Text Generation
├── week6_SamarthJain_Jecrc.ipynb   # Denoising Autoencoder
├── week7_SamarthJain_Jecrc.ipynb   # RAG Document QA
├── week8_SamarthJain_Jecrc.ipynb   # Keyword & Calculator Agent
└── README.md
```

---

<div align="center">

**Samarth Jain** · JECRC University · Celebal Excellence Internship 2026

</div>
