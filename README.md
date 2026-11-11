# Resume Classification and Job Recommendation System Using Machine Learning

An end-to-end Machine Learning web application designed for automated resume screening, domain category prediction, skill extraction, and intelligent job matching with skill gap analysis.

---

## 📌 Project Overview

Recruitment teams process hundreds of resumes for technical roles, making manual screening tedious and prone to bias. This application provides an automated pipeline that:
1. **Parses & Validates PDF Resumes** using PyMuPDF (`pymupdf`).
2. **Cleans Text & Extracts Skills** mapped against a predefined taxonomy across 10 technical domains.
3. **Classifies Professional Domains** using TF-IDF n-gram vectorization and Multi-class Logistic Regression with confidence scores.
4. **Recommends Top 5 Matching Jobs** using Cosine Similarity against an SQLite job listings database.
5. **Highlights Actionable Skill Gaps** by distinguishing matched skills from missing skills required for each job.
6. **Audits Scans Persistently** in SQLite database (`data/jobs.db`).

---

## 🏗️ System Architecture & Workflow

```
[ Candidate PDF Resume ]
          │
          ▼
 [ PyMuPDF Parser ] ──▶ Text Extraction & PDF Integrity Validation
          │
          ├──▶ [ Regex / Taxonomy ] ──▶ Skills & Contact Info
          │
          ▼
[ Text Preprocessing ] ──▶ URL/Email/Special Char Stripping & Token Normalization
          │
          ├──▶ [ TF-IDF Vectorizer ] ──▶ [ Logistic Regression ] ──▶ Predicted Category & Probabilities
          │
          ▼
[ SQLite Jobs Database ] ──▶ [ TF-IDF + Cosine Similarity ] ──▶ Top 5 Job Matches + Missing Skills
          │
          ▼
[ Streamlit Web Dashboard ] (Interactive Analytics, Charts & Real-time Job Exploration)
```

---

## 💻 Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| **Programming Language** | Python 3.12 | Core programming environment |
| **Document Processing** | PyMuPDF (`pymupdf`) | Fast, native PDF text and metadata extraction |
| **Text Preprocessing** | Custom Regex & Preprocessor | Text sanitization, contact info extraction |
| **Feature Extraction** | Scikit-Learn `TfidfVectorizer` | Sublinear term frequency-inverse document frequency (1,500 n-grams) |
| **Classification Model** | Scikit-Learn `LogisticRegression` | Multi-class probability classification across 8 job categories |
| **Recommendation Engine** | Scikit-Learn `cosine_similarity` | Vector space similarity matching with skill overlap blending |
| **Database** | SQLite 3 (`data/jobs.db`) | Relational storage for 36+ job postings and scan audit logs |
| **Interactive Dashboard** | Streamlit & Altair | Clean, modern user interface and probability visualizations |

---

## 📂 Project Directory Structure

```text
resume classification/
│
├── app.py                      # Main Streamlit dashboard application
├── train_model.py              # ML training, evaluation and artifact export script
├── resume_parser.py            # PDF document parser and text validator (PyMuPDF)
├── job_recommender.py          # TF-IDF + Cosine Similarity recommendation engine
├── test_pipeline.py            # Automated end-to-end integration test suite
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation and guide
│
├── modules/                    # Modular utilities and components
│   ├── __init__.py
│   ├── preprocessor.py         # Text sanitization, cleaning and stats calculation
│   ├── skills_dictionary.py    # Domain skills taxonomy and regex extraction
│   ├── classifier.py           # Logistic Regression model inference wrapper
│   ├── database.py             # SQLite database layer (jobs and scan history)
│   └── data_generator.py       # Benchmark dataset and sample PDF resume generator
│
├── data/                       # Datasets and database storage
│   ├── resumes_dataset.csv     # 120 labeled resumes across 8 categories
│   ├── jobs_dataset.csv        # 36 job listings with required skills
│   ├── jobs.db                 # SQLite database file
│   └── sample_resumes/         # Pre-generated PDF resumes for testing
│       ├── sample_data_scientist.pdf
│       ├── sample_fullstack_dev.pdf
│       ├── sample_devops_engineer.pdf
│       ├── sample_cybersecurity_analyst.pdf
│       ├── sample_mobile_developer.pdf
│       └── sample_qa_engineer.pdf
│
└── models/                     # Persisted trained model artifacts
    ├── resume_classifier.pkl   # Serialized Logistic Regression model
    ├── tfidf_vectorizer.pkl    # Serialized TF-IDF vectorizer
    └── model_metadata.json     # Training metrics, categories and evaluation report
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.12 (or 3.11/3.13)
- Windows PowerShell, macOS Terminal, or Linux Bash

### 2. Activate the Virtual Environment
```powershell
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# On Linux / macOS:
source venv/bin/activate
```

*(If creating a fresh virtual environment from scratch)*:
```powershell
python -m venv venv
.\venv\Scripts\pip install -r requirements.txt
```

### 3. Generate Datasets & SQLite Database
```powershell
.\venv\Scripts\python modules\data_generator.py
```
*Outputs: 36 jobs loaded into SQLite (`data/jobs.db`), 120 labeled resumes in `data/resumes_dataset.csv`, and 6 sample PDF resumes in `data/sample_resumes/`.*

### 4. Train the Machine Learning Model
```powershell
.\venv\Scripts\python train_model.py
```
*Outputs evaluation accuracy, detailed classification report, and saves artifacts into `models/`.*

### 5. Run the Automated Test Suite
```powershell
.\venv\Scripts\python test_pipeline.py
```
*Executes an end-to-end test across 6 diverse sample resumes, asserting valid PDF text extraction, correct category prediction, top 5 job recommendations with missing skills, and SQLite scan logging.*

### 6. Launch the Streamlit Web Application
```powershell
.\venv\Scripts\streamlit run app.py
```
Open your browser to: **`http://localhost:8501`**

---

## 🎯 Model & Evaluation Integrity

> **Academic and Industry Transparency Note**:  
> The training and test datasets included with this project are curated benchmark datasets created to represent 8 distinct technical specializations:
> 1. Data Science & Machine Learning
> 2. Full Stack Web Development
> 3. DevOps & Cloud Engineering
> 4. Cybersecurity & Network Engineering
> 5. Mobile Application Development
> 6. UI/UX Design & Frontend
> 7. Quality Assurance & Testing
> 8. Data Engineering & Big Data
> 
> The reported validation accuracy (~96.7%) demonstrates convergence and separability on this benchmark set. In real-world enterprise environments with noisy OCR scans, ambiguous role descriptions, and unstructured typography, accuracy will depend on OCR extraction fidelity and continuous model retraining on expansive industry corpora.

---

## 💡 Key Design Decisions & Interview Talking Points

1. **Why TF-IDF + Logistic Regression over Heavy Deep Learning?**
   - Resumes are structured documents with distinct technical vocabulary. TF-IDF with unigrams and bigrams captures domain-specific phrases (e.g., *"machine learning"*, *"ci/cd"*, *"rest api"*) with high interpretability and sub-millisecond inference times.
   - Logistic Regression provides calibrated, smooth probability distributions (`predict_proba`) across multiple classes, making it ideal for displaying confidence metrics.

2. **Why Cosine Similarity for Job Recommendation?**
   - Cosine similarity measures the angle between high-dimensional TF-IDF vectors regardless of document length. A short, concise resume can match well against a detailed job listing without being penalized for brevity.
   - We combine TF-IDF cosine similarity (70%) with direct skill overlap ratio (30%) to ensure both conceptual context and specific hard technical skills are rewarded.

3. **Missing Skill Gap Identification**:
   - The recommender dynamically computes:
     $$\text{Missing Skills} = \text{Set}(\text{Job Required Skills}) - \text{Set}(\text{Extracted Candidate Skills})$$
   - This provides actionable feedback for candidates to know exactly what skills to learn or highlight for a specific target role.

---

## 🧪 Testing Results

All tests have been verified with live execution on sample resumes:
- [x] PyMuPDF PDF Text Extraction: **Passed**
- [x] Contact & Skill Taxonomy Extraction: **Passed**
- [x] Multi-Class Classification & Confidence: **Passed**
- [x] Top 5 Job Recommendation & Missing Skills: **Passed**
- [x] SQLite Data Persistence & Scan Audit Logging: **Passed**
- [x] Streamlit Web Dashboard: **Passed (HTTP 200 OK)**

