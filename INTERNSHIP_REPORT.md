# Internship Project Report & Viva Defense Guide

**Project Title:** Resume Classification and Job Recommendation System Using Machine Learning  
**Domain:** Applied Natural Language Processing & Information Retrieval  
**Stack:** Python 3.12, Streamlit, Scikit-Learn, PyMuPDF, Pandas, SQLite 3  

---

## 1. Executive Summary & Problem Statement

Modern talent acquisition pipelines face severe bottlenecks when screening hundreds of unstructured candidate resumes. Manual parsing is slow, inconsistent, and often subjective. 

This project delivers an automated, beginner-friendly end-to-end Machine Learning solution that:
- Ingests raw candidate resumes in PDF format.
- Validates file integrity and extracts raw text using PyMuPDF (`pymupdf`).
- Normalizes and sanitizes text for NLP pipelines.
- Identifies technical competencies mapped to a 10-domain skill taxonomy.
- Classifies professional profiles across 8 specialized job domains using TF-IDF n-grams and Logistic Regression.
- Recommends the top 5 matching career opportunities from an SQLite database using Cosine Similarity and provides actionable **Missing Skill Gap Analysis**.
- Persists all scan activity in SQLite for auditability.

---

## 2. Technical Architecture & Mathematical Formulations

### 2.1 PyMuPDF Text Extraction & Document Validation
Text extraction utilizes PyMuPDF's C-accelerated MuPDF engine:
- Verifies non-empty byte streams (> 0 bytes).
- Ensures valid PDF file headers and page counts.
- Flags low-character count documents (< 50 characters) to identify scanned, image-only, or password-protected files.

### 2.2 Text Preprocessing
Raw resume strings are sanitized through sequential regex filters:
- Strips URLs: `https?://\S+|www\.\S+`
- Strips Email addresses: `[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+`
- Strips International & Domestic Phone patterns.
- Preserves technical language identifiers (e.g., `C++` $\rightarrow$ `cplusplus`, `C#` $\rightarrow$ `csharp`, `.NET` $\rightarrow$ `dotnet`).
- Case folding to lowercase and whitespace collapse.

### 2.3 TF-IDF Vectorization
Given a document $d$ and term $t$ in corpus $D$:
$$\text{TF}(t, d) = 1 + \ln(f_{t, d}) \quad \text{for } f_{t, d} > 0$$
$$\text{IDF}(t, D) = \ln\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
Features are restricted to 1,500 unigram and bigram tokens with sublinear scaling to prevent frequent terms from dominating the sparse feature space.

### 2.4 Multi-Class Logistic Regression Classification
For $K = 8$ classes, the model estimates class probabilities using the Softmax formulation:
$$P(Y = k \mid \mathbf{x}) = \frac{e^{\mathbf{w}_k^T \mathbf{x} + b_k}}{\sum_{j=1}^K e^{\mathbf{w}_j^T \mathbf{x} + b_j}}$$
The class with the highest probability is assigned as the primary predicted category, and the probability score represents model confidence.

### 2.5 Job Matching & Skill Gap Blending
Job recommendations are calculated by vectorizing each job description and computing Cosine Similarity:
$$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
The overall candidate match score balances semantic similarity with hard skill requirements:
$$\text{Blended Match Score} = \left(0.70 \times \text{Cosine Similarity}\right) + \left(0.30 \times \frac{|\text{Candidate Skills} \cap \text{Job Skills}|}{|\text{Job Skills}|}\right)$$
Actionable missing skills are calculated as:
$$\text{Missing Skills} = \text{Job Required Skills} \setminus \text{Candidate Skills}$$

---

## 3. Benchmark Dataset & Evaluation Integrity

### 3.1 Dataset Breakdown
- **Training Resumes**: 120 curated profiles (15 per category) across 8 domains:
  1. Data Science & Machine Learning
  2. Full Stack Web Development
  3. DevOps & Cloud Engineering
  4. Cybersecurity & Network Engineering
  5. Mobile Application Development
  6. UI/UX Design & Frontend
  7. Quality Assurance & Testing
  8. Data Engineering & Big Data
- **Job Listings**: 36 listings stored in SQLite (`data/jobs.db`).
- **Validation Split**: 25% stratified test split (30 resumes).

### 3.2 Evaluation Results (Test Split)
- **Overall Accuracy**: **96.7%**
- **Macro Average F1-Score**: **0.96**
- **Weighted Average F1-Score**: **0.97**

> **Note on Prototype Limitations**:
> High accuracy is achieved due to clean vocabulary separation in the curated benchmark corpus. Real-world resumes feature complex graphical layouts, multi-column designs, tables, and varied wording. In production, this system should be supplemented with layout-aware OCR (e.g., Tesseract/LayoutLM) and continuous active learning loops.

---

## 4. Key Viva / Interview Q&A

### Q1: Why use TF-IDF and Logistic Regression instead of deep transformers (like BERT)?
**Answer:**
1. **Inference Latency**: TF-IDF + Logistic Regression computes predictions in under 10 milliseconds without requiring GPU acceleration.
2. **Interpretability**: Logistic Regression coefficients directly reflect which vocabulary terms contributed to a domain classification.
3. **Calibrated Confidence**: Multi-class Logistic Regression yields smooth probability distributions suitable for confidence progress bars.
4. **Beginner-Friendly Deployment**: Minimal memory footprint (~50 MB RAM) allows the application to run smoothly on any laptop or small cloud instance.

### Q2: Why is Cosine Similarity preferred over Euclidean Distance for document matching?
**Answer:**
Euclidean distance is sensitive to document length (a 3-page resume would appear far from a short job description simply due to higher term frequencies). Cosine similarity measures the angle between vectors rather than magnitude, normalizing for document length and focusing purely on contextual direction.

### Q3: How does the system handle "Missing Skills"?
**Answer:**
The system normalizes candidate skills and job requirements into lowercase stems (handling plurals such as `API`/`APIs` and aliases like `React`/`React.js`). It performs set difference to explicitly highlight the skills the candidate lacks for a given role, turning the tool into a career development guide rather than just a filter.

---

## 5. Verification Checklist

- [x] Python 3.12 virtual environment configured
- [x] PyMuPDF PDF parser with document validation
- [x] Text cleaning, contact regex, and 100+ skill taxonomy matching
- [x] 120-sample benchmark dataset across 8 domains
- [x] TF-IDF + Logistic Regression trained and serialized to `models/`
- [x] SQLite database with 36 jobs and scan audit log table
- [x] Blended Cosine Similarity recommender with missing skills
- [x] Modern Streamlit dashboard with probability charts and report export
- [x] Complete end-to-end test suite (`test_pipeline.py`) passing 100%

