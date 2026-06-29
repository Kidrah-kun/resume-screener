# 📄 Project 07: AI-Powered Resume Screening System

**Domain:** Natural Language Processing / HR Technology

An end-to-end resume screening system that automates candidate shortlisting by parsing resumes, extracting key skills and experience, and ranking applicants against a job description using NLP similarity techniques.

Built with a **Python/Flask** backend, a **Streamlit** dashboard for HR personnel, and a **React/TypeScript** frontend.

---

## ✨ Features

- **Resume Parsing** — Extracts raw text from PDF and DOCX files using PyMuPDF and python-docx
- **Information Extraction & NER** — Identifies skills, experience years, education level, email, phone, and named entities (persons, organizations, locations, dates) using spaCy NER, NLTK preprocessing, and regex patterns
- **Semantic Ranking** — Ranks resumes against a job description using Sentence Transformer embeddings (`all-MiniLM-L6-v2`) with cosine similarity
- **TF-IDF Ranking** — Alternative ranking method using TF-IDF vectorization with bigrams
- **Composite Scoring** — Weighted combination of similarity (45%), keyword match (25%), experience (15%), and section completeness (15%)
- **LLM Analysis** — Generates structured candidate summaries (strengths, gaps, recommendation) via Groq's LLaMA 3.3 70B model
- **Resume Classification** — Predicts resume category (e.g., Engineering, IT, Finance) using a trained Random Forest classifier (76.5% accuracy)
- **Bias Analysis** — Comprehensive bias and ethics report evaluating model fairness across gender and ethnicity
- **Dual Frontend** — React UI for production use, Streamlit dashboard for HR quick prototyping

---

## 📊 Datasets

| Dataset | Description |
|---------|-------------|
| **Kaggle Resume Dataset** | 2,400+ labelled resumes across 25 job categories ([source](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset)) |
| **Custom Job Descriptions** | 5 self-created JDs: Software Engineer, Data Analyst, Finance Analyst, HR Manager, Marketing Executive |
| **O\*NET Skills Database** | Skill taxonomy sourced from [O\*NET OnLine](https://www.onetonline.org/) used to build the skills extraction vocabulary |

---

## 🤖 ML Algorithms

| Algorithm | Purpose | Implementation |
|-----------|---------|----------------|
| **Cosine Similarity** (Sentence Embeddings) | Semantic matching between JD and resumes | `matcher.py` — `rank_resumes_semantic()` |
| **TF-IDF + Cosine Similarity** | Keyword-based matching with bigrams | `matcher.py` — `rank_resumes_tfidf()` |
| **Logistic Regression** | Resume category classification (baseline) | `classifier.py` — 66.6% accuracy |
| **Random Forest** | Resume category classification (best model) | `classifier.py` — 76.5% accuracy |
| **Named Entity Recognition (NER)** with spaCy | Extract persons, orgs, locations, dates | `extractor.py` — `extract_entities_ner()` |

---

## 📋 Features / Inputs

The system extracts and computes the following features from each resume:

| Feature | Source |
|---------|--------|
| Extracted skills | Keyword matching against O\*NET-enhanced taxonomy |
| Education level | Regex + keyword matching |
| Experience entities (NER) | spaCy Named Entity Recognition |
| TF-IDF / Sentence embedding similarity to JD | Scikit-learn TF-IDF / Sentence Transformers |
| Years of experience | Multi-pattern regex extraction |
| Keyword match ratio | Skills overlap with job description |
| Section completeness score | Presence of expected resume sections |

---

## 🏗️ Architecture

```
resume-screener/
├── resume-screener-backend/      # Python backend
│   ├── app/
│   │   ├── flask_api.py          # REST API (Flask + CORS)
│   │   └── streamlit_app.py      # Streamlit dashboard for HR personnel
│   ├── src/
│   │   ├── parser.py             # PDF/DOCX text extraction (PyMuPDF)
│   │   ├── extractor.py          # Skills, experience, education, NER extraction
│   │   │                         #   (spaCy + NLTK + O*NET taxonomy)
│   │   ├── matcher.py            # Semantic + TF-IDF ranking with cosine similarity
│   │   ├── classifier.py         # ML model training (LR + Random Forest)
│   │   ├── bias_analyzer.py      # Bias & ethics analysis on model outputs
│   │   ├── llm_summarizer.py     # Groq LLM candidate analysis
│   │   └── process_dataset.py    # Kaggle dataset preprocessing pipeline
│   ├── models/                   # Trained classifier + vectorizer (.pkl)
│   ├── data/
│   │   ├── raw/                  # Resume.csv, sample PDFs, job descriptions
│   │   └── processed/            # Processed JSON, LLM cache
│   ├── notebooks/
│   │   └── 01_eda.ipynb          # Exploratory Data Analysis notebook
│   ├── reports/                  # Visualizations + bias analysis report
│   └── requirements.txt
│
└── resume-screener-ui/           # React frontend
    ├── src/
    │   ├── App.tsx               # Main application component
    │   ├── App.css               # Styling
    │   └── index.tsx             # Entry point
    ├── package.json
    └── tsconfig.json
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/screen` | Upload resumes + JD → ranked candidates with scores and LLM analysis |
| `POST` | `/api/classify` | Upload a single resume → predicted job category with confidence |

### `POST /api/screen`

**Form Data:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `jd_text` | string | ✅ | — | Job description text |
| `resumes` | file[] | ✅ | — | PDF/DOCX resume files |
| `top_n` | int | ❌ | 5 | Number of top candidates to return |
| `use_llm` | bool | ❌ | true | Generate AI summaries via Groq |
| `method` | string | ❌ | semantic | Ranking method: `semantic` or `tfidf` |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- A [Groq API key](https://console.groq.com/) (free tier available)

### Backend Setup

```bash
cd resume-screener/resume-screener-backend

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Configure environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# Start the API server
python app/flask_api.py
# → Running on http://localhost:8000
```

### Frontend Setup

```bash
cd resume-screener/resume-screener-ui

# Install dependencies
npm install

# Start the dev server
npm start
# → Running on http://localhost:3000
```

### Streamlit Dashboard (Alternative UI)

```bash
cd resume-screener/resume-screener-backend
source venv/bin/activate
streamlit run app/streamlit_app.py
```

---

## 🧠 ML Model Performance

Trained on the [Kaggle Resume Dataset](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset) with 2,400+ resumes across 25 categories.

| Model | Test Accuracy | CV F1 Score (5-fold) |
|-------|:------------:|:--------------------:|
| Logistic Regression | 66.6% | 0.640 ± 0.042 |
| **Random Forest** ✅ | **76.5%** | **0.720 ± 0.058** |

---

## ⚖️ Bias Analysis & Ethical AI

A comprehensive bias analysis report is generated by `src/bias_analyzer.py` and saved to `reports/bias_analysis_report.md`. The analysis covers:

1. **Classification Bias** — Evaluates if the Random Forest classifier performs differently on resumes with male vs. female gendered terminology (accuracy gap: **0.94%**)
2. **Embedding Bias** — Tests Sentence Transformer model for name/gender/ethnicity bias using identical resumes with swapped demographic variables
3. **Ethical Safeguards** — PII is never used in scoring; ranking is skill-centric; LLM prompts focus on qualifications only
4. **Production Recommendations** — Anonymization pipeline, continuous monitoring

---

## 🛠️ Tech Stack

| Layer | Technologies |
|-------|-------------|
| **NLP & ML** | Sentence Transformers, scikit-learn, spaCy, NLTK, TF-IDF |
| **LLM** | Groq API (LLaMA 3.3 70B) |
| **Backend** | Flask, Python |
| **Frontend** | React, TypeScript, Axios, Lucide Icons |
| **Data** | PyMuPDF, python-docx, pandas |
| **Visualization** | Matplotlib, Seaborn |
| **Skill Taxonomy** | O\*NET Skills Database |

---

## 📚 Key Learning Outcomes

- **Information Extraction and NER** — Using spaCy's Named Entity Recognition to extract structured data from unstructured resume text
- **Semantic Similarity with Sentence Transformers** — Encoding text into dense vectors and computing cosine similarity for meaning-aware matching
- **PDF/DOCX Parsing in Python** — Handling multiple document formats using PyMuPDF and python-docx libraries
- **Ethical AI Considerations in HR** — Analyzing and mitigating algorithmic bias in automated hiring systems

---

## 📦 Expected Deliverables

| Deliverable | Status | Location |
|-------------|--------|----------|
| Resume parser supporting PDF and DOCX inputs | ✅ | `src/parser.py` |
| Job-description matching engine with ranked candidate list | ✅ | `src/matcher.py` |
| Streamlit dashboard for HR personnel | ✅ | `app/streamlit_app.py` |
| Bias analysis report on model outputs | ✅ | `reports/bias_analysis_report.md` |

---

## 📄 License

This project is for educational and learning purposes.
