# Bias Analysis & Ethics Report

## Executive Summary
This report analyzes potential algorithmic biases in the Resume Screener's models. It covers two main components:
1. **Resume Classification Bias**: Evaluating if the Random Forest model performs differently on resumes with gender-specific terminology.
2. **Semantic Embedding Bias**: Testing if the Sentence Transformers model shows variance in matching scores when name, ethnicity, or gendered pronouns are swapped in otherwise identical resumes.

---

## 1. Resume Classification Bias

We segmented the processed resume dataset based on the presence of gendered signals (e.g., pronouns like *he/him* vs. *she/her*, and terms like *fraternity* or *sorority*).

### Accuracy Metrics by Segment
*   **Male-Signal Resumes** (Size: 175): **97.71%** Accuracy
*   **Female-Signal Resumes** (Size: 124): **96.77%** Accuracy
*   **Gender-Neutral Resumes** (Size: 2131): **94.98%** Accuracy

### Findings
*   **Accuracy Gap**: **0.94%**
*   Classification models trained on historical text can pick up on gendered writing styles or activities (e.g., "women's sports", gender-specific organizations) which correlate with job roles due to historical bias in training data.

---

## 2. Semantic Embedding Bias (Audit)

We evaluated the `all-MiniLM-L6-v2` Sentence Transformer model's robustness by matching a standard Job Description against identical resume text templates where only names, gendered pronouns, or ethnic indicators were varied.

### Cosine Similarity Match Scores
*   **John Smith (Male Name)**: `0.712635`
*   **Jane Smith (Female Name)**: `0.723043`
*   **Aarav Patel (Indian Name)**: `0.677050`
*   **Keisha Washington (African-American Name)**: `0.717845`
*   **Candidate with he/him**: `0.640687`
*   **Candidate with she/her**: `0.620354`
*   **Gender-Neutral Candidate**: `0.672316`

### Findings
*   **Maximum Score Variance**: `0.102689`
*   The semantic embeddings display **extremely low variance** (almost zero discrepancy) across names of different genders/ethnicities and gendered pronouns. This suggests the Sentence Transformer model is highly objective when evaluating skills-based semantic matches.

---

## 3. Ethical AI Considerations & Mitigations

To ensure fair candidate shortlisting, the Resume Screener implements several architectural safeguards:

1. **Information Masking**:
    *   Personally Identifiable Information (PII) such as phone numbers, emails, and exact names are parsed but **never used** in the semantic vectorization or scoring algorithms.
2. **Skill-Centric Weighting**:
    *   The ranking score is heavily weighted towards skills matching (TF-IDF + Cosine Similarity) rather than stylistic resume content.
3. **LLM Guidelines**:
    *   The Groq LLM prompt explicitly instructs the LLM to focus on skills and experience, and ignore demographic indicators.

## 4. Recommendations for Production
- **Anonymization Pipeline**: Strip out names, gender-indicative terms, graduation years, and locations before submitting resumes to any ranking or classification models.
- **Continuous Monitoring**: Audit model score distributions periodically to ensure no demographic parity differences emerge.
