import os
import sys
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_resumes(json_path):
    """Load processed resumes from JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)


def load_job_description(jd_path):
    """Load a job description from a text file."""
    with open(jd_path, "r") as f:
        return f.read()


def compute_keyword_match(resume_skills, jd_text):
    """
    What % of skills in the resume appear in the JD text.
    Returns a score between 0 and 1.
    """
    if not resume_skills:
        return 0.0
    jd_lower = jd_text.lower()
    matched = [skill for skill in resume_skills if skill in jd_lower]
    return len(matched) / len(resume_skills)


def compute_experience_score(experience_years, required_years=2):
    """
    Score based on experience. Returns 1.0 if meets requirement,
    scales down if below, caps at 1.0 if above.
    """
    if required_years == 0:
        return 1.0
    return min(experience_years / required_years, 1.0)


def rank_resumes_tfidf(jd_text, resumes, top_n=10):
    """
    Rank resumes against a job description using TF-IDF cosine similarity
    combined with keyword match and experience score.
    Returns top_n ranked candidates.
    """
    # Build corpus: JD is first, then all resume texts
    corpus = [jd_text] + [r["raw_text"] for r in resumes]

    # Fit TF-IDF on full corpus
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2)  # unigrams + bigrams
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # JD vector is index 0, resumes start at index 1
    jd_vector = tfidf_matrix[0]
    resume_vectors = tfidf_matrix[1:]

    # Cosine similarity between JD and each resume
    similarities = cosine_similarity(jd_vector, resume_vectors)[0]

    results = []
    for i, resume in enumerate(resumes):
        tfidf_score = float(similarities[i])
        keyword_score = compute_keyword_match(resume["skills"], jd_text)
        experience_score = compute_experience_score(resume["experience_years"])
        completeness_score = resume.get("section_completeness", 0.5)

        # Composite score: weighted combination
        composite = (
            0.45 * tfidf_score +
            0.25 * keyword_score +
            0.15 * experience_score +
            0.15 * completeness_score
        )

        results.append({
            "rank": 0,  # filled below
            "filename": resume["filename"],
            "candidate_name": resume.get("candidate_name", ""),
            "category": resume.get("category", "unknown"),
            "skills": resume["skills"],
            "experience_years": resume["experience_years"],
            "education": resume["education"],
            "tfidf_score": round(tfidf_score, 4),
            "keyword_score": round(keyword_score, 4),
            "experience_score": round(experience_score, 4),
            "completeness_score": round(completeness_score, 4),
            "composite_score": round(composite, 4),
        })

    # Sort by composite score descending
    results = sorted(results, key=lambda x: x["composite_score"], reverse=True)

    # Assign ranks
    for i, r in enumerate(results):
        r["rank"] = i + 1

    return results[:top_n]


def print_results(results, jd_name=""):
    """Pretty print ranking results."""
    print(f"\n{'='*60}")
    print(f"Top {len(results)} candidates for: {jd_name}")
    print(f"{'='*60}")
    for r in results:
        # Handle both tfidf and semantic score keys
        similarity_score = r.get("tfidf_score") or r.get("semantic_score") or 0.0
        print(f"\nRank #{r['rank']} | Score: {r['composite_score']}")
        print(f"  File     : {r['filename']}")
        print(f"  Category : {r['category']}")
        print(f"  Skills   : {', '.join(r['skills'][:5])}{'...' if len(r['skills']) > 5 else ''}")
        print(f"  Exp Years: {r['experience_years']}")
        print(f"  Similarity: {similarity_score} | Keyword: {r['keyword_score']} | Exp: {r['experience_score']}")

def rank_resumes_semantic(jd_text, resumes, top_n=10, model_name="all-MiniLM-L6-v2"):
    """
    Rank resumes using sentence-transformer semantic embeddings.
    Much better at understanding meaning beyond keyword overlap.
    """
    from sentence_transformers import SentenceTransformer

    print("Loading sentence transformer model...")
    model = SentenceTransformer(model_name)

    # Encode JD and all resumes
    # Truncate resume text to 512 words to avoid memory issues
    jd_embedding = model.encode([jd_text])
    resume_texts = [" ".join(r["raw_text"].split()[:512]) for r in resumes]

    print(f"Encoding {len(resume_texts)} resumes...")
    resume_embeddings = model.encode(resume_texts, batch_size=32, show_progress_bar=True)

    # Cosine similarity
    similarities = cosine_similarity(jd_embedding, resume_embeddings)[0]

    results = []
    for i, resume in enumerate(resumes):
        semantic_score = float(similarities[i])
        keyword_score = compute_keyword_match(resume["skills"], jd_text)
        experience_score = compute_experience_score(resume["experience_years"])
        completeness_score = resume.get("section_completeness", 0.5)

        composite = (
            0.45 * semantic_score +
            0.25 * keyword_score +
            0.15 * experience_score +
            0.15 * completeness_score
        )

        results.append({
            "rank": 0,
            "filename": resume["filename"],
            "candidate_name": resume.get("candidate_name", ""),
            "category": resume.get("category", "unknown"),
            "skills": resume["skills"],
            "experience_years": resume["experience_years"],
            "education": resume["education"],
            "semantic_score": round(semantic_score, 4),
            "keyword_score": round(keyword_score, 4),
            "experience_score": round(experience_score, 4),
            "completeness_score": round(completeness_score, 4),
            "composite_score": round(composite, 4),
        })

    results = sorted(results, key=lambda x: x["composite_score"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    return results[:top_n]