import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.matcher import (load_resumes, load_job_description,
                          rank_resumes_tfidf, rank_resumes_semantic, print_results)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

resumes = load_resumes(os.path.join(BASE_DIR, "data", "processed", "processed_resumes.json"))
jd_text = load_job_description(os.path.join(BASE_DIR, "data", "raw", "job_descriptions", "software_engineer.txt"))

print("\n--- TF-IDF Ranking ---")
tfidf_results = rank_resumes_tfidf(jd_text, resumes, top_n=5)
print_results(tfidf_results, jd_name="Software Engineer (TF-IDF)")

print("\n--- Semantic Ranking ---")
semantic_results = rank_resumes_semantic(jd_text, resumes, top_n=5)
print_results(semantic_results, jd_name="Software Engineer (Semantic)")