import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.matcher import (load_resumes, load_job_description,
                          rank_resumes_tfidf, rank_resumes_semantic, print_results)

from src.llm_summarizer import summarize_top_candidates
import json


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

resumes = load_resumes(os.path.join(BASE_DIR, "data", "processed", "processed_resumes.json"))
jd_text = load_job_description(os.path.join(BASE_DIR, "data", "raw", "job_descriptions", "software_engineer.txt"))

print("\n--- TF-IDF Ranking ---")
tfidf_results = rank_resumes_tfidf(jd_text, resumes, top_n=5)
print_results(tfidf_results, jd_name="Software Engineer (TF-IDF)")

print("\n--- Semantic Ranking ---")
semantic_results = rank_resumes_semantic(jd_text, resumes, top_n=5)
print_results(semantic_results, jd_name="Software Engineer (Semantic)")


# Load raw texts for top results
with open(os.path.join(BASE_DIR, "data", "processed", "processed_resumes.json")) as f:
    all_resumes = json.load(f)

# Add raw_text to semantic results
resume_map = {r["filename"]: r["raw_text"] for r in all_resumes}
for r in semantic_results:
    r["raw_text"] = resume_map.get(r["filename"], "")

# Generate summaries
results_with_summaries = summarize_top_candidates(semantic_results, jd_text, top_n=3)

print("\n--- LLM Summaries ---")
for r in results_with_summaries[:3]:
    print(f"\nRank #{r['rank']} — {r['filename']}")
    summary = r.get("llm_summary", {})
    print(f"  One liner : {summary.get('one_liner')}")
    print(f"  Strengths : {summary.get('strengths')}")
    print(f"  Gaps      : {summary.get('gaps')}")
    print(f"  Verdict   : {summary.get('recommendation')}")