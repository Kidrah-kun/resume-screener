import os
import json
import hashlib
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Simple file-based cache so we don't re-query for same resume+JD pair
CACHE_FILE = "data/processed/llm_cache.json"


def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def make_cache_key(resume_text, jd_text):
    """Create a unique key for this resume+JD combination."""
    combined = resume_text[:500] + jd_text[:200]
    return hashlib.md5(combined.encode()).hexdigest()


def generate_candidate_summary(resume_text, jd_text, candidate_score):
    """
    Use Groq LLM to generate a structured summary of how well
    a candidate fits a job description.
    Returns a dict with strengths, gaps, fit_score, one_liner.
    """
    cache = load_cache()
    cache_key = make_cache_key(resume_text, jd_text)

    # Return cached result if available
    if cache_key in cache:
        return cache[cache_key]

    # Truncate to avoid token limits
    resume_snippet = resume_text[:1500]
    jd_snippet = jd_text[:500]

    prompt = f"""You are an expert HR assistant. Analyze this candidate's resume against the job description and return ONLY a JSON object with no extra text.

JOB DESCRIPTION:
{jd_snippet}

CANDIDATE RESUME:
{resume_snippet}

Return this exact JSON structure:
{{
  "one_liner": "One sentence summary of this candidate (max 20 words)",
  "strengths": ["strength 1", "strength 2", "strength 3"],
  "gaps": ["gap 1", "gap 2"],
  "recommendation": "Strong Fit" or "Good Fit" or "Partial Fit" or "Weak Fit",
  "reasoning": "2 sentences explaining the recommendation"
}}"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=400,
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown code blocks if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        result = json.loads(raw)

        # Cache and return
        cache[cache_key] = result
        save_cache(cache)
        return result

    except Exception as e:
        print(f"LLM error: {e}")
        # Fallback if LLM fails
        return {
            "one_liner": "Candidate profile could not be summarized.",
            "strengths": ["Profile available for manual review"],
            "gaps": ["Unable to analyze automatically"],
            "recommendation": "Manual Review",
            "reasoning": "Automated analysis unavailable. Please review manually."
        }


def summarize_top_candidates(ranked_results, jd_text, top_n=5):
    """Generate LLM summaries for the top N ranked candidates."""
    print(f"Generating LLM summaries for top {top_n} candidates...")
    for i, candidate in enumerate(ranked_results[:top_n]):
        print(f"  Summarizing candidate {i+1}/{top_n}...")
        summary = generate_candidate_summary(
            candidate.get("raw_text", ""),
            jd_text,
            candidate.get("composite_score", 0)
        )
        candidate["llm_summary"] = summary
    return ranked_results