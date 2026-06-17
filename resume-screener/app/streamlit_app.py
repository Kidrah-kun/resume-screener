import streamlit as st
import os
import sys
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parser import extract_text
from src.extractor import parse_resume
from src.matcher import rank_resumes_semantic
from src.llm_summarizer import summarize_top_candidates

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="Resume Screener",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI-Powered Resume Screener")
st.markdown("Upload a job description and resumes to rank candidates automatically.")

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    top_n = st.slider("Number of top candidates to show", 1, 10, 5)
    use_llm = st.toggle("Generate AI summaries (Groq)", value=True)
    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown("1. Paste or upload a Job Description")
    st.markdown("2. Upload candidate resumes (PDF/DOCX)")
    st.markdown("3. AI ranks them by fit score")
    st.markdown("4. View detailed analysis per candidate")

# ── Job Description Input ─────────────────────────────────
st.header("1️⃣ Job Description")
jd_input_method = st.radio("Input method", ["Paste text", "Upload file"], horizontal=True)

jd_text = ""
if jd_input_method == "Paste text":
    jd_text = st.text_area(
        "Paste job description here",
        height=200,
        placeholder="We are looking for a Software Engineer with experience in React, Node.js..."
    )
else:
    jd_file = st.file_uploader("Upload JD (.txt)", type=["txt"])
    if jd_file:
        jd_text = jd_file.read().decode("utf-8")
        st.success("JD loaded successfully")
        st.text_area("Preview", jd_text[:500], height=150, disabled=True)

# ── Resume Upload ─────────────────────────────────────────
st.header("2️⃣ Upload Resumes")
uploaded_files = st.file_uploader(
    "Upload candidate resumes (PDF or DOCX)",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

# ── Run Analysis ──────────────────────────────────────────
if st.button("🚀 Screen Candidates", type="primary", use_container_width=True):
    if not jd_text.strip():
        st.error("Please provide a job description.")
    elif not uploaded_files:
        st.error("Please upload at least one resume.")
    else:
        with st.spinner("Parsing resumes..."):
            resumes = []
            for uploaded_file in uploaded_files:
                # Save to temp file to extract text
                suffix = ".pdf" if uploaded_file.name.endswith(".pdf") else ".docx"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                raw_text = extract_text(tmp_path)
                os.unlink(tmp_path)  # delete temp file

                resume_data = parse_resume(raw_text, filename=uploaded_file.name)
                resumes.append(resume_data)

            st.success(f"Parsed {len(resumes)} resumes")

        with st.spinner("Ranking candidates using semantic AI..."):
            results = rank_resumes_semantic(jd_text, resumes, top_n=top_n)

        if use_llm:
            with st.spinner("Generating AI summaries via Groq..."):
                # Add raw_text back for LLM
                text_map = {r["filename"]: r["raw_text"] for r in resumes}
                for r in results:
                    r["raw_text"] = text_map.get(r["filename"], "")
                results = summarize_top_candidates(results, jd_text, top_n=top_n)

        # ── Results ───────────────────────────────────────
        st.header("3️⃣ Ranked Candidates")
        st.markdown(f"Showing top **{len(results)}** candidates out of **{len(resumes)}** uploaded")

        for r in results:
            # Colour based on score
            score = r["composite_score"]
            if score >= 0.6:
                colour = "🟢"
            elif score >= 0.4:
                colour = "🟡"
            else:
                colour = "🔴"

            with st.expander(
                f"{colour} Rank #{r['rank']} — {r['filename']} | Score: {score:.2%}",
                expanded=(r["rank"] == 1)
            ):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Composite Score", f"{score:.2%}")
                with col2:
                    st.metric("Experience", f"{r['experience_years']} yrs")
                with col3:
                    st.metric("Skills Found", len(r["skills"]))

                # Score breakdown
                st.markdown("**Score Breakdown**")
                breakdown_cols = st.columns(3)
                sim_score = r.get("semantic_score") or r.get("tfidf_score") or 0
                with breakdown_cols[0]:
                    st.progress(sim_score, text=f"Semantic: {sim_score:.2%}")
                with breakdown_cols[1]:
                    st.progress(r["keyword_score"], text=f"Keywords: {r['keyword_score']:.2%}")
                with breakdown_cols[2]:
                    st.progress(r["experience_score"], text=f"Experience: {r['experience_score']:.2%}")

                # Skills
                if r["skills"]:
                    st.markdown("**Matched Skills**")
                    skills_html = " ".join([
                        f'<span style="background:#1e3a5f;color:white;padding:3px 10px;border-radius:12px;margin:3px;display:inline-block;font-size:13px">{s}</span>'
                        for s in r["skills"]
                    ])
                    st.markdown(skills_html, unsafe_allow_html=True)

                # LLM Summary
                if use_llm and "llm_summary" in r:
                    summary = r["llm_summary"]
                    st.markdown("---")
                    st.markdown("**🤖 AI Analysis**")

                    verdict = summary.get("recommendation", "")
                    verdict_colours = {
                        "Strong Fit": "🟢",
                        "Good Fit": "🟡",
                        "Partial Fit": "🟠",
                        "Weak Fit": "🔴",
                        "Manual Review": "⚪"
                    }
                    st.markdown(f"**Verdict:** {verdict_colours.get(verdict, '')} {verdict}")
                    st.markdown(f"*{summary.get('one_liner', '')}*")
                    st.markdown(f"_{summary.get('reasoning', '')}_")

                    ai_col1, ai_col2 = st.columns(2)
                    with ai_col1:
                        st.markdown("**✅ Strengths**")
                        for s in summary.get("strengths", []):
                            st.markdown(f"- {s}")
                    with ai_col2:
                        st.markdown("**⚠️ Gaps**")
                        for g in summary.get("gaps", []):
                            st.markdown(f"- {g}")