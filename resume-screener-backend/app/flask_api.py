import os
import sys
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env from the project root explicitly (not cwd-dependent)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

from src.parser import extract_text
from src.extractor import parse_resume
from src.matcher import rank_resumes_semantic, rank_resumes_tfidf
from src.llm_summarizer import summarize_top_candidates

app = Flask(__name__)
CORS(app)  # allow React frontend to call this API

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Resume Screener API is running"})


@app.route("/api/screen", methods=["POST"])
def screen_resumes():
    """
    Main endpoint. Accepts:
    - jd_text: string (job description)
    - files: multiple PDF/DOCX resume files
    - top_n: int (optional, default 5)
    - use_llm: bool (optional, default true)
    - method: "semantic" or "tfidf" (optional, default "semantic")
    """
    try:
        # Parse form data
        jd_text = request.form.get("jd_text", "").strip()
        top_n = int(request.form.get("top_n", 5))
        use_llm = request.form.get("use_llm", "true").lower() == "true"
        method = request.form.get("method", "semantic")

        if not jd_text:
            return jsonify({"error": "Job description is required"}), 400

        files = request.files.getlist("resumes")
        if not files:
            return jsonify({"error": "At least one resume is required"}), 400

        # Parse all uploaded resumes
        resumes = []
        for file in files:
            if not file.filename:
                continue
            suffix = ".pdf" if file.filename.lower().endswith(".pdf") else ".docx"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                file.save(tmp.name)
                tmp_path = tmp.name

            try:
                raw_text = extract_text(tmp_path)
            finally:
                os.unlink(tmp_path)

            resume_data = parse_resume(raw_text, filename=file.filename)
            resumes.append(resume_data)

        # Rank resumes
        if method == "semantic":
            results = rank_resumes_semantic(jd_text, resumes, top_n=top_n)
        else:
            results = rank_resumes_tfidf(jd_text, resumes, top_n=top_n)

        # Add LLM summaries
        if use_llm:
            text_map = {r["filename"]: r["raw_text"] for r in resumes}
            for r in results:
                r["raw_text"] = text_map.get(r["filename"], "")
            results = summarize_top_candidates(results, jd_text, top_n=top_n)

        # Clean raw_text from response (too large to send)
        for r in results:
            r.pop("raw_text", None)

        return jsonify({
            "success": True,
            "total_uploaded": len(resumes),
            "results": results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/classify", methods=["POST"])
def classify_resume():
    """
    Classify a single resume into a job category.
    Accepts a PDF/DOCX file.
    """
    try:
        import joblib
        clf = joblib.load(os.path.join(BASE_DIR, "models", "classifier.pkl"))
        vec = joblib.load(os.path.join(BASE_DIR, "models", "vectorizer.pkl"))

        file = request.files.get("resume")
        if not file or not file.filename:
            return jsonify({"error": "No resume file provided"}), 400

        suffix = ".pdf" if file.filename.lower().endswith(".pdf") else ".docx"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        try:
            raw_text = extract_text(tmp_path)
        finally:
            os.unlink(tmp_path)

        X = vec.transform([raw_text])
        category = clf.predict(X)[0]
        probabilities = clf.predict_proba(X)[0]
        top_3_indices = probabilities.argsort()[-3:][::-1]
        top_3 = [
            {"category": clf.classes_[i], "confidence": round(float(probabilities[i]), 3)}
            for i in top_3_indices
        ]

        return jsonify({
            "success": True,
            "predicted_category": category,
            "top_3_predictions": top_3
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(debug=False, host="0.0.0.0", port=port)