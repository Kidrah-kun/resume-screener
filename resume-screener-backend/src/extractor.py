import re
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download NLTK data (silent if already present)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

STOP_WORDS = set(stopwords.words("english"))

# Load spaCy NER model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Warning: spaCy model not found. Run: python -m spacy download en_core_web_sm")
    nlp = None

# Expanded and cleaned skills list — no single letters
# Combines manually curated skills with categories from the O*NET Skills Database
# (https://www.onetonline.org/) for comprehensive skill taxonomy coverage.
SKILLS = [
    # ── Programming Languages ──
    "python", "java", "javascript", "typescript", "c++", "c#", "r",
    "scala", "kotlin", "swift", "php", "ruby", "go", "rust",
    # ── Web Frameworks & Frontend ──
    "react", "angular", "vue", "node.js", "express", "django", "flask",
    "html", "css", "tailwind", "bootstrap", "next.js", "fastapi",
    # ── Databases ──
    "mongodb", "mysql", "postgresql", "sqlite", "redis", "firebase",
    "oracle", "cassandra", "dynamodb",
    # ── Data Science & ML ──
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras",
    "matplotlib", "seaborn", "plotly", "opencv", "nltk", "spacy",
    "machine learning", "deep learning", "natural language processing",
    "computer vision", "data analysis", "data science", "data mining",
    "feature engineering", "model deployment",
    # ── AI / LLM ──
    "generative ai", "large language models", "transformers", "bert",
    "gpt", "llama", "langchain", "hugging face",
    # ── Cloud & DevOps ──
    "aws", "azure", "google cloud", "docker", "kubernetes", "git",
    "github", "gitlab", "jenkins", "terraform", "linux", "bash",
    # ── Tools ──
    "excel", "tableau", "power bi", "postman", "jira", "figma",
    "photoshop", "illustrator",
    # ── General Technical ──
    "sql", "rest api", "graphql", "microservices", "agile", "scrum",
    # ── O*NET Skills Database Categories ──
    # Sourced from O*NET OnLine (onetonline.org) skill taxonomy
    "critical thinking", "complex problem solving", "active learning",
    "analytical thinking", "project management", "systems analysis",
    "quality assurance", "technical writing", "data visualization",
    "statistical analysis", "business analysis", "requirements analysis",
    "software development", "systems design", "database administration",
    "network security", "cloud computing", "devops",
    "user experience", "customer service", "team leadership",
    "strategic planning", "risk management", "financial analysis",
    "supply chain management", "digital marketing", "content management",
    "human resources", "talent acquisition", "performance management",
    "operations management", "process improvement", "lean six sigma",
]

# Sort by length descending so multi-word skills match before substrings
SKILLS = sorted(SKILLS, key=len, reverse=True)

EDUCATION_KEYWORDS = [
    "b.tech", "btech", "bachelor of technology",
    "bachelor of science", "b.sc", "bsc",
    "bachelor of arts", "b.a", "b.e", "be",
    "m.tech", "mtech", "master of technology",
    "master of science", "m.sc", "msc",
    "master of business administration", "mba",
    "phd", "ph.d", "doctorate",
    "diploma", "associate degree",
    "12th", "intermediate", "higher secondary",
    "10th", "matriculation", "secondary",
    "undergraduate", "postgraduate", "bachelor", "master", "degree"
]

# Resume sections we expect to find for completeness scoring
SECTION_HEADINGS = [
    "experience", "work experience", "professional experience", "employment",
    "education", "academic", "qualification",
    "skills", "technical skills", "core competencies",
    "projects", "project",
    "summary", "objective", "profile", "about",
    "certifications", "certification", "awards"
]


def extract_name_ner(text):
    """Extract candidate name using spaCy Named Entity Recognition."""
    if nlp is None:
        return ""
    # Process only the first 500 chars (name is typically at the top)
    doc = nlp(text[:500])
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return ""


def extract_entities_ner(text):
    """Extract named entities from resume using spaCy NER.
    Returns a dict of entity types and their values.
    """
    if nlp is None:
        return {"persons": [], "organizations": [], "locations": [], "dates": []}

    doc = nlp(text[:3000])  # limit text to avoid slow processing

    entities = {
        "persons": [],
        "organizations": [],
        "locations": [],
        "dates": []
    }

    for ent in doc.ents:
        if ent.label_ == "PERSON" and ent.text not in entities["persons"]:
            entities["persons"].append(ent.text)
        elif ent.label_ == "ORG" and ent.text not in entities["organizations"]:
            entities["organizations"].append(ent.text)
        elif ent.label_ in ("GPE", "LOC") and ent.text not in entities["locations"]:
            entities["locations"].append(ent.text)
        elif ent.label_ == "DATE" and ent.text not in entities["dates"]:
            entities["dates"].append(ent.text)

    return entities


def preprocess_text_nltk(text):
    """Preprocess resume text using NLTK tokenization and stopword removal."""
    tokens = word_tokenize(text.lower())
    filtered = [t for t in tokens if t.isalnum() and t not in STOP_WORDS]
    return " ".join(filtered)


def extract_skills(text):
    """Extract skills using word boundary matching to avoid false positives.
    Uses NLTK-preprocessed text for cleaner matching."""
    # Preprocess with NLTK to remove stopwords and normalize
    cleaned_text = preprocess_text_nltk(text)
    text_lower = text.lower()  # Keep original for multi-word phrase matching
    found = []
    for skill in SKILLS:
        # Use word boundaries for single-word skills
        if " " in skill:
            if skill in text_lower:
                found.append(skill)
        else:
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, cleaned_text) or re.search(pattern, text_lower):
                found.append(skill)
    return list(set(found))


def extract_experience_years(text):
    """Extract years of experience using multiple regex patterns."""
    patterns = [
        r'(\d+)\+?\s*years?\s*of\s*experience',
        r'experience\s*of\s*(\d+)\+?\s*years?',
        r'(\d+)\+?\s*years?\s*experience',
        r'(\d+)\+?\s*yrs?\s*of\s*experience',
        r'over\s*(\d+)\s*years?',
    ]
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return int(match.group(1))
    return 0


def extract_education(text):
    """Extract highest education level mentioned."""
    text_lower = text.lower()
    for keyword in EDUCATION_KEYWORDS:
        if keyword in text_lower:
            return keyword
    return "not specified"


def extract_email(text):
    match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    return match.group(0) if match else ""


def extract_phone(text):
    match = re.search(r'[\+\(]?[1-9][0-9 \-\(\)]{7,}[0-9]', text)
    return match.group(0) if match else ""


def compute_section_completeness(text, skills, experience_years, email, phone, education):
    """
    Compute a completeness score for the resume based on
    how many expected sections and fields are present.
    Returns a score between 0.0 and 1.0.
    """
    text_lower = text.lower()
    checks = {
        "has_email": bool(email),
        "has_phone": bool(phone),
        "has_education": education != "not specified",
        "has_skills": len(skills) >= 2,
        "has_experience": experience_years > 0 or any(
            kw in text_lower for kw in ["experience", "employment", "work history"]
        ),
        "has_sections": sum(
            1 for heading in SECTION_HEADINGS if heading in text_lower
        ) >= 3,
        "has_summary": any(
            kw in text_lower for kw in ["summary", "objective", "profile", "about me"]
        ),
        "sufficient_length": len(text.split()) >= 100,
    }
    return sum(checks.values()) / len(checks)


def parse_resume(text, filename=""):
    """Master function — returns structured resume dictionary."""
    skills = extract_skills(text)
    email = extract_email(text)
    phone = extract_phone(text)
    education = extract_education(text)
    experience_years = extract_experience_years(text)

    # spaCy NER entities
    ner_entities = extract_entities_ner(text)
    candidate_name = extract_name_ner(text)

    # Section completeness score
    completeness = compute_section_completeness(
        text, skills, experience_years, email, phone, education
    )

    return {
        "filename": filename,
        "raw_text": text,
        "candidate_name": candidate_name,
        "email": email,
        "phone": phone,
        "skills": skills,
        "experience_years": experience_years,
        "education": education,
        "skill_count": len(skills),
        "section_completeness": round(completeness, 4),
        "ner_entities": ner_entities,
    }