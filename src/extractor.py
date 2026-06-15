import spacy
import re

nlp = spacy.load("en_core_web_sm")

SKILLS = [
    "python", "java", "javascript", "typescript", "react", "node.js",
    "express", "mongodb", "mysql", "postgresql", "pandas", "numpy",
    "scikit-learn", "tensorflow", "pytorch", "machine learning",
    "deep learning", "nlp", "computer vision", "git", "docker",
    "kubernetes", "aws", "azure", "flask", "django", "html", "css",
    "tailwind", "sql", "tableau", "power bi", "excel", "c++", "c",
    "data analysis", "data science", "llm", "generative ai"
]

EDUCATION_KEYWORDS = [
    "b.tech", "btech", "bachelor", "b.e", "m.tech", "mtech",
    "master", "mba", "phd", "diploma", "12th", "10th",
    "undergraduate", "postgraduate", "degree"
]


def extract_skills(text):
    text_lower = text.lower()
    found = [skill for skill in SKILLS if skill in text_lower]
    return list(set(found))


def extract_experience_years(text):
    patterns = [
        r'(\d+)\+?\s*years?\s*of\s*experience',
        r'experience\s*of\s*(\d+)\+?\s*years?',
        r'(\d+)\+?\s*years?\s*experience',
    ]
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return int(match.group(1))
    return 0


def extract_education(text):
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


def parse_resume(text, filename=""):
    return {
        "filename": filename,
        "raw_text": text,
        "email": extract_email(text),
        "phone": extract_phone(text),
        "skills": extract_skills(text),
        "experience_years": extract_experience_years(text),
        "education": extract_education(text),
        "skill_count": len(extract_skills(text))
    }