import re

# Expanded and cleaned skills list — no single letters
SKILLS = [
    # Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "r",
    "scala", "kotlin", "swift", "php", "ruby", "go", "rust",
    # Web
    "react", "angular", "vue", "node.js", "express", "django", "flask",
    "html", "css", "tailwind", "bootstrap", "next.js", "fastapi",
    # Databases
    "mongodb", "mysql", "postgresql", "sqlite", "redis", "firebase",
    "oracle", "cassandra", "dynamodb",
    # Data & ML
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras",
    "matplotlib", "seaborn", "plotly", "opencv", "nltk", "spacy",
    "machine learning", "deep learning", "natural language processing",
    "computer vision", "data analysis", "data science", "data mining",
    "feature engineering", "model deployment",
    # AI
    "generative ai", "large language models", "transformers", "bert",
    "gpt", "llama", "langchain", "hugging face",
    # Cloud & DevOps
    "aws", "azure", "google cloud", "docker", "kubernetes", "git",
    "github", "gitlab", "jenkins", "terraform", "linux", "bash",
    # Tools
    "excel", "tableau", "power bi", "postman", "jira", "figma",
    "photoshop", "illustrator",
    # Other
    "sql", "rest api", "graphql", "microservices", "agile", "scrum"
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


def extract_skills(text):
    """Extract skills using word boundary matching to avoid false positives."""
    text_lower = text.lower()
    found = []
    for skill in SKILLS:
        # Use word boundaries for single-word skills
        if " " in skill:
            if skill in text_lower:
                found.append(skill)
        else:
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text_lower):
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


def parse_resume(text, filename=""):
    """Master function — returns structured resume dictionary."""
    skills = extract_skills(text)
    return {
        "filename": filename,
        "raw_text": text,
        "email": extract_email(text),
        "phone": extract_phone(text),
        "skills": skills,
        "experience_years": extract_experience_years(text),
        "education": extract_education(text),
        "skill_count": len(skills)
    }