import fitz  # PyMuPDF
from docx import Document
import os
import json

def extract_text_from_pdf(file_path):
    """Extract raw text from a PDF file using PyMuPDF."""
    text = ""
    try:
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
    return text.strip()


def extract_text_from_docx(file_path):
    """Extract raw text from a DOCX file."""
    text = ""
    try:
        doc = Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        print(f"Error reading DOCX {file_path}: {e}")
    return text.strip()


def extract_text(file_path):
    """Route to the correct extractor based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    else:
        print(f"Unsupported file type: {ext}")
        return ""


def save_as_json(data, output_path):
    """Save extracted resume data as a JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)