import os
from src.parser import extract_text

# Anchor the path to this script's location, no matter where it's run from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(BASE_DIR, "data", "raw", "sample_resume.pdf")

print("Looking for file at:", pdf_path)

text = extract_text(pdf_path)
print(text[:500])


from src.extractor import parse_resume

result = parse_resume(text, filename="sample_resume.pdf")
print("\n--- Extracted info ---")
print("Email:", result["email"])
print("Phone:", result["phone"])
print("Skills:", result["skills"])
print("Experience years:", result["experience_years"])
print("Education:", result["education"])