import os
import json
import pandas as pd
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.extractor import parse_resume

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def process_kaggle_dataset(csv_filename, output_filename):
    csv_path = os.path.join(BASE_DIR, "data", "raw", csv_filename)
    output_dir = os.path.join(BASE_DIR, "data", "processed")
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    processed = []

    for idx, row in df.iterrows():
        raw_text = str(row.get("Resume_str", ""))
        category = str(row.get("Category", "unknown"))

        resume_data = parse_resume(raw_text, filename=f"resume_{idx}")
        resume_data["category"] = category
        processed.append(resume_data)

        if idx % 100 == 0:
            print(f"Processed {idx}/{len(df)} resumes...")

    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, "w") as f:
        json.dump(processed, f, indent=2)

    print(f"Done. Saved {len(processed)} resumes to {output_path}")


if __name__ == "__main__":
    process_kaggle_dataset(
        csv_filename="Resume.csv",         
        output_filename="processed_resumes.json"
    )
