import os
import sys
import json
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_data(json_path):
    """Load processed resumes and return texts + labels."""
    with open(json_path, "r") as f:
        resumes = json.load(f)
    texts = [r["raw_text"] for r in resumes]
    labels = [r["category"] for r in resumes]
    return texts, labels


def train_classifier(json_path, model_save_dir):
    """Train and evaluate resume category classifiers."""
    os.makedirs(model_save_dir, exist_ok=True)

    print("Loading data...")
    texts, labels = load_data(json_path)
    print(f"Loaded {len(texts)} resumes across {len(set(labels))} categories")

    # TF-IDF features
    print("Vectorising text...")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=10000,
        ngram_range=(1, 2)
    )
    X = vectorizer.fit_transform(texts)
    y = np.array(labels)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = {}

    # --- Logistic Regression ---
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_cv = cross_val_score(lr, X, y, cv=5, scoring="f1_weighted")

    print(f"LR Test Accuracy: {(lr_pred == y_test).mean():.4f}")
    print(f"LR CV F1 (5-fold): {lr_cv.mean():.4f} ± {lr_cv.std():.4f}")
    print(classification_report(y_test, lr_pred))
    results["logistic_regression"] = {
        "accuracy": float((lr_pred == y_test).mean()),
        "cv_f1_mean": float(lr_cv.mean()),
        "cv_f1_std": float(lr_cv.std())
    }

    # --- Random Forest ---
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_cv = cross_val_score(rf, X, y, cv=5, scoring="f1_weighted")

    print(f"RF Test Accuracy: {(rf_pred == y_test).mean():.4f}")
    print(f"RF CV F1 (5-fold): {rf_cv.mean():.4f} ± {rf_cv.std():.4f}")
    print(classification_report(y_test, rf_pred))
    results["random_forest"] = {
        "accuracy": float((rf_pred == y_test).mean()),
        "cv_f1_mean": float(rf_cv.mean()),
        "cv_f1_std": float(rf_cv.std())
    }

    # --- Confusion matrix for best model ---
    best_model = lr if results["logistic_regression"]["accuracy"] > results["random_forest"]["accuracy"] else rf
    best_pred = lr_pred if best_model == lr else rf_pred
    best_name = "Logistic Regression" if best_model == lr else "Random Forest"

    print(f"\nBest model: {best_name}")

    # Plot confusion matrix
    cm = confusion_matrix(y_test, best_pred, labels=sorted(set(labels)))
    plt.figure(figsize=(16, 12))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=sorted(set(labels)),
        yticklabels=sorted(set(labels))
    )
    plt.title(f"Confusion Matrix — {best_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "reports", "confusion_matrix.png"))
    plt.show()
    print("Confusion matrix saved to reports/confusion_matrix.png")

    # Save best model + vectorizer
    joblib.dump(best_model, os.path.join(model_save_dir, "classifier.pkl"))
    joblib.dump(vectorizer, os.path.join(model_save_dir, "vectorizer.pkl"))
    print(f"Model saved to {model_save_dir}/classifier.pkl")

    # Save results summary
    with open(os.path.join(model_save_dir, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return best_model, vectorizer, results


if __name__ == "__main__":
    train_classifier(
        json_path=os.path.join(BASE_DIR, "data", "processed", "processed_resumes.json"),
        model_save_dir=os.path.join(BASE_DIR, "models")
    )