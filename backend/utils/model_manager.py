import csv
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

from utils.preprocessing import preprocess_text


def _load_resume_corpus(dataset_path: Path) -> list[str]:
    if not dataset_path.exists():
        return []

    corpus: list[str] = []
    with open(dataset_path, "r", encoding="utf-8", errors="ignore", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            resume_text = row.get("Resume", "")
            processed = preprocess_text(resume_text)
            if processed:
                corpus.append(processed)
    return corpus


def load_or_train_vectorizer(dataset_path: Path, vectorizer_path: Path) -> TfidfVectorizer:
    if vectorizer_path.exists():
        with open(vectorizer_path, "rb") as model_file:
            return pickle.load(model_file)

    corpus = _load_resume_corpus(dataset_path)
    if not corpus:
        raise ValueError("No training corpus found in resume_dataset.csv.")

    vectorizer = TfidfVectorizer(max_features=7000, ngram_range=(1, 2))
    vectorizer.fit(corpus)

    vectorizer_path.parent.mkdir(parents=True, exist_ok=True)
    with open(vectorizer_path, "wb") as model_file:
        pickle.dump(vectorizer, model_file)

    return vectorizer
