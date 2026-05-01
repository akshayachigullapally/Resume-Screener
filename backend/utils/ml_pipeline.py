"""Dataset-trained ML helpers for resume screening."""

from __future__ import annotations

import csv
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .preprocessing import preprocess_text
from .sections import extract_resume_sections
from .skills import extract_skills


@dataclass(slots=True)
class ResumeModelBundle:
    vectorizer: TfidfVectorizer
    svm_model: SVC
    knn_model: KNeighborsClassifier
    rank_model: Pipeline
    labels: list[str]


def _load_dataset_rows(dataset_path: Path) -> list[dict[str, str]]:
    if not dataset_path.exists():
        return []

    rows: list[dict[str, str]] = []
    with open(dataset_path, "r", encoding="utf-8", errors="ignore", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            category = (row.get("Category") or row.get("category") or "").strip()
            resume_text = (row.get("Resume") or row.get("resume") or "").strip()
            if category and resume_text:
                rows.append({"category": category, "resume": resume_text})
    return rows


def _build_quality_target(text: str) -> float:
    tokens = preprocess_text(text).split()
    word_count = len(tokens)
    unique_ratio = len(set(tokens)) / max(word_count, 1)
    skill_count = len(extract_skills(text))
    sections = extract_resume_sections(text)
    section_presence = sum(1 for section_text in sections.values() if section_text.strip())
    experience_mentions = sum(1 for token in tokens if token.isdigit())
    achievement_signals = 1 if any(symbol in text for symbol in ("%", "$", "+")) else 0

    score = (
        min(word_count / 18.0, 100.0) * 0.25
        + min(unique_ratio * 100.0, 100.0) * 0.20
        + min(skill_count * 7.5, 100.0) * 0.30
        + min(section_presence * 22.5, 100.0) * 0.15
        + min((experience_mentions + achievement_signals) * 12.5, 100.0) * 0.10
    )
    return round(float(np.clip(score, 0.0, 100.0)), 2)


def _build_rank_feature_vector(text: str) -> list[float]:
    normalized = preprocess_text(text)
    tokens = normalized.split()
    word_count = len(tokens)
    unique_count = len(set(tokens))
    unique_ratio = unique_count / max(word_count, 1)
    skill_count = len(extract_skills(text))
    sections = extract_resume_sections(text)
    section_presence = sum(1 for section_text in sections.values() if section_text.strip())
    years = 0
    lowered = text.lower()
    for token in ("years", "year", "yrs", "yr"):
        if token in lowered:
            years += 1
    metric_hits = sum(1 for char in text if char in "%$")
    return [
        float(word_count),
        float(unique_count),
        float(unique_ratio),
        float(skill_count),
        float(section_presence),
        float(years),
        float(metric_hits),
    ]


def _train_rank_model(resume_texts: list[str]) -> Pipeline:
    features = np.asarray([_build_rank_feature_vector(text) for text in resume_texts], dtype=float)
    target = np.asarray([_build_quality_target(text) for text in resume_texts], dtype=float)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )
    model.fit(features, target)
    return model


def _vectorizer_from_dataset(corpus: list[str]) -> TfidfVectorizer:
    vectorizer = TfidfVectorizer(max_features=7000, ngram_range=(1, 2))
    vectorizer.fit(corpus)
    return vectorizer


def load_or_train_models(dataset_path: Path, model_dir: Path) -> ResumeModelBundle:
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = model_dir / "resume_models.pkl"
    metadata_path = model_dir / "resume_models_meta.json"

    if artifact_path.exists():
        with open(artifact_path, "rb") as artifact_file:
            payload = pickle.load(artifact_file)
            if isinstance(payload, ResumeModelBundle):
                return payload
            if isinstance(payload, dict):
                return ResumeModelBundle(**payload)

    rows = _load_dataset_rows(dataset_path)
    if not rows:
        raise ValueError("No training corpus found in resume_dataset.csv.")

    corpus = [preprocess_text(row["resume"]) for row in rows]
    labels = [row["category"] for row in rows]

    vectorizer = _vectorizer_from_dataset(corpus)
    features = vectorizer.transform(corpus)

    svm_model = SVC(kernel="linear", probability=True, class_weight="balanced", random_state=42)
    svm_model.fit(features, labels)

    knn_neighbors = max(1, min(7, len(labels)))
    knn_model = KNeighborsClassifier(n_neighbors=knn_neighbors, metric="cosine", algorithm="brute", weights="distance")
    knn_model.fit(features, labels)

    rank_model = _train_rank_model([row["resume"] for row in rows])

    bundle = ResumeModelBundle(
        vectorizer=vectorizer,
        svm_model=svm_model,
        knn_model=knn_model,
        rank_model=rank_model,
        labels=sorted(set(labels)),
    )

    with open(artifact_path, "wb") as artifact_file:
        pickle.dump(bundle, artifact_file)

    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(
            {
                "dataset_rows": len(rows),
                "labels": bundle.labels,
                "knn_neighbors": knn_neighbors,
            },
            metadata_file,
            indent=2,
            ensure_ascii=False,
        )

    return bundle


def build_rank_features(text: str) -> np.ndarray:
    return np.asarray([_build_rank_feature_vector(text)], dtype=float)


def predict_role(bundle: ResumeModelBundle, text: str) -> dict[str, Any]:
    processed = preprocess_text(text)
    if not processed.strip():
        return {"role": "Generalist", "confidence": 0.0, "alternatives": []}

    vector = bundle.vectorizer.transform([processed])
    predicted_role = bundle.svm_model.predict(vector)[0]
    probabilities = bundle.svm_model.predict_proba(vector)[0]
    class_names = list(bundle.svm_model.classes_)
    role_probabilities = sorted(
        ((class_name, float(prob)) for class_name, prob in zip(class_names, probabilities)),
        key=lambda item: item[1],
        reverse=True,
    )

    knn_probability = 0.0
    knn_role_probabilities: dict[str, float] = {}
    if hasattr(bundle.knn_model, "predict_proba"):
        knn_probs = bundle.knn_model.predict_proba(vector)[0]
        knn_classes = list(bundle.knn_model.classes_)
        knn_role_probabilities = {label: float(prob) for label, prob in zip(knn_classes, knn_probs)}
        knn_probability = knn_role_probabilities.get(predicted_role, 0.0)

    confidence = round(min((max(probabilities) * 0.7) + (knn_probability * 0.3), 1.0) * 100, 2)
    alternatives = [
        {"role": role, "score": round(prob * 100, 2)}
        for role, prob in role_probabilities[1:4]
    ]

    return {
        "role": predicted_role,
        "confidence": confidence,
        "alternatives": alternatives,
        "svm_role_probabilities": [
            {"role": role, "score": round(prob * 100, 2)} for role, prob in role_probabilities[:5]
        ],
        "knn_role_probability": round(knn_probability * 100, 2),
        "knn_role_probabilities": [
            {"role": role, "score": round(prob * 100, 2)}
            for role, prob in sorted(knn_role_probabilities.items(), key=lambda item: item[1], reverse=True)[:5]
        ],
    }


def predict_rank_score(bundle: ResumeModelBundle, text: str) -> float:
    features = build_rank_features(text)
    score = float(bundle.rank_model.predict(features)[0])
    return round(float(np.clip(score, 0.0, 100.0)), 2)


def normalize_similarity(score: float) -> float:
    return round(float(np.clip(score, 0.0, 1.0) * 100.0), 2)
