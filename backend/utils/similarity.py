from __future__ import annotations

from collections import Counter
import re
from typing import Iterable, List

import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .skills import extract_skills, get_skill_match_details


SENTENCE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_SENTENCE_MODEL: SentenceTransformer | None = None


def _get_sentence_model() -> SentenceTransformer:
    global _SENTENCE_MODEL
    if _SENTENCE_MODEL is None:
        _SENTENCE_MODEL = SentenceTransformer(SENTENCE_MODEL_NAME)
    return _SENTENCE_MODEL


def compute_similarity_scores(
    vectorizer: TfidfVectorizer, job_description: str, resume_texts: list[str]
) -> list[float]:
    job_vector = vectorizer.transform([job_description])
    resume_vectors = vectorizer.transform(resume_texts)
    similarities = cosine_similarity(job_vector, resume_vectors).flatten()
    return similarities.tolist()


def compute_semantic_similarity_scores(job_description: str, resume_texts: List[str]) -> List[float]:
    if not resume_texts:
        return []

    model = _get_sentence_model()
    inputs = [job_description, *resume_texts]
    embeddings = model.encode(inputs, convert_to_tensor=True, normalize_embeddings=True)
    job_embedding = embeddings[0]
    resume_embeddings = embeddings[1:]
    similarities = util.cos_sim(job_embedding, resume_embeddings).cpu().numpy().flatten()
    return [round(float(score), 4) for score in similarities]


def combine_similarity_scores(
    tfidf_scores: Iterable[float], semantic_scores: Iterable[float], tfidf_weight: float = 0.6
) -> List[float]:
    weights = []
    semantic_weight = max(0.0, min(1.0, 1 - tfidf_weight))
    for tfidf_score, semantic_score in zip(tfidf_scores, semantic_scores):
        combined = (tfidf_weight * tfidf_score) + (semantic_weight * semantic_score)
        weights.append(round(float(np.clip(combined, 0.0, 1.0)), 4))
    return weights



def extract_keyword_overlap(job_description: str, resume_text: str, top_n: int = 12) -> list[str]:
    jd_tokens = set(job_description.split())
    resume_tokens = resume_text.split()
    token_counts = Counter(token for token in resume_tokens if token in jd_tokens and len(token) > 2)
    return [token for token, _ in token_counts.most_common(top_n)]


def extract_years_of_experience(text: str) -> int:
    matches = re.findall(r"(\d+)\s*\+?\s*(?:years|year|yrs|yr)", text.lower())
    if not matches:
        return 0
    return max(int(match) for match in matches)


def get_experience_match_percentage(required_years: int, candidate_years: int) -> float:
    if required_years <= 0:
        return 100.0
    return round(min((candidate_years / required_years) * 100, 100), 2)


def compute_resume_strength_score(
    similarity_score: float,
    skill_match_percentage: float,
    experience_match_percentage: float,
    resume_raw_text: str,
) -> float:
    text = resume_raw_text.lower()
    section_hits = sum(
        1
        for section in ["experience", "education", "project", "skills", "certification"]
        if section in text
    )
    structure_score = min((section_hits / 5) * 100, 100)

    quantified_achievement = 100.0 if re.search(r"\b\d+%|\$\d+|\b\d+\b", resume_raw_text) else 40.0

    final_score = (
        (similarity_score * 0.45)
        + (skill_match_percentage * 0.3)
        + (experience_match_percentage * 0.15)
        + (structure_score * 0.05)
        + (quantified_achievement * 0.05)
    )
    return round(final_score, 2)
