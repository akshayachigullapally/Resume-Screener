"""Scoring helpers for computing weighted resume scores."""

from __future__ import annotations

from typing import Dict

SKILLS_WEIGHT = 0.4
EXPERIENCE_WEIGHT = 0.3
PROJECTS_WEIGHT = 0.2
SEMANTIC_WEIGHT = 0.1


def _extract_section_score(section_scores: Dict[str, Dict[str, float]], key: str) -> float:
    section = section_scores.get(key) or {}
    return float(section.get("score", 0.0))


def compute_weighted_overall_score(
    section_scores: Dict[str, Dict[str, float]],
    semantic_percentage: float,
) -> Dict[str, float]:
    """Combine section metrics and semantic similarity into a normalized breakdown."""

    skills_score = _extract_section_score(section_scores, "skills")
    experience_score = _extract_section_score(section_scores, "experience")
    project_score = _extract_section_score(section_scores, "projects")
    semantic_score = max(0.0, min(float(semantic_percentage), 100.0))

    overall_score = (
        (skills_score * SKILLS_WEIGHT)
        + (experience_score * EXPERIENCE_WEIGHT)
        + (project_score * PROJECTS_WEIGHT)
        + (semantic_score * SEMANTIC_WEIGHT)
    )

    return {
        "overall_score": round(overall_score, 2),
        "skills_score": round(skills_score, 2),
        "experience_score": round(experience_score, 2),
        "project_score": round(project_score, 2),
        "semantic_score": round(semantic_score, 2),
    }
