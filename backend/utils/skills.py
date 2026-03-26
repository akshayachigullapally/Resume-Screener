"""Skill extraction utilities backed by a curated database of technical keywords."""

from __future__ import annotations

import re
from typing import Dict, List

SKILLS_DB: List[str] = [
    "python",
    "java",
    "c++",
    "javascript",
    "react",
    "node",
    "sql",
    "machine learning",
    "deep learning",
    "nlp",
    "data science",
    "tensorflow",
    "pytorch",
    "flask",
    "django",
    "aws",
    "azure",
    "gcp",
    "docker",
    "kubernetes",
    "git",
    "linux",
    "html",
    "css",
]

_SPECIAL_CHAR_SKILLS = {skill for skill in SKILLS_DB if any(ch in skill for ch in "+#./")}


def _matches_skill(text: str, skill: str) -> bool:
    """Return True when the given skill appears as a standalone term in text."""

    if skill in _SPECIAL_CHAR_SKILLS:
        return skill in text
    pattern = rf"\b{re.escape(skill)}\b"
    return re.search(pattern, text) is not None


def extract_skills(text: str) -> List[str]:
    """Extract only skills from SKILLS_DB, yielding deterministic, lowercase matches."""

    lowered = text.lower()
    found: List[str] = []
    for skill in SKILLS_DB:
        if _matches_skill(lowered, skill):
            found.append(skill)
    return sorted(set(found))


def get_skill_match_details(job_skills: List[str], resume_skills: List[str]) -> Dict[str, object]:
    """Compare job requirement skills to resume skills and compute match insights."""

    job_set = set(job_skills)
    resume_set = set(resume_skills)

    matched = sorted(job_set.intersection(resume_set))
    missing = sorted(job_set.difference(resume_set))
    match_percentage = round((len(matched) / len(job_set)) * 100, 2) if job_set else 0.0

    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "skill_match_percentage": match_percentage,
    }
