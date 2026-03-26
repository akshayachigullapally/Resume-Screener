"""Utility helpers for generating resume feedback and explanations."""

from __future__ import annotations

import re
from typing import Iterable, List

from .skills import extract_skills

SECTION_KEYWORDS = {
    "experience": ["experience", "employment", "work history"],
    "projects": ["project", "case study", "portfolio"],
    "skills": ["skill", "competency", "tools"],
    "education": ["education", "academics", "degree"],
    "achievements": ["achievement", "accomplishment", "impact"],
}

SKILL_PROJECT_HINTS = {
    "machine learning": "Include projects related to machine learning to mirror the JD.",
    "data science": "Discuss data science case studies that relate to the role.",
    "aws": "Add cloud deployments or migrations completed on AWS.",
    "azure": "Document Azure-based solutions you have implemented.",
    "gcp": "Mention workloads you have built or migrated on GCP.",
}

_METRIC_PATTERN = re.compile(r"(\d+%|\b\d+\s+(?:users|clients|models|projects))", re.IGNORECASE)


def _section_present(text: str, keywords: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def generate_resume_suggestions(
    job_description: str,
    resume_raw_text: str,
    missing_skills: List[str],
) -> List[str]:
    suggestions: list[str] = []
    lowered = resume_raw_text.lower()
    job_skills = extract_skills(job_description)
    resume_skills = extract_skills(resume_raw_text)
    matched_skills = sorted(set(job_skills).intersection(resume_skills))

    for skill in missing_skills[:5]:
        display = skill.title()
        suggestions.append(f"Add experience with {display} (required in job description).")
        hint = SKILL_PROJECT_HINTS.get(skill)
        if hint:
            suggestions.append(hint)

    if matched_skills:
        highlight = ", ".join(skill.title() for skill in matched_skills[:3])
        suggestions.append(f"Highlight skills like {highlight} more clearly in your resume.")

    if not _METRIC_PATTERN.search(resume_raw_text):
        suggestions.append("Add measurable achievements (e.g., improved accuracy by 20%).")

    if missing_skills and not _section_present(lowered, SECTION_KEYWORDS["projects"]):
        focus_skill = missing_skills[0].title()
        suggestions.append(f"Include a project that demonstrates {focus_skill} in practice.")

    if not _section_present(lowered, SECTION_KEYWORDS["skills"]):
        suggestions.append("Create a dedicated skills section so recruiters can scan key tools quickly.")

    deduped: list[str] = []
    for suggestion in suggestions:
        if suggestion and suggestion not in deduped:
            deduped.append(suggestion)
    return deduped[:6]


def generate_improvement_plan(
    resume_text: str,
    job_description: str,
    missing_skills: List[str],
) -> dict:
    section_gaps = {
        section: _section_present(resume_text, keywords)
        for section, keywords in SECTION_KEYWORDS.items()
    }

    actionable = generate_resume_suggestions(job_description, resume_text, missing_skills)

    return {
        "missing_sections": [section for section, present in section_gaps.items() if not present],
        "suggestions": actionable,
    }


def build_candidate_explanation(
    candidate_name: str,
    overall_score: float,
    semantic_score: float,
    matched_skills: List[str],
    missing_skills: List[str],
    experience_match: float,
) -> str:
    reasons: list[str] = []
    if matched_skills:
        reasons.append(f"strong overlap in required skills ({', '.join(matched_skills[:5])})")
    if semantic_score >= 0.6:
        reasons.append("high semantic similarity to the JD")
    if experience_match >= 80:
        reasons.append("experience meets the requirement")

    gaps: list[str] = []
    if missing_skills:
        gaps.append(f"should emphasize {', '.join(missing_skills[:4])}")
    if experience_match < 50:
        gaps.append("needs clearer experience details")

    explanation = f"{candidate_name} ranks here due to "
    explanation += ", ".join(reasons) if reasons else "overall strong relevance"
    if gaps:
        explanation += ", but " + "; ".join(gaps)
    explanation += f". Final score: {overall_score:.1f}%"
    return explanation
