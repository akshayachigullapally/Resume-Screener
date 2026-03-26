"""Utility helpers for generating resume feedback and explanations."""

from __future__ import annotations

from typing import Iterable, List

from .similarity import extract_keyword_overlap

SECTION_KEYWORDS = {
    "experience": ["experience", "employment", "work history"],
    "projects": ["project", "case study", "portfolio"],
    "skills": ["skill", "competency", "tools"],
    "education": ["education", "academics", "degree"],
    "achievements": ["achievement", "accomplishment", "impact"],
}


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

    if missing_skills:
        suggestions.append(
            "Highlight or add experience with: " + ", ".join(missing_skills[:8])
        )

    if not _section_present(lowered, SECTION_KEYWORDS["projects"]):
        suggestions.append("Add a projects section to show how you applied core skills.")

    if not _section_present(lowered, SECTION_KEYWORDS["achievements"]):
        suggestions.append("Quantify achievements (metrics, percentages, savings, impact).")

    if not _section_present(lowered, SECTION_KEYWORDS["education"]):
        suggestions.append("Include an education section with degree, university, and graduation year.")

    if "team" not in lowered and "lead" not in lowered:
        suggestions.append("Mention collaboration or leadership highlights to show soft skills.")

    jd_keywords = extract_keyword_overlap(job_description.lower(), resume_raw_text.lower(), top_n=6)
    if jd_keywords:
        suggestions.append(
            f"Ensure these JD keywords appear naturally in your resume summary: {', '.join(jd_keywords)}."
        )

    if "summary" not in lowered and "objective" not in lowered:
        suggestions.append("Add a short professional summary tailored to the job description.")

    deduped: list[str] = []
    for suggestion in suggestions:
        if suggestion not in deduped:
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
