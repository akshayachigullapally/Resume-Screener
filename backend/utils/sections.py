"""Resume section parsing, scoring, and role-fit prediction helpers."""

from __future__ import annotations

import re
from typing import Dict, List

from .skills import SKILLS_DB

SECTION_ALIASES: Dict[str, List[str]] = {
    "skills": ["skills", "technical skills", "core skills", "competencies"],
    "experience": [
        "experience",
        "work experience",
        "professional experience",
        "employment history",
        "work history",
    ],
    "projects": ["projects", "project experience", "case studies", "portfolio"],
    "education": ["education", "academic background", "qualifications"],
}

SECTION_SEARCH_TERMS: Dict[str, List[str]] = {
    name: [alias.lower() for alias in aliases]
    for name, aliases in SECTION_ALIASES.items()
}

ACTION_VERBS = {
    "built",
    "designed",
    "implemented",
    "deployed",
    "optimized",
    "trained",
    "architected",
    "launched",
}

DEGREE_KEYWORDS = {
    "bachelor",
    "master",
    "phd",
    "b.sc",
    "m.sc",
    "b.tech",
    "m.tech",
    "bs",
    "ms",
}

ROLE_PROFILES: Dict[str, List[str]] = {
    "Backend Developer": ["python", "java", "c++", "node", "sql", "docker", "aws", "git"],
    "Frontend Developer": ["javascript", "react", "css", "html"],
    "Full Stack Developer": ["javascript", "react", "node", "python", "sql"],
    "ML Engineer": ["python", "machine learning", "deep learning", "tensorflow", "pytorch", "aws"],
    "Data Scientist": ["python", "data science", "nlp", "sql", "machine learning"],
    "DevOps Engineer": ["linux", "docker", "kubernetes", "aws", "azure", "git"],
}

ROLE_KEYWORDS: Dict[str, List[str]] = {
    "Backend Developer": ["api", "microservice", "database"],
    "Frontend Developer": ["ui", "ux", "responsive", "web app"],
    "Full Stack Developer": ["end-to-end", "full stack"],
    "ML Engineer": ["model", "training", "inference", "ml"],
    "Data Scientist": ["analysis", "insights", "statistics"],
    "DevOps Engineer": ["pipeline", "ci/cd", "infrastructure", "automation"],
}

_METRIC_PATTERN = re.compile(r"(\d+%|\b\d+\s+(?:users|clients|models|projects|deployments))", re.IGNORECASE)


def _normalize_heading(line: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9+&/ ]", " ", line).strip().lower()
    return re.sub(r"\s+", " ", cleaned)


def _detect_section(line: str) -> str | None:
    normalized = _normalize_heading(line).rstrip(":")
    for section, aliases in SECTION_SEARCH_TERMS.items():
        for alias in aliases:
            if normalized.startswith(alias):
                return section
    return None


def extract_resume_sections(text: str) -> Dict[str, str]:
    sections: Dict[str, List[str]] = {name: [] for name in SECTION_ALIASES}
    current = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        detected = _detect_section(line)
        if detected:
            current = detected
            continue
        if current:
            sections[current].append(line)
    return {name: "\n".join(lines).strip() for name, lines in sections.items()}


def extract_section_texts(text: str) -> Dict[str, str]:
    """Backward compatible alias for resume section extraction."""

    return extract_resume_sections(text)


def section_present(text: str, section: str) -> bool:
    lowered = text.lower()
    keywords = SECTION_SEARCH_TERMS.get(section, [])
    return any(keyword in lowered for keyword in keywords)


def _word_coverage(text: str, target_words: int) -> float:
    if not text:
        return 0.0
    return min(len(text.split()) / max(target_words, 1), 1.0)


def score_resume_sections(
    sections: Dict[str, str],
    skill_match_percentage: float,
    experience_match_percentage: float,
    resume_skills: List[str],
) -> Dict[str, Dict[str, float | str]]:
    skills_text = sections.get("skills", "")
    experience_text = sections.get("experience", "")
    projects_text = sections.get("projects", "")
    education_text = sections.get("education", "")

    skills_depth = _word_coverage(skills_text, 80)
    skills_score = round(min((skill_match_percentage * 0.85) + (skills_depth * 15), 100), 2)

    exp_depth = _word_coverage(experience_text, 220)
    exp_metric_bonus = 10 if _METRIC_PATTERN.search(experience_text) else 0
    experience_score = round(min((experience_match_percentage * 0.7) + (exp_depth * 20) + exp_metric_bonus, 100), 2)

    project_depth = _word_coverage(projects_text, 150)
    verb_hits = sum(projects_text.lower().count(verb) for verb in ACTION_VERBS)
    skill_alignment = min(len(resume_skills) / 6, 1.0)
    projects_score = round(min((project_depth * 60) + (verb_hits * 4) + (skill_alignment * 20), 100), 2)

    education_depth = _word_coverage(education_text, 60)
    degree_bonus = 30 if any(keyword in education_text.lower() for keyword in DEGREE_KEYWORDS) else 0
    education_score = round(min((education_depth * 70) + degree_bonus, 100), 2)

    score_payload: Dict[str, Dict[str, float]] = {
        "skills": {"score": skills_score},
        "experience": {"score": experience_score},
        "projects": {"score": projects_score},
        "education": {"score": education_score},
    }

    score_payload["skills_score"] = {"score": skills_score}
    score_payload["experience_score"] = {"score": experience_score}
    score_payload["project_score"] = {"score": projects_score}

    return score_payload


def predict_role_fit(
    resume_skills: List[str],
    sections: Dict[str, str],
    resume_text: str,
) -> Dict[str, object]:
    lowered = resume_text.lower()
    scores: Dict[str, float] = {}
    resume_skill_set = set(resume_skills)

    for role, role_skills in ROLE_PROFILES.items():
        skill_overlap = len(resume_skill_set.intersection(role_skills))
        keyword_hits = sum(1 for keyword in ROLE_KEYWORDS.get(role, []) if keyword in lowered)
        scores[role] = (skill_overlap * 2) + keyword_hits

    best_role = max(scores, key=scores.get, default="Generalist")
    best_score = scores.get(best_role, 0.0)

    if best_score == 0:
        return {"role": "Generalist", "confidence": 0.0, "alternatives": []}

    total = sum(scores.values()) or 1.0
    confidence = round((best_score / total) * 100, 2)
    ordered = sorted(
        (
            {"role": role, "score": score}
            for role, score in scores.items()
            if score > 0 and role != best_role
        ),
        key=lambda item: item["score"],
        reverse=True,
    )
    return {
        "role": best_role,
        "confidence": confidence,
        "alternatives": ordered[:3],
    }
