"""Extract candidate names from resume text."""

from __future__ import annotations

import re
from typing import Optional


def extract_candidate_name(resume_text: str) -> Optional[str]:
    """Extract candidate name from resume text using heuristics."""
    lines = resume_text.strip().split("\n")
    
    # Check first 10 lines for name (typically at the top)
    for line in lines[:10]:
        line = line.strip()
        if not line or len(line) < 2:
            continue
        
        if any(keyword in line.lower() for keyword in ["email", "phone", "linkedin", "address", "@", "http", "linkedin.com", "github"]):
            continue
        
        word_count = len(line.split())
        if word_count > 6:
            continue
        
        has_special = bool(re.search(r"[0-9@#$%^&*()_+=\[\]{};:'\",.<>?/\\|`~\-]", line))
        if has_special:
            continue
        
        if re.match(r"^[A-Za-z\s\-'\.]+$", line) and word_count >= 1 and word_count <= 4:
            return line.strip()
    
    # Fall back to searching entire document
    for line in lines:
        line = line.strip()
        if not line or len(line) < 2:
            continue
        
        if any(keyword in line.lower() for keyword in ["email", "phone", "linkedin", "address", "@", "http", "summary", "objective", "experience", "education", "skills", "projects", "certification"]):
            continue
        
        if re.match(r"^[A-Za-z\s\-'\.]+$", line):
            word_count = len(line.split())
            if 1 <= word_count <= 5 and len(line) < 60:
                return line.strip()
    
    return None
