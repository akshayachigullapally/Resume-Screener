import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def _ensure_nltk_data() -> None:
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    ]
    for resource_path, download_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(download_name, quiet=True)


_ensure_nltk_data()
STOP_WORDS = set(stopwords.words("english"))


def preprocess_text(text: str) -> str:
    if not text:
        return ""

    normalized = text.lower()
    normalized = re.sub(r"[^a-z\s]", " ", normalized)
    try:
        tokens = word_tokenize(normalized)
    except LookupError:
        tokens = re.findall(r"\b[a-z]+\b", normalized)
    filtered_tokens = [token for token in tokens if token.isalpha() and token not in STOP_WORDS]
    return " ".join(filtered_tokens)
