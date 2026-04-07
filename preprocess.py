"""Utilities for cleaning and tokenizing Tamil text."""

from __future__ import annotations

import re
from typing import List


# Keep Tamil and English letters, numbers, and whitespace.
_NON_TEXT_PATTERN = re.compile(r"[^0-9A-Za-z\u0B80-\u0BFF\s]")
_MULTI_SPACE_PATTERN = re.compile(r"\s+")


def clean_tamil_text(text: str) -> str:
    """
    Remove punctuation and extra symbols from Tamil text.

    Parameters
    ----------
    text : str
        Raw Tamil sentence.

    Returns
    -------
    str
        Cleaned text.
    """
    if not isinstance(text, str):
        return ""

    cleaned = _NON_TEXT_PATTERN.sub(" ", text)
    cleaned = _MULTI_SPACE_PATTERN.sub(" ", cleaned).strip()
    return cleaned


def tokenize_tamil_text(text: str) -> List[str]:
    """
    Tokenize Tamil text.

    Tamil text is handled with simple whitespace tokenization after cleaning.
    This is more stable than `wordpunct_tokenize` for Tamil Unicode text.
    """
    cleaned_text = clean_tamil_text(text)
    if not cleaned_text:
        return []

    return [token for token in cleaned_text.split() if token.strip()]


if __name__ == "__main__":
    sample = "தமிழ்நாட்டில் கடும் மழை!"
    print("Original :", sample)
    print("Cleaned  :", clean_tamil_text(sample))
    print("Tokens   :", tokenize_tamil_text(sample))
