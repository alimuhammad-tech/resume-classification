"""
Text Preprocessing and Cleaning Module.
Cleans raw text for machine learning pipelines and extracts contact metadata.
"""

from typing import Dict, List, Any
import re

EMAIL_REGEX = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
PHONE_REGEX = r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
URL_REGEX = r"https?://\S+|www\.\S+"
GITHUB_REGEX = r"(?:https?://)?(?:www\.)?github\.com/[A-Za-z0-9_-]+"
LINKEDIN_REGEX = r"(?:https?://)?(?:www\.)?linkedin\.com/in/[A-Za-z0-9_-]+"


def extract_contact_info(text: str) -> Dict[str, Any]:
    """
    Extract contact metadata such as email, phone numbers, and profile links.
    """
    emails = list(set(re.findall(EMAIL_REGEX, text)))
    phones = list(set(re.findall(PHONE_REGEX, text)))
    github_matches = list(set(re.findall(GITHUB_REGEX, text, re.IGNORECASE)))
    linkedin_matches = list(set(re.findall(LINKEDIN_REGEX, text, re.IGNORECASE)))

    return {
        "emails": emails,
        "phones": phones,
        "github": github_matches[0] if github_matches else None,
        "linkedin": linkedin_matches[0] if linkedin_matches else None
    }


def clean_text(text: str) -> str:
    """
    Clean text for TF-IDF feature extraction.
    Removes URLs, emails, phone numbers, special characters, and normalizes spaces.
    """
    if not text:
        return ""

    # Convert to string if not already
    text = str(text)

    # 1. Remove URLs
    text = re.sub(URL_REGEX, " ", text)

    # 2. Remove Emails
    text = re.sub(EMAIL_REGEX, " ", text)

    # 3. Remove Phone numbers
    text = re.sub(PHONE_REGEX, " ", text)

    # 4. Remove special characters and numbers (preserve letters and basic spacing)
    # Retain technical terms with plus/hash like c++, c# by temporarily tokenizing or preserving
    text = re.sub(r"c\+\+", "cplusplus", text, flags=re.IGNORECASE)
    text = re.sub(r"c\#", "csharp", text, flags=re.IGNORECASE)
    text = re.sub(r"\.net", "dotnet", text, flags=re.IGNORECASE)

    # Replace punctuation and digits with spaces
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # 5. Lowercase
    text = text.lower()

    # 6. Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def compute_text_stats(raw_text: str) -> Dict[str, int]:
    """
    Calculate text statistics for summary display.
    """
    words = raw_text.split()
    lines = raw_text.splitlines()
    characters = len(raw_text)
    
    return {
        "word_count": len(words),
        "char_count": characters,
        "line_count": len([line for line in lines if line.strip()]),
        "estimated_read_time_sec": max(1, round(len(words) / 3.5))  # ~200-250 wpm
    }

