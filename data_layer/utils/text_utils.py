"""
Text processing utilities.
"""

import re
from typing import List


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/\\]', '', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """Truncate text to maximum length."""
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    if not text:
        return []
    
    # Simple sentence splitting - can be enhanced
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


def extract_keywords(text: str, min_length: int = 3) -> List[str]:
    """Extract keywords from text."""
    if not text:
        return []
    
    # Simple keyword extraction - can be enhanced with NLP
    words = re.findall(r'\b[a-zA-Z]{' + str(min_length) + ',}\b', text.lower())
    
    # Remove common stopwords
    stopwords = {
        'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day',
        'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did',
        'she', 'use', 'her', 'than', 'when', 'make', 'made', 'what', 'with', 'have', 'this', 'will', 'your', 'from',
        'they', 'know', 'want', 'been', 'good', 'much', 'some', 'time', 'very', 'when', 'come', 'here', 'just', 'like',
        'long', 'many', 'them', 'well', 'were', 'will'
    }
    
    keywords = [word for word in words if word not in stopwords]
    
    # Return unique keywords
    return list(set(keywords))
