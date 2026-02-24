"""
Query processing for different retrieval modes.
"""

import re
from typing import Dict, Any, List

from ..core.config import DEFAULT_CONFIG


class QueryProcessor:
    """Processes queries for different retrieval modes."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize query processor."""
        self.config = DEFAULT_CONFIG
        if config:
            self.config = self.config.from_dict(config)
    
    def extract_key_terms(self, query: str) -> str:
        """Extract key terms from complex queries for better sparse retrieval."""
        stopwords = {
            'what', 'how', 'why', 'when', 'where', 'who', 'which', 'can', 'could', 'would', 
            'should', 'do', 'does', 'did', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'having', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on',
            'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through',
            'during', 'before', 'after', 'above', 'below', 'between', 'under', 'along', 'following',
            'across', 'behind', 'beyond', 'plus', 'except', 'yet', 'nor', 'not', 'no',
            'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'now'
        }
        
        words = re.findall(r'\b[a-zA-Z]+\b', query.lower())
        key_terms = [word for word in words if word not in stopwords and len(word) > 2]
        
        if not key_terms:
            return query
        
        return ' OR '.join(key_terms)
    
    def preprocess_for_sparse(self, query: str) -> str:
        """Preprocess query for sparse retrieval."""
        processed = self.extract_key_terms(query)
        return processed.lower()
    
    def preprocess_for_dense(self, query: str) -> str:
        """Preprocess query for dense retrieval."""
        # Dense retrieval typically works well with natural language
        # Just basic cleanup
        return query.strip()
    
    def analyze_query_type(self, query: str) -> str:
        """Analyze query type for optimal routing."""
        query_lower = query.lower()
        
        # Check for technical terms (favor sparse)
        technical_patterns = [
            r'\b\w+\.\w+\b',  # File names, model names
            r'\b\d+\b',        # Numbers
            r'\b[A-Z]{2,}\b',  # Acronyms
        ]
        
        for pattern in technical_patterns:
            if re.search(pattern, query):
                return 'technical'
        
        # Check for question words (favor dense)
        question_words = ['what', 'how', 'why', 'when', 'where', 'who']
        if any(word in query_lower for word in question_words):
            return 'question'
        
        # Check for short queries (favor sparse)
        if len(query.split()) <= 3:
            return 'short'
        
        return 'general'
    
    def should_use_hybrid(self, query: str) -> bool:
        """Determine if query should use hybrid retrieval."""
        query_type = self.analyze_query_type(query)
        
        # Use hybrid for complex queries
        if query_type in ['question', 'general']:
            return True
        
        # Use single mode for simple queries
        return False
