"""
Validation utilities for data layer components.
"""

from typing import Dict, Any, List


def validate_metadata(metadata: Dict[str, Any]) -> bool:
    """Validate metadata structure."""
    if not isinstance(metadata, dict):
        return False
    
    # Check required fields
    required_fields = ['source']
    for field in required_fields:
        if field not in metadata:
            return False
    
    return True


def validate_chunk(chunk: Dict[str, Any]) -> bool:
    """Validate chunk structure."""
    if not isinstance(chunk, dict):
        return False
    
    # Check required fields
    required_fields = ['chunk_uid', 'content']
    for field in required_fields:
        if field not in chunk:
            return False
    
    # Validate content
    if not isinstance(chunk['content'], str) or not chunk['content'].strip():
        return False
    
    return True


def validate_query(query: str) -> bool:
    """Validate search query."""
    if not isinstance(query, str):
        return False
    
    query = query.strip()
    if not query or len(query) < 2:
        return False
    
    return True


def validate_file_path(file_path: str) -> bool:
    """Validate file path."""
    if not isinstance(file_path, str):
        return False
    
    file_path = file_path.strip()
    if not file_path:
        return False
    
    # Check for invalid characters
    invalid_chars = '<>:"|?*'
    for char in invalid_chars:
        if char in file_path:
            return False
    
    return True


def validate_config(config: Dict[str, Any]) -> List[str]:
    """Validate configuration and return list of errors."""
    errors = []
    
    if not isinstance(config, dict):
        errors.append("Configuration must be a dictionary")
        return errors
    
    # Validate specific config fields
    if 'chunk_size' in config:
        if not isinstance(config['chunk_size'], int) or config['chunk_size'] <= 0:
            errors.append("chunk_size must be a positive integer")
    
    if 'chunk_overlap' in config:
        if not isinstance(config['chunk_overlap'], int) or config['chunk_overlap'] < 0:
            errors.append("chunk_overlap must be a non-negative integer")
    
    if 'top_k' in config:
        if not isinstance(config['top_k'], int) or config['top_k'] <= 0:
            errors.append("top_k must be a positive integer")
    
    return errors
