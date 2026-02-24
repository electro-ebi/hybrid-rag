"""
File processing utilities.
"""

import os
import hashlib
from pathlib import Path
from typing import Optional


def get_file_hash(file_path: str, algorithm: str = 'md5') -> str:
    """Calculate file hash."""
    hash_func = hashlib.new(algorithm)
    
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
    except Exception as e:
        raise ValueError(f"Failed to calculate hash for {file_path}: {e}")


def get_file_size(file_path: str) -> int:
    """Get file size in bytes."""
    try:
        return os.path.getsize(file_path)
    except Exception as e:
        raise ValueError(f"Failed to get size for {file_path}: {e}")


def get_file_extension(file_path: str) -> str:
    """Get file extension."""
    return Path(file_path).suffix.lower()


def is_supported_file_type(file_path: str, supported_types: list = None) -> bool:
    """Check if file type is supported."""
    if supported_types is None:
        supported_types = ['.pdf', '.txt', '.md', '.docx', '.doc']
    
    extension = get_file_extension(file_path)
    return extension in supported_types


def ensure_directory(directory: str) -> bool:
    """Ensure directory exists."""
    try:
        os.makedirs(directory, exist_ok=True)
        return True
    except Exception:
        return False


def safe_filename(filename: str) -> str:
    """Create safe filename by removing invalid characters."""
    # Remove invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip(' .')
    
    # Ensure filename is not empty
    if not filename:
        filename = 'unnamed_file'
    
    return filename


def get_relative_path(file_path: str, base_path: str) -> str:
    """Get relative path from base path."""
    try:
        return os.path.relpath(file_path, base_path)
    except ValueError:
        # If on different drives, return absolute path
        return os.path.abspath(file_path)
