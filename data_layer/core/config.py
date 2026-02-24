"""
Configuration for data layer components.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path


@dataclass
class DataLayerConfig:
    """Configuration for the data layer."""
    
    # Storage paths
    sparse_index_dir: str = "whoosh_index"
    vector_index_dir: str = "chroma_db"
    cache_dir: str = "cache"
    
    # Retrieval configuration
    default_top_k: int = 5
    hybrid_top_k_multiplier: int = 2
    rrf_k: int = 60
    
    # Embedding configuration
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # Sparse search configuration
    sparse_analyzer: str = "standard"
    sparse_stoplist: Optional[str] = None
    
    # Processing configuration
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_file_size_mb: int = 50
    
    # Reranking configuration
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_device: str = "cpu"
    
    # OCR configuration
    ocr_engine: str = "easyocr"
    ocr_languages: list = None
    
    # Vision configuration
    vision_model: str = "llava"
    
    # Cache configuration
    enable_cache: bool = True
    cache_ttl: int = 3600  # seconds
    
    # Logging configuration
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Initialize defaults."""
        if self.ocr_languages is None:
            self.ocr_languages = ['en']
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DataLayerConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_file(cls, config_path: Path) -> 'DataLayerConfig':
        """Load config from file."""
        import json
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            field.name: getattr(self, field.name) 
            for field in self.__dataclass_fields__.values()
        }
    
    def save(self, config_path: Path):
        """Save config to file."""
        import json
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# Default configuration instance
DEFAULT_CONFIG = DataLayerConfig()
