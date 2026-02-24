"""
Factory for creating data layer components.
"""

from typing import Dict, Any, Optional

from .core.config import DataLayerConfig, DEFAULT_CONFIG
from .storage import SparseStore, VectorStore


class DataLayerFactory:
    """Factory for creating data layer components."""
    
    def __init__(self, config: Optional[DataLayerConfig] = None):
        """Initialize factory with configuration."""
        self.config = config or DEFAULT_CONFIG
    
    def create_sparse_store(self) -> SparseStore:
        """Create sparse store instance."""
        return SparseStore(self.config.to_dict())
    
    def create_vector_store(self) -> VectorStore:
        """Create vector store instance."""
        return VectorStore(self.config.to_dict())
    
    @classmethod
    def from_config_file(cls, config_path: str) -> 'DataLayerFactory':
        """Create factory from config file."""
        config = DataLayerConfig.from_file(config_path)
        return cls(config)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DataLayerFactory':
        """Create factory from config dictionary."""
        config = DataLayerConfig.from_dict(config_dict)
        return cls(config)


# Global factory instance
_default_factory = DataLayerFactory()

def get_sparse_store() -> SparseStore:
    """Get default sparse store instance."""
    return _default_factory.create_sparse_store()

def get_vector_store() -> VectorStore:
    """Get default vector store instance."""
    return _default_factory.create_vector_store()
