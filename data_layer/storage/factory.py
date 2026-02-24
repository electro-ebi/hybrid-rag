"""
Storage factory for creating storage instances.
"""

from typing import Dict, Any, Optional

from ..core.config import DEFAULT_CONFIG
from .sparse_store import SparseStore
from .vector_store import VectorStore
from .cache import CacheManager


class StorageFactory:
    """Factory for creating storage instances."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize factory with configuration."""
        self.config = DEFAULT_CONFIG
        if config:
            self.config = self.config.from_dict(config)
    
    def create_sparse_store(self) -> SparseStore:
        """Create sparse store instance."""
        return SparseStore(self.config.to_dict())
    
    def create_vector_store(self) -> VectorStore:
        """Create vector store instance."""
        return VectorStore(self.config.to_dict())
    
    def create_cache_manager(self) -> CacheManager:
        """Create cache manager instance."""
        return CacheManager(self.config.to_dict())
