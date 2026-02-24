"""
Data layer - modular storage, retrieval, and processing.
"""

# Core components
from .core import (
    BaseRetriever, BaseIndexer, BaseProcessor,
    RetrievalResult, IndexingResult, ProcessingResult,
    DataLayerConfig, RetrievalMode, ProcessingLevel
)

# Storage
from .storage import SparseStore, VectorStore, CacheManager

# Legacy compatibility
from .storage.sparse_store import (
    search as sparse_search,
    add_sparse_chunks,
    count_docs
)
from .storage.vector_store import (
    query_similar,
    add_chunks,
    count_vectors
)

# Main factory
from .factory import DataLayerFactory, get_sparse_store, get_vector_store

__all__ = [
    # Core
    'BaseRetriever', 'BaseIndexer', 'BaseProcessor',
    'RetrievalResult', 'IndexingResult', 'ProcessingResult',
    'DataLayerConfig', 'RetrievalMode', 'ProcessingLevel',
    
    # Storage
    'SparseStore', 'VectorStore', 'CacheManager',
    
    # Legacy
    'sparse_search', 'add_sparse_chunks', 'count_docs',
    'query_similar', 'add_chunks', 'count_vectors',
    
    # Factory
    'DataLayerFactory', 'get_sparse_store', 'get_vector_store'
]