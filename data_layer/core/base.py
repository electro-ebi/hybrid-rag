"""
Base classes for data layer components.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from .types import RetrievalResult, IndexingResult, ProcessingResult, Document, RetrievalMode


class BaseRetriever(ABC):
    """Base class for retrieval systems."""
    
    @abstractmethod
    def retrieve(
        self, 
        query: str, 
        mode: RetrievalMode,
        top_k: int = 5,
        **kwargs
    ) -> RetrievalResult:
        """Retrieve chunks for a query."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if retriever is available."""
        pass


class BaseIndexer(ABC):
    """Base class for indexing systems."""
    
    @abstractmethod
    def index_document(self, document: Document) -> IndexingResult:
        """Index a document."""
        pass
    
    @abstractmethod
    def index_file(self, file_path: str, **kwargs) -> IndexingResult:
        """Index a file."""
        pass
    
    @abstractmethod
    def delete_document(self, document_id: str) -> bool:
        """Delete a document from index."""
        pass


class BaseProcessor(ABC):
    """Base class for document processors."""
    
    @abstractmethod
    def process(self, content: Any, **kwargs) -> ProcessingResult:
        """Process content."""
        pass
    
    @abstractmethod
    def supports_file_type(self, file_type: str) -> bool:
        """Check if processor supports file type."""
        pass


class BaseStorage(ABC):
    """Base class for storage systems."""
    
    @abstractmethod
    def store(self, key: str, value: Any) -> bool:
        """Store a value."""
        pass
    
    @abstractmethod
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a value."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a value."""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass
    
    @abstractmethod
    def list_keys(self, pattern: str = "*") -> List[str]:
        """List keys matching pattern."""
        pass
