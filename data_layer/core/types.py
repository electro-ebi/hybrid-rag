"""
Core type definitions for the data layer.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
from enum import Enum


class RetrievalMode(Enum):
    """Retrieval modes."""
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"


class ProcessingLevel(Enum):
    """Processing levels for indexing."""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"


@dataclass
class RetrievalResult:
    """Result from retrieval operation."""
    chunks: List[str]
    chunk_ids: List[str]
    mode: RetrievalMode
    query: str
    scores: List[float]
    metadata: Dict[str, Any]
    latency_ms: float


@dataclass
class IndexingResult:
    """Result from indexing operation."""
    status: str
    chunks_indexed: int
    file_path: str
    processing_level: ProcessingLevel
    metadata: Dict[str, Any]
    errors: Dict[str, Any]
    sparse_indexed: bool
    dense_indexed: bool


@dataclass
class ProcessingResult:
    """Result from processing operation."""
    success: bool
    content: Dict[str, Any]
    metadata: Dict[str, Any]
    processing_time: float
    errors: List[str]


@dataclass
class Chunk:
    """A document chunk."""
    id: str
    content: str
    metadata: Dict[str, Any]
    source: str
    chunk_type: str = "text"


@dataclass
class Document:
    """A document with its chunks."""
    id: str
    source: str
    content: Dict[str, Any]
    metadata: Dict[str, Any]
    chunks: List[Chunk]
    file_type: str
    processing_level: ProcessingLevel
