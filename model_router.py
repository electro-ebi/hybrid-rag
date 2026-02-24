"""
Config-driven model routing for generation, rerank, embedding, and vision.
"""
from __future__ import annotations

import os

_DEFAULTS = {
    "generation": "qwen2.5:14b",
    "rerank": "BAAI/bge-reranker-base",  # Cross-encoder reranker
    "embedding": "nomic-embed-text",
    "vision": "llava:7b",
    "coding": "deepseek-coder:6.7b",  # Specialized coding model
}

_ENV_KEYS = {
    "generation": "GENERATION_MODEL",
    "rerank": "RERANK_MODEL",
    "embedding": "EMBED_MODEL",
    "vision": "VISION_MODEL",
    "coding": "CODING_MODEL",
}


def route_model(task_type: str) -> str:
    """
    Return the model name for the given task type.

    Supported task_type: "generation", "rerank", "embedding", "vision", "coding".
    Uses env vars GENERATION_MODEL, RERANK_MODEL, EMBED_MODEL, VISION_MODEL, CODING_MODEL
    with sensible defaults.
    """
    if task_type not in _DEFAULTS:
        raise ValueError(f"Unknown task_type: {task_type}. Use one of {list(_DEFAULTS)}")
    key = _ENV_KEYS[task_type]
    return os.getenv(key, _DEFAULTS[task_type])
