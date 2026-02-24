"""
Embedding generation for dense retrieval.
"""

import requests
from config import EMBED_MODEL, OLLAMA_EMBED_TIMEOUT, OLLAMA_EMBED_URL


def generate_embedding(text: str) -> list:
    """
    Generate embedding vector for given text using Ollama.
    """

    response = requests.post(
        OLLAMA_EMBED_URL,
        json={
            "model": EMBED_MODEL,
            "prompt": text
        },
        timeout=OLLAMA_EMBED_TIMEOUT,
    )

    if response.status_code != 200:
        raise Exception(f"Embedding API error: {response.text}")

    data = response.json()

    return data["embedding"]
