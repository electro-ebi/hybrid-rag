import os

DATA_DIR = os.getenv("DATA_DIR", "../data")
WHOOSH_INDEX_DIR = os.getenv("WHOOSH_INDEX_DIR", "whoosh_index")
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")

# Model routing (see model_router.py for defaults)
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "qwen2.5:14b")
RERANK_MODEL = os.getenv("RERANK_MODEL", "qwen2.5:7b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
VISION_MODEL = os.getenv("VISION_MODEL", "llava:7b")

# Backward compatibility
LLM_MODEL = os.getenv("MODEL_NAME", GENERATION_MODEL)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_CHAT_URL = os.getenv("OLLAMA_CHAT_URL", "http://localhost:11434/api/chat")
OLLAMA_EMBED_URL = os.getenv("OLLAMA_EMBED_URL", "http://localhost:11434/api/embeddings")
OLLAMA_TAGS_URL = os.getenv("OLLAMA_TAGS_URL", "http://localhost:11434/api/tags")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "60"))
OLLAMA_EMBED_TIMEOUT = int(os.getenv("OLLAMA_EMBED_TIMEOUT", "60"))
