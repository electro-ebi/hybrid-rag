from fastapi import FastAPI
from fastapi import HTTPException

from pydantic import BaseModel
import os
from typing import Literal

import requests

from config import DATA_DIR, OLLAMA_URL, WHOOSH_INDEX_DIR
from data_layer.sparse_store import count_docs
from data_layer.vector_store import count_vectors
from logger import get_logger

# Import the function to call the LLM from the llm module
from llm import call_llm

# Import the function to call the RAG from the rag module
from rag import retrieve
from data_layer.indexing import index_file
from data_layer.retrieval import retrieve_chunks
from data_layer.validator import validate_answer
from vision_handler import analyze_image

app = FastAPI(title="Autonomous Agent API")
logger = get_logger(__name__)


class Query(BaseModel):
    question: str


class IndexRequest(BaseModel):
    path: str | None = None


class RetrieveConfig(BaseModel):
    top_k: int = 5
    mode: Literal["dense", "sparse", "hybrid"] = "hybrid"


class QueryRequest(BaseModel):
    question: str
    retrieval: RetrieveConfig = RetrieveConfig()
    debug: bool = False


class VisionRequest(BaseModel):
    question: str
    image_path: str | None = None
    image_base64: str | None = None


@app.get("/health")
def health() -> dict[str, int | str | bool]:
    try:
        requests.get(OLLAMA_URL, timeout=2)
        ollama_status = "reachable"
    except Exception:
        ollama_status = "unreachable"

    return {
        "status": "ok",
        "dense_vectors": count_vectors(),
        "sparse_documents": count_docs(WHOOSH_INDEX_DIR),
        "ollama_status": ollama_status,
        "whoosh_index_exists": os.path.isdir(WHOOSH_INDEX_DIR),
    }



@app.post("/index")
def index_endpoint(req: IndexRequest):
    """
    Index a single file or (by default) all files in ../data.
    """
    target = req.path
    if not target:
        target = os.path.abspath(os.path.join(os.path.dirname(__file__), DATA_DIR))

    if os.path.isdir(target):
        results = []
        for name in sorted(os.listdir(target)):
            fp = os.path.join(target, name)
            if os.path.isfile(fp):
                try:
                    results.append(index_file(fp))
                except Exception as e:
                    results.append({"file": fp, "status": "error", "error": str(e)})
        return {"indexed": results}

    try:
        return index_file(target)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/query")
def query_llm(req: QueryRequest):
    # Step 1: Retrieve relevant context
    if req.debug:
        context_chunks, retrieval_debug = retrieve_chunks(
            req.question,
            top_k=req.retrieval.top_k,
            mode=req.retrieval.mode,
            data_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), DATA_DIR)),
            include_debug=True,
        )
    else:
        context_chunks = retrieve(
            req.question,
            top_k=req.retrieval.top_k,
            mode=req.retrieval.mode,
        )
        retrieval_debug = None

    if not context_chunks:
        logger.warning("No relevant documents found for query=%s", req.question)
        raise HTTPException(status_code=404, detail="No relevant documents found.")

    context_text = "\n\n".join(context_chunks)

    # Step 2: Create RAG prompt
    rag_prompt = f"""
You are an AI assistant that answers strictly based on the provided context chunks.

Context:
{context_text}

Question:
{req.question}

Rules:
1) Do not use outside knowledge.
2) Provide a structured answer with concise bullet points.
3) For every factual claim, include citation(s) using [Chunk N] notation from the context labels.
4) If information is missing, explicitly state: "The answer is not found in the provided document."
"""

    # Step 3: Call LLM with augmented prompt
    answer = call_llm(rag_prompt)
    validation = validate_answer(req.question, context_text, answer)

    response = {
        "answer": answer,
        "retrieved_chunks": context_chunks,
        "validation": validation,
    }
    if not validation.get("supported", False):
        response["warning"] = "Answer may not be fully supported by the retrieved context."
    if req.debug and retrieval_debug is not None:
        response["debug"] = {
            "mode": req.retrieval.mode,
            "top_k": req.retrieval.top_k,
            "dense_vector_count": count_vectors(),
            "sparse_doc_count": count_docs(WHOOSH_INDEX_DIR),
            "retrieved_count": len(context_chunks),
            "dense_results_count": retrieval_debug["dense_results_count"],
            "sparse_results_count": retrieval_debug["sparse_results_count"],
            "fused_ranking_order": retrieval_debug["fused_ranking_order"],
        }
    return response


@app.post("/vision")
def vision_endpoint(req: VisionRequest):
    """
    Analyze an image with the vision model (e.g. llava:7b).
    Provide either image_path (server path) or image_base64.
    """
    if not req.image_path and not req.image_base64:
        raise HTTPException(status_code=400, detail="Provide image_path or image_base64")
    result = analyze_image(
        req.question,
        image_path=req.image_path,
        image_base64=req.image_base64,
    )
    return {"answer": result}
