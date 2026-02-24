"""
Vision pipeline: analyze images with the vision model (e.g. llava:7b).
"""
from __future__ import annotations

import base64
import os

import requests

from config import OLLAMA_CHAT_URL, OLLAMA_TIMEOUT
from logger import get_logger
from model_router import route_model

logger = get_logger("vision")


def analyze_image(
    question: str,
    *,
    image_path: str | None = None,
    image_base64: str | None = None,
) -> str:
    """
    Run vision model on the image and answer the question.
    Provide either image_path (file path) or image_base64 (base64-encoded image string).
    """
    if image_base64:
        b64 = image_base64.strip()
    elif image_path and os.path.isfile(image_path):
        try:
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
        except OSError as e:
            logger.error("Failed to read image %s: %s", image_path, e)
            return f"Error: Could not read image: {e}"
    else:
        return f"Error: Image file not found or no base64 provided: {image_path}"

    prompt = "You are analyzing an image. Answer clearly and technically.\n\n" + question
    model = route_model("vision")

    try:
        logger.info("Vision request model=%s", model)
        response = requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [b64],
                    }
                ],
                "stream": False,
            },
            timeout=OLLAMA_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        msg = data.get("message") or {}
        text = (msg.get("content") or "").strip()
        logger.info("Vision response received")
        return text or "No response from vision model."
    except requests.exceptions.Timeout:
        logger.error("Vision request timed out")
        return "Error: Vision request timed out."
    except requests.exceptions.ConnectionError:
        logger.error("Vision connection error to %s", OLLAMA_CHAT_URL)
        return "Error: Cannot connect to Ollama server."
    except requests.exceptions.HTTPError as e:
        logger.error("Vision HTTP error: %s", e)
        return f"HTTP Error: {str(e)}"
    except Exception as e:
        logger.exception("Vision unexpected error")
        return f"Unexpected error: {str(e)}"
