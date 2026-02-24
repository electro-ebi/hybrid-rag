import requests
from dotenv import load_dotenv

from config import OLLAMA_TAGS_URL, OLLAMA_TIMEOUT, OLLAMA_URL
from logger import get_logger
from model_router import route_model

load_dotenv()

logger = get_logger("llm")

SYSTEM_PROMPT = """
You are an intelligent autonomous AI assistant.
Be clear, structured, and concise.
If unsure, say you are unsure.
"""


def call_llm(
    user_prompt: str,
    temperature: float = 0.7,
    system_prompt: str = SYSTEM_PROMPT,
    task_type: str = "generation",
) -> str:
    """
    Sends a prompt to the local Ollama LLM and returns the generated response.

    task_type selects the model via model_router: "generation", "rerank", "embedding", "vision".
    Default is "generation".
    """
    model = route_model(task_type)
    full_prompt = f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:"

    try:
        logger.info("Calling LLM model=%s task_type=%s", model, task_type)
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                },
            },
            timeout=OLLAMA_TIMEOUT,
        )

        response.raise_for_status()

        data = response.json()
        logger.info("LLM response received")

        return data.get("response", "").strip()

    except requests.exceptions.Timeout:
        logger.error("LLM request timed out model=%s", model)
        return "Error: LLM request timed out."

    except requests.exceptions.ConnectionError:
        logger.error("LLM connection error to %s", OLLAMA_URL)
        return "Error: Cannot connect to Ollama server."

    except requests.exceptions.HTTPError as e:
        logger.error("LLM HTTP error: %s", e)
        return f"HTTP Error: {str(e)}"

    except Exception as e:
        logger.exception("LLM unexpected error")
        return f"Unexpected error: {str(e)}"


def check_ollama_connectivity() -> bool:
    try:
        response = requests.get(OLLAMA_TAGS_URL, timeout=OLLAMA_TIMEOUT)
        response.raise_for_status()
        return True
    except Exception:
        return False
