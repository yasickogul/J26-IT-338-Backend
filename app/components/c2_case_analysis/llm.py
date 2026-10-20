"""OpenRouter client (notebook cells 7 / 20). All calls use temperature 0.0 like the notebook."""
import threading

from . import config

_client = None
_lock = threading.Lock()


def get_client():
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                from openai import OpenAI

                if not config.OPENROUTER_API_KEY:
                    raise RuntimeError("OPENROUTER_API_KEY is not set.")
                _client = OpenAI(
                    base_url=config.OPENROUTER_BASE_URL,
                    api_key=config.OPENROUTER_API_KEY,
                    timeout=180,
                )
    return _client


def chat(messages, temperature=0.0):
    """Returns the stripped text content of the first choice."""
    response = get_client().chat.completions.create(
        model=config.LLM_MODEL,
        messages=messages,
        temperature=temperature,
    )
    return (response.choices[0].message.content or "").strip()
