"""
core/llm_client.py
------------------
Thin wrapper around the Capgemini OpenAI-compatible endpoint.
All LLM calls in the project go through this module.
"""

import os
import logging
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def get_llm_client() -> OpenAI:
    """Return a configured OpenAI client pointing at the Capgemini endpoint."""
    api_key = os.getenv("CAPGEMINI_API_KEY")
    base_url = os.getenv("CAPGEMINI_BASE_URL")

    if not api_key or not base_url:
        raise EnvironmentError(
            "CAPGEMINI_API_KEY and CAPGEMINI_BASE_URL must be set in your .env file."
        )

    return OpenAI(api_key=api_key, base_url=base_url)


def get_model_id() -> str:
    """Return the configured model ID."""
    model_id = os.getenv("MODEL_ID")
    if not model_id:
        raise EnvironmentError("MODEL_ID must be set in your .env file.")
    return model_id


def call_llm(prompt: str, system: str = "", max_tokens: int = 4096) -> str:
    """
    Single-turn LLM call.

    Args:
        prompt:     User-turn message.
        system:     Optional system message.
        max_tokens: Maximum tokens in the response.

    Returns:
        Response text as a plain string.
    """
    client = get_llm_client()
    model = get_model_id()

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    logger.debug("LLM call | model=%s | prompt_chars=%d", model, len(prompt))

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0,         # deterministic — critical for structured extraction
    )

    return response.choices[0].message.content.strip()
