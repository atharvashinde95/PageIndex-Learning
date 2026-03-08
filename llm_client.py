import os
import time
import logging
from openai import OpenAI, RateLimitError
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Minimum seconds between every LLM call — prevents burst 429s during summary generation
THROTTLE_DELAY = 2.0


def get_llm_client() -> OpenAI:
    api_key = os.getenv("CAPGEMINI_API_KEY")
    base_url = os.getenv("CAPGEMINI_BASE_URL")
    if not api_key or not base_url:
        raise EnvironmentError(
            "CAPGEMINI_API_KEY and CAPGEMINI_BASE_URL must be set in your .env file."
        )
    # max_retries=0: disable the SDK's own retry so WE control it with proper delays
    return OpenAI(api_key=api_key, base_url=base_url, max_retries=0)


def get_model_id() -> str:
    model_id = os.getenv("MODEL_ID")
    if not model_id:
        raise EnvironmentError("MODEL_ID must be set in your .env file.")
    return model_id


def call_llm(
    prompt: str,
    system: str = "",
    max_tokens: int = 4096,
    max_retries: int = 8,
    base_delay: float = 15.0,
) -> str:
    """
    Single-turn LLM call with:
      1. A THROTTLE_DELAY pause before every call to prevent burst 429s
      2. Exponential backoff when a 429 occurs despite throttling

    The OpenAI SDK's built-in retry is disabled (max_retries=0 in client)
    because its default retry intervals are too short for Capgemini's limits.

    Backoff schedule (base_delay=15):
      429 hit -> wait 15s -> retry
      429 hit -> wait 30s -> retry
      429 hit -> wait 60s -> retry
      429 hit -> wait 120s -> retry (capped)
    """
    client = get_llm_client()
    model = get_model_id()

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    logger.debug("LLM call | model=%s | prompt_chars=%d", model, len(prompt))

    # Throttle: always wait before firing to avoid bursts
    time.sleep(THROTTLE_DELAY)

    delay = base_delay

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()

        except RateLimitError:
            if attempt == max_retries:
                logger.error(
                    "call_llm: rate limit hit on final attempt %d — giving up.", attempt
                )
                raise

            logger.warning(
                "call_llm: 429 Rate Limit (attempt %d/%d) — waiting %ds then retrying...",
                attempt, max_retries, int(delay),
            )
            time.sleep(delay)
            delay = min(delay * 2, 120)  # cap at 2 minutes

        except Exception as e:
            logger.error("call_llm: unexpected error: %s", e)
            raise
