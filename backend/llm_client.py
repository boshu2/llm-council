"""LLM client for making requests via LiteLLM proxy."""

import httpx
import asyncio
from typing import List, Dict, Any, Optional
from .config import (
    LITELLM_API_URL,
    LITELLM_API_KEY,
    DEFAULT_TIMEOUT,
    MAX_RETRIES,
    RETRY_BASE_DELAY,
)


async def query_model(
    model: str,
    messages: List[Dict[str, str]],
    timeout: float = None
) -> Optional[Dict[str, Any]]:
    """
    Query a single model via LiteLLM proxy.

    Args:
        model: Model identifier (e.g., "ollama/llama3.1")
        messages: List of message dicts with 'role' and 'content'
        timeout: Request timeout in seconds

    Returns:
        Response dict with 'content' and optional 'reasoning_details', or None if failed
    """
    if timeout is None:
        timeout = DEFAULT_TIMEOUT

    headers = {
        "Content-Type": "application/json",
    }

    # Add API key if configured
    if LITELLM_API_KEY:
        headers["Authorization"] = f"Bearer {LITELLM_API_KEY}"

    payload = {
        "model": model,
        "messages": messages,
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                LITELLM_API_URL,
                headers=headers,
                json=payload
            )
            response.raise_for_status()

            data = response.json()
            message = data['choices'][0]['message']

            return {
                'content': message.get('content'),
                'reasoning_details': message.get('reasoning_details')
            }

    except Exception as e:
        print(f"Error querying model {model}: {e}")
        return None


async def query_models_parallel(
    models: List[str],
    messages: List[Dict[str, str]]
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Query multiple models in parallel.

    Args:
        models: List of model identifiers
        messages: List of message dicts to send to each model

    Returns:
        Dict mapping model identifier to response dict (or None if failed)
    """
    tasks = [query_model(model, messages) for model in models]
    responses = await asyncio.gather(*tasks)
    return {model: response for model, response in zip(models, responses)}


async def query_model_with_retry(
    model: str,
    messages: List[Dict[str, str]],
    max_retries: int = None,
    base_delay: float = None,
    timeout: float = None
) -> Optional[Dict[str, Any]]:
    """
    Query a model with automatic retry on failure.

    Args:
        model: Model identifier
        messages: List of message dicts
        max_retries: Maximum number of retry attempts
        base_delay: Base delay between retries (exponential backoff)
        timeout: Request timeout in seconds

    Returns:
        Response dict or None if all retries failed
    """
    if max_retries is None:
        max_retries = MAX_RETRIES
    if base_delay is None:
        base_delay = RETRY_BASE_DELAY
    if timeout is None:
        timeout = DEFAULT_TIMEOUT

    last_error = None

    for attempt in range(max_retries):
        try:
            result = await query_model(model, messages, timeout)
            if result is not None:
                return result
        except Exception as e:
            last_error = e

        # Exponential backoff
        if attempt < max_retries - 1:
            delay = base_delay * (2 ** attempt)
            await asyncio.sleep(delay)

    if last_error:
        print(f"All {max_retries} retries failed for model {model}: {last_error}")

    return None


async def query_models_parallel_with_retry(
    models: List[str],
    messages: List[Dict[str, str]],
    max_retries: int = None
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Query multiple models in parallel with retry logic.

    Args:
        models: List of model identifiers
        messages: List of message dicts
        max_retries: Maximum retries per model

    Returns:
        Dict mapping model to response
    """
    if max_retries is None:
        max_retries = MAX_RETRIES

    tasks = [query_model_with_retry(model, messages, max_retries) for model in models]
    responses = await asyncio.gather(*tasks)
    return {model: response for model, response in zip(models, responses)}
