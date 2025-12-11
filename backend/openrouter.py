"""OpenRouter API client for making LLM requests."""

import httpx
import asyncio
from typing import List, Dict, Any, Optional
from .config import OPENROUTER_API_KEY, OPENROUTER_API_URL


async def query_model(
    model: str,
    messages: List[Dict[str, str]],
    timeout: float = 120.0
) -> Optional[Dict[str, Any]]:
    """
    Query a single model via OpenRouter API.

    Args:
        model: OpenRouter model identifier (e.g., "openai/gpt-4o")
        messages: List of message dicts with 'role' and 'content'
        timeout: Request timeout in seconds

    Returns:
        Response dict with 'content' and optional 'reasoning_details', or None if failed
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                OPENROUTER_API_URL,
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
        models: List of OpenRouter model identifiers
        messages: List of message dicts to send to each model

    Returns:
        Dict mapping model identifier to response dict (or None if failed)
    """
    import asyncio

    # Create tasks for all models
    tasks = [query_model(model, messages) for model in models]

    # Wait for all to complete
    responses = await asyncio.gather(*tasks)

    # Map models to their responses
    return {model: response for model, response in zip(models, responses)}


# Feature 5: Retry logic for failed model queries
async def query_model_with_retry(
    model: str,
    messages: List[Dict[str, str]],
    max_retries: int = 3,
    base_delay: float = 1.0,
    timeout: float = 120.0
) -> Optional[Dict[str, Any]]:
    """
    Query a model with automatic retry on failure.

    Args:
        model: OpenRouter model identifier
        messages: List of message dicts
        max_retries: Maximum number of retry attempts
        base_delay: Base delay between retries (exponential backoff)
        timeout: Request timeout in seconds

    Returns:
        Response dict or None if all retries failed
    """
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
    max_retries: int = 3
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Query multiple models in parallel with retry logic.

    Args:
        models: List of OpenRouter model identifiers
        messages: List of message dicts
        max_retries: Maximum retries per model

    Returns:
        Dict mapping model to response
    """
    tasks = [query_model_with_retry(model, messages, max_retries) for model in models]
    responses = await asyncio.gather(*tasks)
    return {model: response for model, response in zip(models, responses)}
