"""Health check and system status utilities for LLM Council."""

from datetime import datetime
from typing import Dict, Any
from .config import LITELLM_API_URL, LITELLM_API_KEY, COUNCIL_MODELS, CHAIRMAN_MODEL


def check_api_health() -> Dict[str, Any]:
    """
    Check the health of the API and its dependencies.

    Returns:
        Dict with health status information
    """
    return {
        "status": "healthy",
        "litellm_configured": is_litellm_configured(),
        "timestamp": datetime.utcnow().isoformat(),
        "council_models_count": len(COUNCIL_MODELS),
        "chairman_model": CHAIRMAN_MODEL
    }


def is_litellm_configured() -> bool:
    """
    Check if LiteLLM is configured.

    Returns:
        True if LiteLLM URL is configured
    """
    return bool(LITELLM_API_URL and len(LITELLM_API_URL) > 0)


def get_available_models() -> Dict[str, Any]:
    """
    Get information about available models.

    Returns:
        Dict with model information
    """
    return {
        "council_models": COUNCIL_MODELS,
        "chairman_model": CHAIRMAN_MODEL,
        "total_council_members": len(COUNCIL_MODELS)
    }


def get_system_status() -> Dict[str, Any]:
    """
    Get comprehensive system status.

    Returns:
        Dict with system status information
    """
    return {
        "api_health": check_api_health(),
        "models": get_available_models(),
        "litellm_url": LITELLM_API_URL,
        "version": "1.0.0"
    }
