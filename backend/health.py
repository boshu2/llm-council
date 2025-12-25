"""Health check and system status utilities for LLM Council."""

from datetime import datetime
from typing import Dict, Any, List
from .config import OPENROUTER_API_KEY, COUNCIL_MODELS, CHAIRMAN_MODEL


def check_api_health() -> Dict[str, Any]:
    """
    Check the health of the API and its dependencies.

    Returns:
        Dict with health status information
    """
    return {
        "status": "healthy",
        "api_key_configured": is_api_key_configured(),
        "timestamp": datetime.utcnow().isoformat(),
        "council_models_count": len(COUNCIL_MODELS),
        "chairman_model": CHAIRMAN_MODEL
    }


def is_api_key_configured() -> bool:
    """
    Check if the OpenRouter API key is configured.

    Returns:
        True if API key is present and non-empty
    """
    return bool(OPENROUTER_API_KEY and len(OPENROUTER_API_KEY) > 0)


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
        "version": "1.0.0"
    }
