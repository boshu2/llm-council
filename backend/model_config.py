"""Model configuration utilities for LLM Council."""

from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """
    Configuration for model queries.
    """
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty
        }


# Default configurations for different use cases
DEFAULT_CONFIGS = {
    "council": ModelConfig(temperature=0.7, max_tokens=4096),
    "ranking": ModelConfig(temperature=0.3, max_tokens=4096),
    "chairman": ModelConfig(temperature=0.5, max_tokens=8192),
    "title": ModelConfig(temperature=0.3, max_tokens=50)
}


def get_model_config(config_type: str) -> ModelConfig:
    """
    Get a model configuration by type.

    Args:
        config_type: Type of configuration

    Returns:
        ModelConfig instance
    """
    return DEFAULT_CONFIGS.get(config_type, ModelConfig())


def create_custom_config(
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    presence_penalty: Optional[float] = None
) -> ModelConfig:
    """
    Create a custom model configuration.

    Args:
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        top_p: Nucleus sampling parameter
        frequency_penalty: Frequency penalty
        presence_penalty: Presence penalty

    Returns:
        ModelConfig instance
    """
    config = ModelConfig()

    if temperature is not None:
        config.temperature = temperature
    if max_tokens is not None:
        config.max_tokens = max_tokens
    if top_p is not None:
        config.top_p = top_p
    if frequency_penalty is not None:
        config.frequency_penalty = frequency_penalty
    if presence_penalty is not None:
        config.presence_penalty = presence_penalty

    return config
