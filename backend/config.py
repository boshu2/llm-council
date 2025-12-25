"""Configuration for the LLM Council with LiteLLM backend."""

import os
from dotenv import load_dotenv

load_dotenv()

# LiteLLM proxy configuration
LITELLM_API_URL = os.getenv("LITELLM_API_URL", "http://localhost:4000/v1/chat/completions")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", "")  # Optional, depends on your LiteLLM setup

# Council members - self-hosted models via LiteLLM
# Update these to match your deployed models
COUNCIL_MODELS = [
    "ollama/llama3.1",
    "ollama/mistral",
    "ollama/codellama",
]

# Chairman model - synthesizes final response
# Can be any model from your LiteLLM config
CHAIRMAN_MODEL = "ollama/llama3.1"

# Data directory for conversation storage
DATA_DIR = "data/conversations"

# Model query timeout (seconds)
DEFAULT_TIMEOUT = 120.0

# Retry configuration
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0
