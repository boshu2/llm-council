# LLM Council

![llmcouncil](header.jpg)

A 3-stage deliberation system where multiple self-hosted LLMs collaboratively answer questions. Uses LiteLLM as the routing layer to your self-hosted models, with Open WebUI as the frontend.

## How It Works

1. **Stage 1: First opinions**. Your query is sent to all council LLMs individually. Responses are collected in parallel.
2. **Stage 2: Review**. Each LLM reviews and ranks the others' responses (anonymized to prevent bias).
3. **Stage 3: Final response**. The Chairman LLM synthesizes all responses and rankings into a final answer.

## Architecture

```
Open WebUI → LLM Council API → LiteLLM Proxy → Self-hosted Models (Ollama, vLLM, etc.)
```

## Setup

### 1. Install Dependencies

The project uses [uv](https://docs.astral.sh/uv/) for project management.

**Backend:**
```bash
uv sync
```

**Frontend (optional - use Open WebUI instead):**
```bash
cd frontend
npm install
cd ..
```

### 2. Configure LiteLLM

Create a `.env` file in the project root:

```bash
LITELLM_API_URL=http://localhost:4000/v1/chat/completions
LITELLM_API_KEY=your-optional-api-key
```

### 3. Configure Models

Edit `backend/config.py` to match your self-hosted models:

```python
COUNCIL_MODELS = [
    "ollama/llama3.1",
    "ollama/mistral",
    "ollama/codellama",
]

CHAIRMAN_MODEL = "ollama/llama3.1"
```

## Running the Application

### Option 1: With Open WebUI (Recommended)

1. Start the LLM Council backend:
```bash
uv run python -m backend.main
```

2. Configure Open WebUI to use LLM Council as an OpenAI-compatible endpoint:
   - URL: `http://localhost:8001/v1`
   - Select "llm-council" as the model for multi-model deliberation
   - Or select individual models for direct queries

### Option 2: With Built-in Frontend

Terminal 1 (Backend):
```bash
uv run python -m backend.main
```

Terminal 2 (Frontend):
```bash
cd frontend
npm run dev
```

Then open http://localhost:5173 in your browser.

## OpenAI-Compatible API

The backend exposes OpenAI-compatible endpoints for Open WebUI integration:

- `GET /v1/models` - List available models (includes "llm-council" + individual models)
- `POST /v1/chat/completions` - Chat completions (streaming supported)

When you select "llm-council" as the model, it runs the full 3-stage deliberation process.

## Tech Stack

- **Backend:** FastAPI (Python 3.10+), async httpx, LiteLLM-compatible
- **Frontend:** Open WebUI (recommended) or React + Vite
- **LLM Routing:** LiteLLM proxy
- **Models:** Self-hosted via Ollama, vLLM, or any LiteLLM-supported backend
- **Storage:** JSON files in `data/conversations/`
- **Package Management:** uv for Python, npm for JavaScript
