# CLAUDE.md - Technical Notes for LLM Council

This file contains technical details, architectural decisions, and important implementation notes for future development sessions.

## Project Overview

LLM Council is a 3-stage deliberation system where multiple self-hosted LLMs collaboratively answer user questions. Uses LiteLLM as the routing layer and supports Open WebUI as the frontend. The key innovation is anonymized peer review in Stage 2, preventing models from playing favorites.

## Architecture

```
Open WebUI → LLM Council API → LiteLLM Proxy → Self-hosted Models
                ↓
         3-Stage Deliberation
```

### Backend Structure (`backend/`)

**`config.py`**
- Contains `COUNCIL_MODELS` (list of model identifiers for your LiteLLM setup)
- Contains `CHAIRMAN_MODEL` (model that synthesizes final answer)
- Uses environment variables `LITELLM_API_URL` and `LITELLM_API_KEY` from `.env`
- Backend runs on **port 8001**

**`llm_client.py`**
- `query_model()`: Single async model query via LiteLLM proxy
- `query_models_parallel()`: Parallel queries using `asyncio.gather()`
- `query_model_with_retry()`: Retry logic with exponential backoff
- Returns dict with 'content' and optional 'reasoning_details'
- Graceful degradation: returns None on failure, continues with successful responses

**`council.py`** - The Core Logic
- `stage1_collect_responses()`: Parallel queries to all council models
- `stage2_collect_rankings()`:
  - Anonymizes responses as "Response A, B, C, etc."
  - Creates `label_to_model` mapping for de-anonymization
  - Prompts models to evaluate and rank (with strict format requirements)
  - Returns tuple: (rankings_list, label_to_model_dict)
  - Each ranking includes both raw text and `parsed_ranking` list
- `stage3_synthesize_final()`: Chairman synthesizes from all responses + rankings
- `parse_ranking_from_text()`: Extracts "FINAL RANKING:" section
- `calculate_aggregate_rankings()`: Computes average rank position across all peer evaluations

**`storage.py`**
- JSON-based conversation storage in `data/conversations/`
- Each conversation: `{id, created_at, messages[]}`
- Assistant messages contain: `{role, stage1, stage2, stage3}`
- Note: metadata (label_to_model, aggregate_rankings) is NOT persisted to storage, only returned via API

**`main.py`**
- FastAPI app with CORS enabled
- Native LLM Council API endpoints (`/api/conversations/*`)
- **OpenAI-compatible endpoints for Open WebUI** (`/v1/models`, `/v1/chat/completions`)

### OpenAI-Compatible API (for Open WebUI)

**`GET /v1/models`**
- Returns available models including "llm-council" as a virtual model
- Also lists individual council models for direct queries

**`POST /v1/chat/completions`**
- When `model: "llm-council"`: Runs full 3-stage deliberation
- When `model: "<individual-model>"`: Direct query to that model
- Supports streaming (`stream: true`)
- Returns markdown-formatted response with council decision, rankings, and individual responses

### Frontend Options

**Option 1: Open WebUI (Recommended)**
- Configure as OpenAI-compatible connection
- URL: `http://localhost:8001/v1`
- Select "llm-council" model for deliberation

**Option 2: Built-in React Frontend (`frontend/src/`)**

**`App.jsx`**
- Main orchestration: manages conversations list and current conversation
- Handles message sending and metadata storage

**`components/ChatInterface.jsx`**
- Multiline textarea (3 rows, resizable)
- Enter to send, Shift+Enter for new line

**`components/Stage1.jsx`**
- Tab view of individual model responses
- ReactMarkdown rendering

**`components/Stage2.jsx`**
- Tab view showing RAW evaluation text from each model
- De-anonymization happens CLIENT-SIDE for display
- Shows "Extracted Ranking" below each evaluation

**`components/Stage3.jsx`**
- Final synthesized answer from chairman
- Green-tinted background to highlight conclusion

## Key Design Decisions

### LiteLLM Integration
- Uses LiteLLM proxy for unified access to self-hosted models
- Ollama, vLLM, and any LiteLLM-supported backend work
- Model identifiers follow LiteLLM format (e.g., `ollama/llama3.1`)

### Stage 2 Prompt Format
The Stage 2 prompt is very specific to ensure parseable output:
```
1. Evaluate each response individually first
2. Provide "FINAL RANKING:" header
3. Numbered list format: "1. Response C", "2. Response A", etc.
4. No additional text after ranking section
```

### De-anonymization Strategy
- Models receive: "Response A", "Response B", etc.
- Backend creates mapping: `{"Response A": "ollama/llama3.1", ...}`
- Frontend displays model names in **bold** for readability
- This prevents bias while maintaining transparency

### Error Handling Philosophy
- Continue with successful responses if some models fail (graceful degradation)
- Never fail the entire request due to single model failure
- Log errors but don't expose to user unless all models fail

## Configuration

### Environment Variables (`.env`)
```bash
LITELLM_API_URL=http://localhost:4000/v1/chat/completions
LITELLM_API_KEY=optional-key
```

### Model Configuration (`backend/config.py`)
```python
COUNCIL_MODELS = [
    "ollama/llama3.1",
    "ollama/mistral",
    "ollama/codellama",
]
CHAIRMAN_MODEL = "ollama/llama3.1"
DEFAULT_TIMEOUT = 120.0
MAX_RETRIES = 3
```

## Important Implementation Details

### Relative Imports
All backend modules use relative imports (e.g., `from .config import ...`) not absolute imports. This is critical for Python's module system to work correctly when running as `python -m backend.main`.

### Port Configuration
- Backend: 8001
- Frontend: 5173 (Vite default)
- LiteLLM proxy: 4000 (default)

### Markdown Rendering
All ReactMarkdown components must be wrapped in `<div className="markdown-content">` for proper spacing.

## Common Gotchas

1. **Module Import Errors**: Always run backend as `python -m backend.main` from project root
2. **CORS Issues**: Frontend must match allowed origins in `main.py` CORS middleware
3. **Ranking Parse Failures**: If models don't follow format, fallback regex extracts any "Response X" patterns
4. **LiteLLM Connection**: Ensure LiteLLM proxy is running before starting the council backend

## 50 Features

The codebase includes 50 features across 6 categories:
- Features 1-10: Core Backend (delete, timeout, health, retry, rate limiting, logging, metrics)
- Features 11-20: Council Logic (weighted rankings, Borda count, tie-breaking, confidence, multi-turn)
- Features 21-30: Storage (pagination, archiving, export/import, tags, search)
- Features 31-40: Validation & Security (sanitization, length limits, content filtering, rate limits)
- Features 41-50: Utilities (token estimation, similarity, quality metrics, consensus scoring)

All features are tested in `backend/tests/test_features.py` (75 tests).

## Data Flow Summary

```
User Query (via Open WebUI or React frontend)
    ↓
/v1/chat/completions or /api/conversations/{id}/message
    ↓
Stage 1: Parallel queries → [individual responses]
    ↓
Stage 2: Anonymize → Parallel ranking queries → [evaluations + parsed rankings]
    ↓
Aggregate Rankings Calculation → [sorted by avg position]
    ↓
Stage 3: Chairman synthesis with full context
    ↓
Return: Formatted markdown (OpenAI API) or structured JSON (native API)
```

The entire flow is async/parallel where possible to minimize latency.
