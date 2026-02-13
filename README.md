# Analytical Chatbot (A2A + A2UI + DeepAgents)

AI-powered data analysis assistant with code execution and visualization. Uses Google's [A2UI protocol](https://a2ui.org/) for agent-to-UI communication and the [A2A protocol](https://google.github.io/A2A/) for agent interoperability.

## Architecture

```
React App (@a2ui-sdk/react)
    ↕ SSE (A2UI JSONL down) + POST (userActions up)
A2A Server (FastAPI + A2UI extension)
    ↕
DeepAgent (LangGraph create_deep_agent)
    ↕
Custom Tools: execute_python, query_database, get_database_schema
    ↕
Sandboxed Kernel (Docker or Multiprocessing) + DuckDB
```

## Key Modules

- `server.py`: FastAPI server with A2A endpoints, A2UI streaming, file upload
- `agent.py`: DeepAgent configuration with custom tools and system prompt
- `tools/`: Custom LangChain tools for sandbox execution and database queries
- `a2ui_adapter.py`: Translates agent output to A2UI v0.8 JSONL messages
- `frontend/`: React app with `@a2ui-sdk/react` and custom component catalog
- `utils/sandbox_factory.py`: Auto-detects best sandbox backend
- `utils/database.py`: DuckDB helpers and schema
- `utils/call_llm.py`: LLM wrapper and provider selection

## Quickstart

```bash
# Install Python dependencies
pip install -r requirements.txt

# Start the backend
python server.py

# In another terminal, start the frontend
cd frontend && npm install && npm run dev
```

Open http://localhost:5173 to use the chatbot.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | `openai` or `anthropic` | `openai` |
| `OPENAI_API_KEY` | OpenAI API key | — |
| `OPENAI_MODEL` | OpenAI model name | `gpt-4o` |
| `ANTHROPIC_API_KEY` | Anthropic API key | — |
| `ANTHROPIC_MODEL` | Anthropic model name | `claude-sonnet-4-5-20250929` |
| `SANDBOX_FORCE_BACKEND` | Force `docker` or `multiprocessing` | Auto-detect |
| `SANDBOX_DOCKER_IMAGE` | Custom Docker image | Default Python image |
| `SANDBOX_TIMEOUT` | Execution timeout in seconds | `30` |
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `8000` |

## Sandbox Configuration

| Backend | Isolation | Startup | Use Case |
|---------|-----------|---------|----------|
| **Docker** (llm-sandbox) | Full container isolation | ~2-30s | Production, untrusted code |
| **Multiprocessing** | Process isolation | Instant | Development, trusted environments |

### Using Docker Backend

```bash
# Build pre-configured image (recommended)
./docker/build-sandbox-image.sh

# Start with Docker sandbox
SANDBOX_DOCKER_IMAGE=analytical-chatbot-sandbox:latest SANDBOX_FORCE_BACKEND=docker python server.py
```

### Using Multiprocessing Backend

```bash
SANDBOX_FORCE_BACKEND=multiprocessing python server.py
```

## A2A Agent Card

The server exposes an A2A agent card at `/.well-known/agent-card.json` with A2UI extension capability. Other A2A agents can discover and communicate with this agent.

## Documentation

- `docs/plans/2026-02-12-a2ui-refactor-design.md`: Architecture design document
- `docs/plans/2026-02-12-a2ui-refactor-plan.md`: Implementation plan
