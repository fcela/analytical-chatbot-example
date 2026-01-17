# Analytical Chatbot (PocketFlow + A2A + React)

A web-based chatbot built with PocketFlow that can analyze uploaded CSV/JSON files, execute Python code in a sandboxed kernel, and stream results back over the A2A protocol (HTTP/JSON-RPC and gRPC).

## Architecture

- **A2A Backend**: HTTP/JSON-RPC + gRPC servers (`a2a_server.py`, `agent_executor.py`)
- **BFF REST API**: UI-facing REST service that bridges to A2A gRPC (`rest_server.py`)
- **Flow Control**: PocketFlow orchestration (`utils/flow.py`, `utils/nodes.py`)
- **Sandbox**: Multiprocessing Python kernel with a restricted namespace (`utils/sandbox.py`, `utils/kernel.py`)
- **Frontend**: React + Vite + TypeScript

```
Browser (React)  ->  BFF REST (8000)  ->  A2A gRPC (50051)
                                        |-> A2A HTTP/JSON-RPC (8001)
```

## Project Structure

```
analytical-chatbot-example/
├── a2a_server.py       # A2A server setup (HTTP + gRPC)
├── agent_executor.py   # A2A AgentExecutor (PocketFlow wrapper)
├── rest_server.py      # REST BFF for the React UI
├── main.py             # Launches backend + BFF
├── utils/nodes.py      # PocketFlow node definitions
├── utils/flow.py       # Flow orchestration
├── utils/
│   ├── kernel.py       # Sandboxed kernel worker (artifacts)
│   ├── sandbox.py      # Kernel process manager
│   ├── database.py     # DuckDB helpers
│   └── call_llm.py      # LLM wrapper
├── frontend/
│   ├── src/App.tsx     # React app
│   ├── src/style.css   # Styles
│   └── vite.config.ts  # Dev proxy to BFF
├── docs/
│   └── design.md       # Architecture/design notes
└── requirements.txt
```

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
cd frontend && npm install && cd ..
```

### 2. Set environment variables

```bash
# Option A: OpenAI (default)
export LLM_PROVIDER='openai'
export OPENAI_API_KEY='your-key'

# Option B: Anthropic (Claude)
export LLM_PROVIDER='anthropic'
export ANTHROPIC_API_KEY='your-key'

# Option C: Ollama (local)
export LLM_PROVIDER='ollama'
export OLLAMA_MODEL='qwen3:4b'

# Option D: Mock LLM
export MOCK_LLM='true'
```

### 3. Start the services

```bash
# Starts A2A backend (HTTP 8001 + gRPC 50051) and BFF REST (8000)
python main.py

# In another terminal
cd frontend && npm run dev
```

### 4. Open the app

Navigate to http://localhost:5173.

## A2A Protocol Endpoints

- **Agent Card**: `http://localhost:8001/.well-known/agent-card.json`
- **HTTP/JSON-RPC**: `http://localhost:8001/messages`
- **gRPC**: `localhost:50051`

## Environment Variables

### LLM Provider Selection

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | Provider (`openai`, `anthropic`, `ollama`) | `openai` |
| `MOCK_LLM` | Force mock responses | `false` |
| `LLM_TEMPERATURE` | Generation temperature | `0.7` |
| `LLM_DEBUG` | Enable debug logging | `false` |

### Provider-Specific Variables

**OpenAI:** `OPENAI_API_KEY`, `OPENAI_MODEL` (default `gpt-4o`)

**Anthropic:** `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` (default `claude-sonnet-4-20250514`)

**Ollama:** `OLLAMA_MODEL` (default `qwen3:4b`), `LLM_TIMEOUT` (seconds)

### Server Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `BACKEND_HOST` | A2A host | `0.0.0.0` |
| `BACKEND_HTTP_PORT` | A2A HTTP port | `8001` |
| `BACKEND_GRPC_PORT` | A2A gRPC port | `50051` |
| `HOST` | BFF host | `0.0.0.0` |
| `PORT` | BFF port | `8000` |

## Features

- **Intent Classification**: routes between conversation and code execution
- **Code Generation**: Polars + Altair-first Python generation
- **Sandboxed Execution**: persistent kernel with timeout and safety checks
- **Artifacts**: tables, plots, HTML dashboards, Mermaid diagrams
- **Database**: DuckDB tables (employees, products, sales, customers)
- **A2A Streaming**: status updates + artifacts over gRPC

## Sandbox Environment

The kernel exposes a constrained namespace with common data tools:
- **Polars** (`pl`), **Pandas** (`pd`), **NumPy** (`np`), **Altair** (`alt`)
- **DuckDB** via `query_db()` and helper functions
- **display()** / `show_table()` / `show_html()` to emit rich artifacts

Safety checks are lightweight string guards, so this is intended for local, trusted use and educational purposes.

## Development Notes

See `docs/design.md` for a deeper walkthrough of the A2A flow, artifact rendering, and sandbox internals.
