# Analytical Chatbot (A2A Agent + PocketFlow + React)

This directory contains the reusable modules that power the analytical chatbot: the PocketFlow nodes, sandbox kernel, and documentation. The runnable entrypoints live at the repo root (`main.py`, `rest_server.py`, `a2a_server.py`).

## How the pieces fit

```
Browser (React)  ->  BFF REST (rest_server.py)  ->  A2A gRPC (a2a_server.py)
                                                         |
                                                         v
                                               AgentExecutor + PocketFlow
                                                         |
                                                         v
                                                Sandboxed Kernel (Docker or multiprocessing)
```

## Key Modules

- `utils/flow.py`: PocketFlow orchestration
- `utils/sandbox_factory.py`: Auto-detects and creates the best available sandbox backend
- `utils/llm_sandbox_kernel.py`: Docker-based sandbox using llm-sandbox library
- `utils/sandbox.py`, `utils/kernel.py`: Multiprocessing sandbox fallback
- `utils/database.py`: DuckDB helpers and schema
- `utils/call_llm.py`: LLM wrapper and provider selection

## Quickstart (from repo root)

```bash
pip install -r requirements.txt
python main.py
```

Then start the UI:

```bash
cd frontend && npm install && npm run dev
```

## Sandbox Configuration

The chatbot supports two sandbox backends for executing Python code:

| Backend | Isolation | Startup | Use Case |
|---------|-----------|---------|----------|
| **Docker** (llm-sandbox) | Full container isolation | ~2-30s | Production, untrusted code |
| **Multiprocessing** | Process isolation | Instant | Development, trusted environments |

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SANDBOX_FORCE_BACKEND` | Force `docker` or `multiprocessing` | Auto-detect |
| `SANDBOX_PREFER_DOCKER` | Prefer Docker when available | `true` |
| `SANDBOX_TIMEOUT` | Execution timeout in seconds | `30` |
| `SANDBOX_DOCKER_IMAGE` | Custom Docker image | Default Python image |
| `SANDBOX_SKIP_INSTALL` | Skip pip install in container | Auto-detect |

### Using Docker Backend

```bash
# Basic Docker backend (slow first startup due to pip install)
SANDBOX_FORCE_BACKEND=docker python main.py

# With pre-built image (fast startup)
./docker/build-sandbox-image.sh
SANDBOX_DOCKER_IMAGE=analytical-chatbot-sandbox:latest SANDBOX_FORCE_BACKEND=docker python main.py
```

### Using Multiprocessing Backend

```bash
SANDBOX_FORCE_BACKEND=multiprocessing python main.py
```

## Pre-built Docker Image

For fast Docker startup (~2-3s instead of ~30-60s), build the pre-configured image:

```bash
# Build once
./docker/build-sandbox-image.sh

# Use it
export SANDBOX_DOCKER_IMAGE=analytical-chatbot-sandbox:latest
export SANDBOX_FORCE_BACKEND=docker
python main.py
```

The pre-built image includes: pandas, numpy, polars, pyarrow, altair, vl-convert-python, duckdb, tabulate.

## Logging

The sandbox system provides detailed logging to help debug issues:

```
[SANDBOX] Creating sandbox instance...
[SANDBOX] SANDBOX_FORCE_BACKEND=docker -> forcing 'docker' backend
[SANDBOX] SUCCESS: Using Docker backend via llm-sandbox (FORCED)
[DOCKER-SANDBOX] Starting Docker sandbox session...
[DOCKER-SANDBOX] Docker sandbox session started successfully!
```

Set `LOG_LEVEL=DEBUG` for more verbose output.

## Documentation

- `docs/design.md` explains the A2A + PocketFlow dataflow, sandbox architecture, and artifact pipeline.
