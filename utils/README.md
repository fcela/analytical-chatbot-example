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
                                                Sandboxed Kernel (utils/kernel.py)
```

## Key Modules

- `flow.py`: PocketFlow orchestration (in this folder)
- `sandbox.py`, `kernel.py`: multiprocessing sandbox and artifact handling
- `database.py`: DuckDB helpers and schema
- `call_llm.py`: LLM wrapper and provider selection

## Quickstart (from repo root)

```bash
pip install -r requirements.txt
python main.py
```

Then start the UI:

```bash
cd frontend && npm install && npm run dev
```

## Documentation

- `docs/design.md` explains the A2A + PocketFlow dataflow and artifact pipeline.
