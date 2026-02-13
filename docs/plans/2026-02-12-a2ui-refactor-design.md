# A2UI Refactor Design

Replace the current three-tier architecture (React SPA + REST BFF + A2A/PocketFlow backend) with a two-tier A2UI architecture using DeepAgents, the A2A protocol with A2UI extension, and `@a2ui-sdk/react`.

## Architecture

```
React App (@a2ui-sdk/react)
    | SSE (A2UI JSONL down) + POST (userActions up)
A2A Server (langgraph-api + A2UI extension)
    |
DeepAgent (create_deep_agent + custom tools)
    |
Custom Tools: execute_python, query_database, get_database_schema, load_uploaded_file
    |
Sandbox (Docker/multiprocessing) + DuckDB
```

The frontend and BFF merge into a single layer: the React client connects directly to the A2A server's `message/stream` endpoint. The agent produces A2UI JSONL messages (wrapped in A2A `DataPart`s with `mimeType: "application/json+a2ui"`) that the client renders via `A2UIRenderer`.

## Agent Design

PocketFlow is replaced by a DeepAgent (`create_deep_agent()` from the `deepagents` library). The hardcoded flow graph (classify intent -> generate code -> execute -> format) becomes an LLM-driven agent that uses tools as needed.

### Custom Tools

| Tool | Signature | Purpose |
|------|-----------|---------|
| `execute_python` | `(code: str) -> str` | Run Python in sandbox kernel, return stdout + artifact refs |
| `query_database` | `(sql: str) -> str` | Run SQL against DuckDB, return formatted results |
| `get_database_schema` | `() -> str` | Return table schemas for LLM context |
| `load_uploaded_file` | `(filename: str, data: bytes) -> str` | Load file into sandbox kernel |

### A2UI Message Generation

The agent's responses are translated to A2UI messages by an adapter layer (`a2ui_adapter.py`):

- Chat text -> `surfaceUpdate` with Text components
- Code execution -> `surfaceUpdate` with custom CodeBlock + OutputBlock components
- Plots (base64 SVG) -> `surfaceUpdate` with custom PlotViewer + `dataModelUpdate`
- Tables (HTML) -> `surfaceUpdate` with custom DataTable + `dataModelUpdate`
- Mermaid diagrams -> `surfaceUpdate` with custom MermaidChart component
- HTML output -> `surfaceUpdate` with custom HtmlViewer component (iframe)

Messages use the A2UI v0.8 format: `surfaceUpdate` -> `dataModelUpdate` -> `beginRendering`.

## Frontend Design

### Stack

- React 19 + TypeScript + Vite
- `@a2ui-sdk/react` (A2UIProvider, A2UIRenderer, useA2UIMessageHandler)
- Custom catalog extending `standardCatalog`

### File Structure

```
frontend/src/
  App.tsx              # A2UIProvider + SSE connection + chat input
  catalog.ts           # Custom component catalog
  components/
    PlotViewer.tsx     # base64 SVG renderer
    CodeBlock.tsx      # Collapsible Python code
    OutputBlock.tsx    # Collapsible stdout
    DataTable.tsx      # HTML table renderer
    MermaidChart.tsx   # Mermaid diagram renderer
    HtmlViewer.tsx     # Sandboxed iframe
    FileUpload.tsx     # File upload -> A2A DataPart
  main.tsx
  style.css
```

### Transport

1. User sends message -> POST to A2A `message/stream` with TextPart
2. Agent streams back -> SSE with A2A events containing A2UI DataParts
3. Client extracts A2UI JSONL from DataParts, feeds to `useA2UIMessageHandler`
4. User interactions -> `onAction` sends userAction via POST
5. File uploads -> A2A message with DataPart (Arrow IPC stream, base64)

## Server Design

Uses `langgraph-api` with built-in A2A endpoint:

- `/a2a/analytical-chatbot` - A2A endpoint (message/send, message/stream, tasks/get)
- `/.well-known/agent-card.json` - Agent discovery with A2UI extension capability

AgentCard advertises A2UI v0.8 extension support and custom catalog components.

Session management uses LangGraph's thread_id (mapped from A2A contextId). Sandbox kernel persists in agent state via checkpointer.

## File Changes

### New Files

| File | Purpose |
|------|---------|
| `server.py` | langgraph-api server setup |
| `agent.py` | DeepAgent configuration (model, tools, prompt) |
| `tools/execute_python.py` | Sandbox execution tool |
| `tools/database.py` | DuckDB query + schema tools |
| `tools/file_upload.py` | File loading tool |
| `a2ui_adapter.py` | Agent output -> A2UI JSONL translation |

### Deleted Files

| File | Reason |
|------|--------|
| `rest_server.py` | BFF eliminated |
| `a2a_server.py` | Replaced by langgraph-api |
| `agent_executor.py` | Replaced by agent.py |
| `main.py` | Replaced by server.py |
| `utils/flow.py` | PocketFlow removed |
| `utils/nodes.py` | PocketFlow removed |

### Kept (unchanged)

- `utils/call_llm.py`, `utils/database.py`, `utils/parse_code.py`
- `utils/sandbox_factory.py`, `utils/sandbox_interface.py`, `utils/llm_sandbox_kernel.py`
- `utils/sandbox.py`, `utils/kernel.py`
- `docker/`

### Modified

| File | Changes |
|------|---------|
| `requirements.txt` | Add deepagents, langgraph, langchain; remove pocketflow, a2a-sdk |
| `frontend/package.json` | Add @a2ui-sdk/react, upgrade react to 19; remove react-markdown, remark-gfm |
| `frontend/vite.config.ts` | Update proxy: /a2a/* instead of /api/* |
| `frontend/src/*` | Complete rewrite |

## Dependencies

### Python (new)

- `deepagents` >= 0.4
- `langgraph` >= 0.4
- `langgraph-api` >= 0.4.21
- `langchain` >= 0.3
- `langchain-anthropic` or `langchain-openai` (model providers)

### Python (removed)

- `pocketflow`

### Frontend (new)

- `@a2ui-sdk/react`
- `react@^19.0.0`
- `react-dom@^19.0.0`

### Frontend (removed)

- `react-markdown`
- `remark-gfm`
- `mermaid` (moves into MermaidChart custom component, still needed)
