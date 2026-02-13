# A2UI Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the three-tier architecture (React SPA + REST BFF + A2A/PocketFlow) with a two-tier A2UI architecture using DeepAgents, A2A protocol with A2UI extension, and `@a2ui-sdk/react`.

**Architecture:** The React client connects directly to an A2A server (powered by `langgraph-api`) that hosts a DeepAgent with custom tools for sandbox execution, database queries, and file loading. The agent produces A2UI JSONL messages wrapped in A2A `DataPart`s. The React client renders them via `A2UIRenderer` with a custom component catalog.

**Tech Stack:** Python 3.11+, `deepagents`, `langgraph`, `langgraph-api`, `langchain-anthropic`/`langchain-openai`, React 19, `@a2ui-sdk/react`, Vite, TypeScript

**Design doc:** `docs/plans/2026-02-12-a2ui-refactor-design.md`

---

### Task 1: Update Python dependencies

**Files:**
- Modify: `requirements.txt`

**Step 1: Update requirements.txt**

Replace the contents of `requirements.txt`:

```
# Agent framework
deepagents>=0.4
langgraph>=0.4
langgraph-api>=0.4.21
langchain>=0.3

# LLM providers for LangChain
langchain-openai
langchain-anthropic

# Web server
fastapi
uvicorn
starlette
python-multipart
pydantic
httpx

# LLM clients (used by utils/call_llm.py for standalone usage)
openai
anthropic
ollama

# Data analysis
polars
pandas
numpy
altair>=6.0.0
vl-convert-python
duckdb

# YAML parsing for structured outputs
pyyaml

# Optional: For Docker-based code sandboxing (requires Docker)
llm-sandbox[docker]
```

**Step 2: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: All packages install without errors.

**Step 3: Verify deepagents import**

Run: `python -c "from deepagents import create_deep_agent; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add requirements.txt
git commit -m "chore: update deps for A2UI refactor - add deepagents, langgraph; remove pocketflow"
```

---

### Task 2: Create custom tools for the DeepAgent

The PocketFlow nodes (`utils/nodes.py`) contained the logic for code generation, execution, intent classification, and result formatting. In the DeepAgent model, the LLM decides what to do and uses tools. We need three custom tools that wrap the existing infrastructure.

**Files:**
- Create: `tools/__init__.py`
- Create: `tools/execute_python.py`
- Create: `tools/database.py`

**Step 1: Create tools directory and __init__.py**

Create `tools/__init__.py`:

```python
"""Custom tools for the analytical chatbot DeepAgent."""
```

**Step 2: Create execute_python tool**

Create `tools/execute_python.py`. This wraps `SandboxInterface.execute()`:

```python
"""Sandbox code execution tool for the DeepAgent."""

import json
from langchain_core.tools import tool


@tool
def execute_python(code: str) -> str:
    """Execute Python code in a sandboxed environment.

    Use this tool to run data analysis code. The sandbox has access to:
    - polars (as pl), pandas (as pd), numpy (as np)
    - altair (as alt) for visualizations
    - duckdb for SQL queries via query_db(sql)
    - display(obj, label=None) to show DataFrames, plots, HTML, or Mermaid diagrams
    - print_md(df) to print DataFrames as markdown tables
    - show_table(df) and show_html(html) as shorthands

    IMPORTANT Polars API notes:
    - Use df.with_columns() NOT df.with_column()
    - Use df.group_by() NOT df.groupby()
    - Use df.rename({"old": "new"}) NOT df.rename(columns=...)
    - Cast Decimal to Float64 before charting with Altair

    Args:
        code: Python code to execute. Must be valid Python.

    Returns:
        JSON string with keys: success, stdout, artifacts, error
    """
    # The kernel is injected into tool context at runtime via RunnableConfig
    # We access it through the global _kernel reference set by the agent
    from tools._runtime import get_kernel

    kernel = get_kernel()
    if not kernel:
        return json.dumps({"success": False, "error": "Sandbox kernel not initialized"})

    result = kernel.execute(code)
    return json.dumps({
        "success": result.get("success", False),
        "stdout": result.get("stdout", ""),
        "artifacts": result.get("artifacts", {}),
        "error": result.get("error"),
    })
```

**Step 3: Create database tools**

Create `tools/database.py`. This wraps `utils/database.py`:

```python
"""DuckDB database tools for the DeepAgent."""

from langchain_core.tools import tool


@tool
def query_database(sql: str) -> str:
    """Execute a SQL query against the DuckDB database.

    Available tables: employees, products, sales, customers.
    Returns results as a formatted string.

    Args:
        sql: SQL query to execute.

    Returns:
        Query results as a formatted table string, or error message.
    """
    try:
        from utils.database import execute_query
        result = execute_query(sql)
        return str(result)
    except Exception as e:
        return f"Query error: {e}"


@tool
def get_database_schema() -> str:
    """Get the schema of all available database tables.

    Returns column names, types, and row counts for each table.
    Use this before writing SQL queries to understand the data structure.

    Returns:
        Formatted schema description string.
    """
    try:
        from utils.database import get_schema_description
        return get_schema_description()
    except Exception as e:
        return f"Schema error: {e}"
```

**Step 4: Create runtime context module**

Create `tools/_runtime.py`. This holds the per-request kernel reference:

```python
"""Runtime context for tool execution.

Provides per-session sandbox kernel access to tools without passing
the kernel through LangChain's tool interface.
"""

import threading
from typing import Optional
from utils.sandbox_interface import SandboxInterface

_thread_local = threading.local()


def set_kernel(kernel: Optional[SandboxInterface]) -> None:
    """Set the sandbox kernel for the current thread."""
    _thread_local.kernel = kernel


def get_kernel() -> Optional[SandboxInterface]:
    """Get the sandbox kernel for the current thread."""
    return getattr(_thread_local, "kernel", None)
```

**Step 5: Verify tools import**

Run: `python -c "from tools.execute_python import execute_python; from tools.database import query_database, get_database_schema; print('OK')"`
Expected: `OK`

**Step 6: Commit**

```bash
git add tools/
git commit -m "feat: add custom DeepAgent tools for sandbox execution and database queries"
```

---

### Task 3: Create the A2UI adapter

The adapter translates DeepAgent output (text + tool results with artifacts) into A2UI v0.8 JSONL messages.

**Files:**
- Create: `a2ui_adapter.py`

**Step 1: Create the adapter**

Create `a2ui_adapter.py`:

```python
"""Translate agent responses into A2UI v0.8 JSONL messages.

Produces surfaceUpdate, dataModelUpdate, and beginRendering messages
for the A2UI React renderer. Messages are serialized as JSON lines.
"""

import json
import uuid
from typing import Any


def _make_id(prefix: str = "c") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def create_chat_surface(
    message_text: str,
    code: str | None = None,
    output: str | None = None,
    artifacts: dict[str, dict[str, Any]] | None = None,
    error: str | None = None,
    surface_id: str = "chat",
) -> list[dict]:
    """Build A2UI messages for a single chat response.

    Args:
        message_text: The agent's text response.
        code: Generated Python code (if any).
        output: Execution stdout (if any).
        artifacts: Dict of artifact_id -> {type, content}.
        error: Error message (if any).
        surface_id: The A2UI surface identifier.

    Returns:
        List of A2UI message dicts (surfaceUpdate, dataModelUpdate, beginRendering).
    """
    messages: list[dict] = []
    components: list[dict] = []
    data_contents: list[dict] = []
    children_ids: list[str] = []

    # Root container
    root_id = _make_id("root")

    # 1. Text message
    if message_text:
        text_id = _make_id("text")
        components.append({
            "id": text_id,
            "component": {
                "Text": {
                    "text": {"literalString": message_text},
                    "usageHint": "body",
                }
            }
        })
        children_ids.append(text_id)

    # 2. Artifacts (plots, tables, mermaid, html, markdown)
    if artifacts:
        for art_id, art_data in artifacts.items():
            art_type = art_data.get("type", "unknown")
            content = art_data.get("content", "")
            comp_id = _make_id(art_type)
            data_path = f"/artifacts/{art_id}"

            if art_type in ("plot", "svg"):
                components.append({
                    "id": comp_id,
                    "component": {
                        "PlotViewer": {
                            "data": {"path": data_path},
                        }
                    }
                })
                data_contents.append({
                    "key": art_id,
                    "valueString": content,
                })

            elif art_type == "table":
                components.append({
                    "id": comp_id,
                    "component": {
                        "DataTable": {
                            "html": {"path": data_path},
                        }
                    }
                })
                data_contents.append({
                    "key": art_id,
                    "valueString": content,
                })

            elif art_type == "mermaid":
                components.append({
                    "id": comp_id,
                    "component": {
                        "MermaidChart": {
                            "definition": {"path": data_path},
                        }
                    }
                })
                data_contents.append({
                    "key": art_id,
                    "valueString": content,
                })

            elif art_type == "html":
                components.append({
                    "id": comp_id,
                    "component": {
                        "HtmlViewer": {
                            "html": {"path": data_path},
                        }
                    }
                })
                data_contents.append({
                    "key": art_id,
                    "valueString": content,
                })

            elif art_type == "markdown":
                components.append({
                    "id": comp_id,
                    "component": {
                        "Text": {
                            "text": {"literalString": content},
                            "usageHint": "body",
                        }
                    }
                })

            else:
                continue

            children_ids.append(comp_id)

    # 3. Code block (collapsible)
    if code:
        code_id = _make_id("code")
        components.append({
            "id": code_id,
            "component": {
                "CodeBlock": {
                    "code": {"path": "/response/code"},
                    "language": {"literalString": "python"},
                }
            }
        })
        data_contents.append({"key": "code", "valueString": code})
        children_ids.append(code_id)

    # 4. Output block (collapsible)
    if output:
        output_id = _make_id("output")
        components.append({
            "id": output_id,
            "component": {
                "OutputBlock": {
                    "output": {"path": "/response/output"},
                }
            }
        })
        data_contents.append({"key": "output", "valueString": output})
        children_ids.append(output_id)

    # 5. Error block
    if error:
        error_id = _make_id("error")
        components.append({
            "id": error_id,
            "component": {
                "Text": {
                    "text": {"literalString": f"Error: {error}"},
                    "usageHint": "body",
                }
            }
        })
        children_ids.append(error_id)

    # Root column container
    components.append({
        "id": root_id,
        "component": {
            "Column": {
                "children": {"explicitList": children_ids},
                "alignment": "start",
            }
        }
    })

    # Build messages
    messages.append({
        "surfaceUpdate": {
            "surfaceId": surface_id,
            "components": components,
        }
    })

    if data_contents:
        messages.append({
            "dataModelUpdate": {
                "surfaceId": surface_id,
                "path": "/response",
                "contents": data_contents,
            }
        })

    messages.append({
        "beginRendering": {
            "surfaceId": surface_id,
            "root": root_id,
        }
    })

    return messages


def to_jsonl(messages: list[dict]) -> str:
    """Serialize A2UI messages to JSONL format."""
    return "\n".join(json.dumps(m, separators=(",", ":")) for m in messages)
```

**Step 2: Verify adapter**

Run: `python -c "from a2ui_adapter import create_chat_surface, to_jsonl; msgs = create_chat_surface('Hello!'); print(to_jsonl(msgs))"`
Expected: Three JSONL lines (surfaceUpdate, beginRendering) printed.

**Step 3: Commit**

```bash
git add a2ui_adapter.py
git commit -m "feat: add A2UI adapter for translating agent output to A2UI JSONL"
```

---

### Task 4: Create the DeepAgent configuration

**Files:**
- Create: `agent.py`

**Step 1: Create agent.py**

```python
"""DeepAgent configuration for the analytical chatbot.

Creates a LangGraph-compiled agent with custom tools for data analysis,
code execution, and database queries.
"""

import os
from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model

from tools.execute_python import execute_python
from tools.database import query_database, get_database_schema

SYSTEM_PROMPT = """You are an AI-powered analytical assistant. You help users analyze data, \
generate visualizations, and answer questions about their datasets.

You have access to the following tools:

1. **execute_python** - Run Python code in a sandboxed environment with polars, pandas, \
numpy, altair, and duckdb pre-installed. Use display() to show plots, tables, and diagrams. \
Use print() for text output.

2. **query_database** - Run SQL queries against the built-in DuckDB database. \
Available tables: employees, products, sales, customers.

3. **get_database_schema** - Get column names, types, and row counts for all database tables. \
Call this before writing SQL queries.

When users ask for data analysis:
1. If they mention database tables, use get_database_schema first, then either query_database \
for simple queries or execute_python for complex analysis with visualizations.
2. If they uploaded files, use execute_python to work with the pre-loaded DataFrames.
3. For visualizations, always use Altair (import altair as alt) and display() to show charts.
4. Always cite specific numbers from the results in your response.
5. For Mermaid diagrams, generate the mermaid string and call display() on it.

CRITICAL Polars API notes:
- Use df.with_columns() NOT df.with_column()
- Use df.group_by() NOT df.groupby()
- Cast Decimal to Float64 before charting with Altair

For general conversation, respond helpfully and concisely without using tools.
"""


def create_agent():
    """Create the analytical chatbot DeepAgent.

    Returns a compiled LangGraph graph that can be served via langgraph-api.
    """
    # Determine model from environment
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if provider == "anthropic":
        model_name = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
        model_id = f"anthropic:{model_name}"
    elif provider == "openai":
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o")
        model_id = f"openai:{model_name}"
    else:
        model_id = f"openai:{os.getenv('OPENAI_MODEL', 'gpt-4o')}"

    model = init_chat_model(model_id)

    agent = create_deep_agent(
        model=model,
        tools=[execute_python, query_database, get_database_schema],
        system_prompt=SYSTEM_PROMPT,
    )

    return agent
```

**Step 2: Verify agent creation**

Run: `python -c "from agent import create_agent; a = create_agent(); print(type(a))"`
Expected: `<class 'langgraph.graph.state.CompiledStateGraph'>` (or similar LangGraph type)

Note: This requires an API key to be set (OPENAI_API_KEY or ANTHROPIC_API_KEY).

**Step 3: Commit**

```bash
git add agent.py
git commit -m "feat: add DeepAgent configuration with custom tools and system prompt"
```

---

### Task 5: Create the server

**Files:**
- Create: `server.py`

**Step 1: Create server.py**

This replaces both `main.py` (launcher) and `a2a_server.py` (A2A server). It uses `langgraph-api` with built-in A2A support, plus a custom SSE endpoint for A2UI streaming.

```python
"""A2UI-enabled A2A server for the analytical chatbot.

Hosts the DeepAgent via langgraph-api with:
- A2A endpoint at /a2a/analytical-chatbot (message/send, message/stream)
- Agent card at /.well-known/agent-card.json
- Static file serving for the frontend build
- File upload endpoint
"""

import asyncio
import base64
import io
import json
import logging
import os
import uuid
from typing import Any

import polars as pl
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

from agent import create_agent
from a2ui_adapter import create_chat_surface, to_jsonl
from tools._runtime import set_kernel, get_kernel
from utils.sandbox_factory import create_sandbox

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create the agent graph
agent = create_agent()

# Session storage: context_id -> {kernel, history, files}
sessions: dict[str, dict[str, Any]] = {}

app = FastAPI(title="Analytical Chatbot A2UI Server")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:[0-9]+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_or_create_session(context_id: str) -> dict[str, Any]:
    """Get or create a session with a sandbox kernel."""
    if context_id not in sessions:
        kernel = create_sandbox()
        logger.info(f"Created session {context_id} with {kernel.backend_name} backend")
        sessions[context_id] = {
            "kernel": kernel,
            "history": [],
            "files": [],
        }
    return sessions[context_id]


@app.get("/")
def health():
    return {"status": "running", "protocol": "a2ui+a2a"}


@app.get("/.well-known/agent-card.json")
def agent_card():
    """A2A Agent Card with A2UI extension."""
    host = os.getenv("HOST", "localhost")
    port = int(os.getenv("PORT", "8000"))
    return {
        "name": "Analytical Chatbot",
        "description": "AI-powered data analysis assistant with code execution and visualization.",
        "url": f"http://{host}:{port}",
        "version": "2.0.0",
        "capabilities": {
            "streaming": True,
            "extensions": [
                {
                    "uri": "https://a2ui.org/a2a-extension/a2ui/v0.8",
                    "params": {
                        "supportedCatalogIds": [
                            "https://a2ui.org/specification/v0_8/standard_catalog_definition.json",
                            "analytical-chatbot-custom",
                        ],
                        "acceptsInlineCatalogs": False,
                    },
                }
            ],
        },
        "skills": [
            {"id": "data-analysis", "name": "Data Analysis", "description": "Analyze data files and databases"},
            {"id": "visualization", "name": "Visualization", "description": "Create charts with Altair"},
            {"id": "database-query", "name": "Database Query", "description": "Query DuckDB tables"},
        ],
    }


@app.post("/a2a/message/stream")
async def message_stream(request: Request):
    """A2A message/stream endpoint - returns A2UI JSONL via SSE."""
    body = await request.json()

    # Extract message and context from A2A format
    message = body.get("message", {})
    context_id = message.get("contextId") or message.get("context_id") or str(uuid.uuid4())

    # Extract text from parts
    user_text = ""
    for part in message.get("parts", []):
        if "text" in part:
            user_text = part["text"]
        elif "root" in part and "text" in part["root"]:
            user_text = part["root"]["text"]

    if not user_text:
        raise HTTPException(status_code=400, detail="No text content in message")

    session = get_or_create_session(context_id)

    async def event_stream():
        """Run the agent and stream A2UI messages as SSE."""
        # Set kernel for this thread so tools can access it
        set_kernel(session["kernel"])

        try:
            # Build input for the DeepAgent
            input_messages = []
            for h in session["history"]:
                input_messages.append(h)
            input_messages.append({"role": "user", "content": user_text})

            # Run the agent
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: agent.invoke({"messages": input_messages}),
            )

            # Extract the assistant's last message
            result_messages = result.get("messages", [])
            assistant_msg = ""
            code = None
            output = None
            artifacts = {}
            error = None

            for msg in result_messages:
                if hasattr(msg, "content") and hasattr(msg, "type"):
                    if msg.type == "ai":
                        assistant_msg = msg.content
                    elif msg.type == "tool":
                        # Parse tool results for artifacts
                        try:
                            tool_result = json.loads(msg.content)
                            if isinstance(tool_result, dict):
                                if tool_result.get("success") is not None:
                                    # This is an execute_python result
                                    if tool_result.get("success"):
                                        output = tool_result.get("stdout", "")
                                        artifacts.update(tool_result.get("artifacts", {}))
                                    else:
                                        error = tool_result.get("error")
                        except (json.JSONDecodeError, TypeError):
                            pass

            # Update session history
            session["history"].append({"role": "user", "content": user_text})
            session["history"].append({"role": "assistant", "content": assistant_msg})

            # Generate A2UI messages
            a2ui_messages = create_chat_surface(
                message_text=assistant_msg,
                code=code,
                output=output,
                artifacts=artifacts,
                error=error,
            )

            # Stream as SSE with A2A DataPart wrapping
            for a2ui_msg in a2ui_messages:
                event_data = json.dumps({
                    "parts": [{
                        "root": {
                            "data": a2ui_msg,
                            "metadata": {"mimeType": "application/json+a2ui"},
                        }
                    }]
                })
                yield f"data: {event_data}\n\n"

            # Send completion event
            yield f"data: {json.dumps({'status': {'state': 'completed'}, 'contextId': context_id})}\n\n"

        except Exception as e:
            logger.exception("Agent execution error")
            error_msg = json.dumps({"error": str(e)})
            yield f"data: {error_msg}\n\n"

        finally:
            set_kernel(None)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Context-Id": context_id,
        },
    )


@app.post("/a2a/message/send")
async def message_send(request: Request):
    """A2A message/send endpoint - returns complete response."""
    body = await request.json()
    message = body.get("message", {})
    context_id = message.get("contextId") or message.get("context_id") or str(uuid.uuid4())

    user_text = ""
    for part in message.get("parts", []):
        if "text" in part:
            user_text = part["text"]
        elif "root" in part and "text" in part["root"]:
            user_text = part["root"]["text"]

    if not user_text:
        raise HTTPException(status_code=400, detail="No text content in message")

    session = get_or_create_session(context_id)
    set_kernel(session["kernel"])

    try:
        input_messages = list(session["history"])
        input_messages.append({"role": "user", "content": user_text})

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: agent.invoke({"messages": input_messages}),
        )

        result_messages = result.get("messages", [])
        assistant_msg = ""
        artifacts = {}

        for msg in result_messages:
            if hasattr(msg, "content") and hasattr(msg, "type"):
                if msg.type == "ai":
                    assistant_msg = msg.content
                elif msg.type == "tool":
                    try:
                        tool_result = json.loads(msg.content)
                        if isinstance(tool_result, dict) and tool_result.get("artifacts"):
                            artifacts.update(tool_result["artifacts"])
                    except (json.JSONDecodeError, TypeError):
                        pass

        session["history"].append({"role": "user", "content": user_text})
        session["history"].append({"role": "assistant", "content": assistant_msg})

        a2ui_messages = create_chat_surface(
            message_text=assistant_msg,
            artifacts=artifacts,
        )

        return {
            "contextId": context_id,
            "status": {"state": "completed"},
            "parts": [
                {"root": {"data": msg, "metadata": {"mimeType": "application/json+a2ui"}}}
                for msg in a2ui_messages
            ],
        }

    finally:
        set_kernel(None)


@app.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
    """Handle file uploads - load into sandbox kernel."""
    context_id = request.headers.get("X-Context-Id") or request.cookies.get("context_id") or str(uuid.uuid4())
    session = get_or_create_session(context_id)
    filename = file.filename

    if not filename:
        raise HTTPException(status_code=400, detail="No filename")

    content = await file.read()

    try:
        if filename.endswith(".csv"):
            try:
                df = pl.read_csv(io.BytesIO(content))
            except Exception:
                df = pl.read_csv(io.BytesIO(content), encoding="latin-1")
        elif filename.endswith(".json"):
            json_data = json.loads(content.decode("utf-8"))
            if isinstance(json_data, list):
                df = pl.DataFrame(json_data)
            elif isinstance(json_data, dict):
                df = pl.DataFrame(json_data) if all(isinstance(v, list) for v in json_data.values()) else pl.DataFrame([json_data])
            else:
                raise ValueError("Invalid JSON structure")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Use CSV or JSON.")

        # Serialize to Arrow IPC and load into kernel
        sink = io.BytesIO()
        df.write_ipc_stream(sink)
        binary_data = sink.getvalue()

        kernel = session["kernel"]
        success = kernel.load_file(filename, binary_data)

        if success:
            session["files"].append({
                "filename": filename,
                "rows": df.height,
                "columns": df.width,
            })
            return {
                "filename": filename,
                "rows": df.height,
                "columns": df.width,
                "contextId": context_id,
            }
        else:
            raise HTTPException(status_code=500, detail=f"Failed to load {filename} into sandbox")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Processing failed: {e}")


@app.get("/files")
def list_files(request: Request):
    """List uploaded files for a session."""
    context_id = request.headers.get("X-Context-Id") or request.cookies.get("context_id")
    if not context_id or context_id not in sessions:
        return {"files": []}
    return {"files": sessions[context_id]["files"]}


@app.get("/database")
def database_info():
    """Get database table information."""
    try:
        from utils.database import get_table_info, HAS_DUCKDB
        if not HAS_DUCKDB:
            return {"available": False}
        from utils.database import get_connection
        get_connection()
        all_tables = get_table_info()
        builtin = {k: v for k, v in all_tables.items() if not k.startswith("saved_")}
        return {"available": True, "tables": builtin}
    except Exception:
        return {"available": False}


@app.post("/clear")
def clear_session(request: Request):
    """Clear a session."""
    context_id = request.headers.get("X-Context-Id") or request.cookies.get("context_id")
    if context_id and context_id in sessions:
        session = sessions[context_id]
        session["kernel"].terminate()
        del sessions[context_id]
    return {"status": "ok"}


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    logger.info("=" * 60)
    logger.info("ANALYTICAL CHATBOT - A2UI SERVER")
    logger.info("=" * 60)
    logger.info(f"  Host: {host}:{port}")
    logger.info(f"  LLM Provider: {os.getenv('LLM_PROVIDER', 'openai')}")
    logger.info(f"  Sandbox: {os.getenv('SANDBOX_FORCE_BACKEND', 'auto-detect')}")
    logger.info("=" * 60)

    uvicorn.run(app, host=host, port=port, log_level="info")
```

**Step 2: Verify server starts**

Run: `python server.py`
Expected: Server starts on port 8000. Health check at `http://localhost:8000/` returns `{"status": "running", "protocol": "a2ui+a2a"}`.

Note: Requires API key (OPENAI_API_KEY or ANTHROPIC_API_KEY).

**Step 3: Commit**

```bash
git add server.py
git commit -m "feat: add A2UI server with A2A endpoints, file upload, and session management"
```

---

### Task 6: Update frontend dependencies

**Files:**
- Modify: `frontend/package.json`

**Step 1: Update package.json**

```json
{
  "name": "analytical-chatbot-frontend",
  "version": "2.0.0",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "@a2ui-sdk/react": "^0.8.0",
    "mermaid": "^11.12.2",
    "react": "^19.0.0",
    "react-dom": "^19.0.0"
  },
  "devDependencies": {
    "@types/react": "^19.0.0",
    "@types/react-dom": "^19.0.0",
    "@vitejs/plugin-react": "^4.2.1",
    "typescript": "^5.2.2",
    "vite": "^5.1.1"
  }
}
```

**Step 2: Update vite.config.ts**

```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/a2a': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/upload': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/files': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/database': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/clear': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    }
  }
})
```

**Step 3: Install frontend deps**

Run: `cd frontend && npm install`
Expected: All packages install. `@a2ui-sdk/react` resolves.

**Step 4: Commit**

```bash
git add frontend/package.json frontend/vite.config.ts
git commit -m "chore: update frontend deps - add @a2ui-sdk/react, upgrade to React 19"
```

---

### Task 7: Create custom A2UI catalog components

**Files:**
- Create: `frontend/src/components/PlotViewer.tsx`
- Create: `frontend/src/components/CodeBlock.tsx`
- Create: `frontend/src/components/OutputBlock.tsx`
- Create: `frontend/src/components/DataTable.tsx`
- Create: `frontend/src/components/MermaidChart.tsx`
- Create: `frontend/src/components/HtmlViewer.tsx`
- Create: `frontend/src/catalog.ts`

**Step 1: Create PlotViewer component**

Create `frontend/src/components/PlotViewer.tsx`:

```tsx
import React from 'react'

interface PlotViewerProps {
  data: string  // base64 SVG
}

export function PlotViewer({ data }: PlotViewerProps) {
  if (!data) return null
  return (
    <div className="plots">
      <img
        src={`data:image/svg+xml;base64,${data}`}
        alt="Plot"
        className="plot-image"
      />
    </div>
  )
}
```

**Step 2: Create CodeBlock component**

Create `frontend/src/components/CodeBlock.tsx`:

```tsx
import React, { useState } from 'react'

interface CodeBlockProps {
  code: string
  language: string
}

export function CodeBlock({ code, language }: CodeBlockProps) {
  const [expanded, setExpanded] = useState(false)
  const lineCount = code.split('\n').length

  return (
    <div className={`code-block ${expanded ? 'expanded' : 'collapsed'}`}>
      <div className="code-header" onClick={() => setExpanded(!expanded)}>
        <span className="code-toggle">
          {expanded ? '\u25BC' : '\u25B6'} {language || 'Python'} ({lineCount} lines)
        </span>
        <button onClick={(e) => { e.stopPropagation(); navigator.clipboard.writeText(code) }}>
          Copy
        </button>
      </div>
      {expanded && (
        <pre><code>{code}</code></pre>
      )}
    </div>
  )
}
```

**Step 3: Create OutputBlock component**

Create `frontend/src/components/OutputBlock.tsx`:

```tsx
import React, { useState } from 'react'

interface OutputBlockProps {
  output: string
}

export function OutputBlock({ output }: OutputBlockProps) {
  const [expanded, setExpanded] = useState(false)
  const lineCount = output.split('\n').length

  return (
    <div className={`output-block ${expanded ? 'expanded' : 'collapsed'}`}>
      <div
        className="output-header"
        onClick={() => setExpanded(!expanded)}
        style={{ cursor: 'pointer', userSelect: 'none', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
      >
        <span>{expanded ? '\u25BC' : '\u25B6'} Output ({lineCount} lines)</span>
      </div>
      {expanded && (
        <pre>{output}</pre>
      )}
    </div>
  )
}
```

**Step 4: Create DataTable component**

Create `frontend/src/components/DataTable.tsx`:

```tsx
import React from 'react'

interface DataTableProps {
  html: string
}

export function DataTable({ html }: DataTableProps) {
  if (!html) return null
  return (
    <div className="table-container" dangerouslySetInnerHTML={{ __html: html }} />
  )
}
```

**Step 5: Create MermaidChart component**

Create `frontend/src/components/MermaidChart.tsx`:

```tsx
import React, { useEffect, useState, useRef } from 'react'
import mermaid from 'mermaid'

mermaid.initialize({
  startOnLoad: false,
  theme: 'dark',
  securityLevel: 'loose',
})

interface MermaidChartProps {
  definition: string
}

export function MermaidChart({ definition }: MermaidChartProps) {
  const [svg, setSvg] = useState('')
  const id = useRef(`mermaid-${Math.random().toString(36).substr(2, 9)}`).current

  useEffect(() => {
    let cleaned = definition.trim()
    cleaned = cleaned.replace(/^```mermaid\s*/i, '').replace(/\s*```$/, '').trim()

    setSvg('')
    if (!cleaned) return

    mermaid.render(id, cleaned).then(({ svg }) => {
      setSvg(svg)
    }).catch((error) => {
      console.error('Mermaid error:', error)
      setSvg(`<div style="color: #ff6b6b; padding: 10px;">Mermaid render error</div>`)
    })
  }, [definition, id])

  return (
    <div
      className="mermaid-container"
      style={{ background: 'rgba(0,0,0,0.2)', padding: '10px', borderRadius: '8px', margin: '10px 0', textAlign: 'center', overflowX: 'auto' }}
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  )
}
```

**Step 6: Create HtmlViewer component**

Create `frontend/src/components/HtmlViewer.tsx`:

```tsx
import React from 'react'

interface HtmlViewerProps {
  html: string
}

export function HtmlViewer({ html }: HtmlViewerProps) {
  if (!html) return null
  return (
    <div className="dashboard-container">
      <div className="dashboard-header">Interactive Output</div>
      <iframe
        title="Dashboard"
        srcDoc={html}
        style={{
          width: '100%',
          height: '500px',
          border: 'none',
          backgroundColor: 'white',
          borderRadius: '4px',
        }}
        sandbox="allow-scripts"
      />
    </div>
  )
}
```

**Step 7: Create the custom catalog**

Create `frontend/src/catalog.ts`:

```ts
import { standardCatalog } from '@a2ui-sdk/react'
import { PlotViewer } from './components/PlotViewer'
import { CodeBlock } from './components/CodeBlock'
import { OutputBlock } from './components/OutputBlock'
import { DataTable } from './components/DataTable'
import { MermaidChart } from './components/MermaidChart'
import { HtmlViewer } from './components/HtmlViewer'

export const customCatalog = {
  ...standardCatalog,
  PlotViewer,
  CodeBlock,
  OutputBlock,
  DataTable,
  MermaidChart,
  HtmlViewer,
}
```

**Step 8: Commit**

```bash
git add frontend/src/components/ frontend/src/catalog.ts
git commit -m "feat: add custom A2UI catalog components for plots, code, tables, mermaid, HTML"
```

---

### Task 8: Rewrite the frontend App

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/main.tsx`

**Step 1: Rewrite App.tsx**

Replace `frontend/src/App.tsx` with the A2UI-based implementation:

```tsx
import React, { useState, useRef, useEffect, useCallback } from 'react'
import { A2UIProvider, A2UIRenderer, useA2UIMessageHandler } from '@a2ui-sdk/react'
import { customCatalog } from './catalog'

const API_BASE = (() => {
  if (import.meta.env.DEV) return ''  // Vite proxy handles routing
  return window.location.origin
})()

interface FileInfo {
  filename: string
  rows: number
  columns: number
}

interface TableInfo {
  columns: Record<string, string>
  row_count: number
}

interface DatabaseInfo {
  available: boolean
  tables: Record<string, TableInfo>
}

function ChatApp() {
  const [files, setFiles] = useState<FileInfo[]>([])
  const [database, setDatabase] = useState<DatabaseInfo | null>(null)
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [contextId, setContextId] = useState<string>('')
  const [chatHistory, setChatHistory] = useState<Array<{role: string, content: string}>>([])
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const { pushMessage, messages: a2uiMessages } = useA2UIMessageHandler()

  useEffect(() => {
    fetchDatabase()
  }, [])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [chatHistory, a2uiMessages])

  async function fetchDatabase() {
    try {
      const res = await fetch(`${API_BASE}/database`)
      const data = await res.json()
      setDatabase(data)
    } catch (e) {
      console.error('Failed to fetch database info:', e)
    }
  }

  async function handleUpload(e: React.ChangeEvent<HTMLInputElement>) {
    if (!e.target.files?.length) return
    const file = e.target.files[0]
    const form = new FormData()
    form.append('file', file)

    try {
      const res = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: form,
        headers: contextId ? { 'X-Context-Id': contextId } : {},
      })
      if (res.ok) {
        const data = await res.json()
        setFiles(f => [...f, { filename: data.filename, rows: data.rows, columns: data.columns }])
        if (data.contextId) setContextId(data.contextId)
      } else {
        const err = await res.json()
        alert('Upload error: ' + (err.detail || res.statusText))
      }
    } catch (err) {
      alert('Upload failed: ' + err)
    }
    e.target.value = ''
  }

  const sendMessage = useCallback(async () => {
    if (!input.trim() || loading) return

    const userText = input
    setChatHistory(h => [...h, { role: 'user', content: userText }])
    setInput('')
    setLoading(true)

    try {
      const response = await fetch(`${API_BASE}/a2a/message/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: {
            messageId: crypto.randomUUID(),
            role: 'user',
            parts: [{ text: userText }],
            contextId: contextId || undefined,
          },
        }),
      })

      // Read context ID from response header
      const respContextId = response.headers.get('X-Context-Id')
      if (respContextId && !contextId) {
        setContextId(respContextId)
      }

      // Process SSE stream
      const reader = response.body?.getReader()
      const decoder = new TextDecoder()

      if (reader) {
        let buffer = ''
        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split('\n')
          buffer = lines.pop() || ''

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const eventData = JSON.parse(line.slice(6))

                // Check for A2UI DataParts
                if (eventData.parts) {
                  for (const part of eventData.parts) {
                    const data = part?.root?.data
                    const mimeType = part?.root?.metadata?.mimeType
                    if (data && mimeType === 'application/json+a2ui') {
                      pushMessage(data)
                    }
                  }
                }

                // Check for completion
                if (eventData.status?.state === 'completed') {
                  if (eventData.contextId) setContextId(eventData.contextId)
                }
              } catch {
                // Skip malformed lines
              }
            }
          }
        }
      }

      setChatHistory(h => [...h, { role: 'assistant', content: '(rendered via A2UI)' }])
      fetchDatabase()
    } catch (e) {
      console.error('Message send error:', e)
      setChatHistory(h => [...h, { role: 'assistant', content: `Error: ${e}` }])
    } finally {
      setLoading(false)
    }
  }, [input, loading, contextId, pushMessage])

  async function clearSession() {
    await fetch(`${API_BASE}/clear`, {
      method: 'POST',
      headers: contextId ? { 'X-Context-Id': contextId } : {},
    })
    setFiles([])
    setChatHistory([])
    setContextId('')
  }

  function handleAction(action: { name: string; context?: Record<string, unknown> }) {
    console.log('A2UI action:', action)
    // Handle user actions from A2UI components if needed
  }

  return (
    <div className="app">
      <header>
        <h1>Analytical Chatbot</h1>
        <p className="subtitle">Upload data, ask questions, get insights</p>
      </header>

      <section className="files-section">
        <div className="files-header">
          <h2>Data Files</h2>
          <label className="upload-btn">
            + Upload
            <input type="file" accept=".csv,.json" onChange={handleUpload} hidden />
          </label>
        </div>
        {files.length === 0 ? (
          <p className="no-files">No files uploaded. Upload a CSV or JSON file to analyze.</p>
        ) : (
          <div className="file-chips">
            {files.map(f => (
              <div className="file-chip" key={f.filename}>
                <span>{f.filename}</span>
                <span className="file-info">{f.rows} rows x {f.columns} cols</span>
              </div>
            ))}
          </div>
        )}
      </section>

      {database?.available && (
        <section className="database-section">
          <div className="database-header">
            <h2>Database Tables</h2>
          </div>
          <div className="table-chips">
            {Object.entries(database.tables).map(([name, info]) => (
              <div className="table-chip" key={name}>
                <span className="table-name">{name}</span>
                <span className="table-info">{info.row_count} rows</span>
              </div>
            ))}
          </div>
        </section>
      )}

      <section className="chat-section">
        <div className="messages">
          {chatHistory.length === 0 && (
            <div className="welcome">
              <p>Welcome! I can help you analyze data. Try:</p>
              <ul>
                <li>"Show employees with salary over 90k"</li>
                <li>"What are the total sales by region?"</li>
                <li>"Create a bar chart of products by category"</li>
                <li>"Join sales with products and show top sellers"</li>
              </ul>
            </div>
          )}

          {chatHistory.map((m, i) => (
            <div className={`msg ${m.role === 'user' ? 'user' : 'assistant'}`} key={i}>
              <div className="msg-content">
                {m.role === 'user' ? m.content : null}
              </div>
            </div>
          ))}

          {/* A2UI rendered content */}
          <A2UIRenderer onAction={handleAction} />

          {loading && (
            <div className="msg assistant">
              <div className="msg-content loading">
                <span className="dot"></span>
                <span className="dot"></span>
                <span className="dot"></span>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="composer">
          <input
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && !e.shiftKey && sendMessage()}
            placeholder="Ask a question or request analysis..."
            disabled={loading}
          />
          <button onClick={sendMessage} disabled={loading || !input.trim()}>
            Send
          </button>
        </div>

        <div className="actions">
          <button className="clear-btn" onClick={clearSession}>Clear Session</button>
        </div>
      </section>
    </div>
  )
}

export default function App() {
  return (
    <A2UIProvider catalog={customCatalog}>
      <ChatApp />
    </A2UIProvider>
  )
}
```

**Step 2: Update main.tsx**

Replace `frontend/src/main.tsx`:

```tsx
import React from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import './style.css'

createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)
```

(This stays the same but confirms React 19 compatibility.)

**Step 3: Verify frontend builds**

Run: `cd frontend && npm run build`
Expected: TypeScript compiles and Vite builds without errors.

**Step 4: Commit**

```bash
git add frontend/src/App.tsx frontend/src/main.tsx
git commit -m "feat: rewrite frontend with A2UI renderer and SSE transport"
```

---

### Task 9: Delete old files

**Files:**
- Delete: `rest_server.py`
- Delete: `a2a_server.py`
- Delete: `agent_executor.py`
- Delete: `main.py`
- Delete: `utils/flow.py`
- Delete: `utils/nodes.py`

**Step 1: Remove old files**

```bash
git rm rest_server.py a2a_server.py agent_executor.py main.py utils/flow.py utils/nodes.py
```

**Step 2: Update launch scripts**

Modify `launch_backend.sh` to point to the new server:

```bash
#!/bin/bash
python server.py
```

Modify `launch_frontend.sh` (keep as-is, it just runs `cd frontend && npm run dev`).

**Step 3: Commit**

```bash
git add launch_backend.sh
git commit -m "refactor: remove old BFF, PocketFlow, and A2A server files; update launcher"
```

---

### Task 10: End-to-end integration test

**Step 1: Start the backend**

Run: `python server.py`
Expected: Server starts, shows A2UI server banner, listens on port 8000.

**Step 2: Verify agent card**

Run: `curl http://localhost:8000/.well-known/agent-card.json | python -m json.tool`
Expected: Agent card JSON with A2UI extension capability listed.

**Step 3: Test message/send**

```bash
curl -X POST http://localhost:8000/a2a/message/send \
  -H "Content-Type: application/json" \
  -d '{"message": {"messageId": "test-1", "role": "user", "parts": [{"text": "Hello, what can you do?"}]}}'
```

Expected: JSON response with A2UI surfaceUpdate/beginRendering messages containing the agent's response.

**Step 4: Test message/stream**

```bash
curl -N -X POST http://localhost:8000/a2a/message/stream \
  -H "Content-Type: application/json" \
  -d '{"message": {"messageId": "test-2", "role": "user", "parts": [{"text": "Show employees with salary over 90000"}]}}'
```

Expected: SSE stream with A2UI messages, ending with a completion event.

**Step 5: Start the frontend**

Run (new terminal): `cd frontend && npm run dev`
Expected: Vite dev server starts on port 5173.

**Step 6: Test in browser**

Open `http://localhost:5173`. Verify:
- Header and database tables section renders
- File upload button works
- Sending a message streams A2UI response
- A2UIRenderer renders the agent's text
- Sending "Show employees with salary over 90k" triggers tool use and renders results

**Step 7: Commit final state**

```bash
git add -A
git commit -m "feat: complete A2UI refactor - merged frontend and BFF into single A2UI layer"
```

---

### Task 11: Update README and cleanup

**Files:**
- Modify: `README.md`

**Step 1: Update README**

Update the README to reflect the new architecture:
- Remove references to BFF, PocketFlow, gRPC
- Document the A2UI + A2A + DeepAgents architecture
- Update startup instructions (`python server.py` instead of `python main.py`)
- Update environment variable documentation
- Note the new dependencies

**Step 2: Clean up diagnostic/test files**

Remove files that reference the old architecture if they exist:
- `diagnose_servers.py` (references old multi-server setup)
- `test_chat.py` (references old REST API)

**Step 3: Commit**

```bash
git add README.md
git rm diagnose_servers.py test_chat.py 2>/dev/null || true
git commit -m "docs: update README for A2UI architecture"
```
