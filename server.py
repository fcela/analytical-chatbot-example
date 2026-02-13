"""A2UI-enabled A2A server for the analytical chatbot."""

import asyncio
import io
import json
import logging
import os
import queue
import re
import uuid
from typing import Any

import polars as pl
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

from agent import create_agent
from a2ui_adapter import create_chat_surface
from tools._runtime import set_kernel
from utils.sandbox_factory import create_sandbox

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

agent = create_agent()
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
    if context_id not in sessions:
        kernel = create_sandbox()
        logger.info(f"Created session {context_id} with {kernel.backend_name} backend")
        sessions[context_id] = {
            "kernel": kernel,
            "history": [],
            "files": [],
        }
    return sessions[context_id]


def extract_user_text(message: dict) -> str:
    """Extract user text from an A2A message's parts."""
    for part in message.get("parts", []):
        if "text" in part:
            return part["text"]
        if "root" in part and "text" in part["root"]:
            return part["root"]["text"]
    return ""


def extract_agent_result(result_messages: list) -> dict[str, Any]:
    """Extract structured data from LangGraph agent result messages."""
    assistant_msg = ""
    code = None
    output = None
    artifacts: dict[str, Any] = {}
    error = None

    for msg in result_messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.get("name") == "execute_python":
                    code = tc.get("args", {}).get("code")

        if hasattr(msg, "content") and hasattr(msg, "type"):
            if msg.type == "ai":
                if isinstance(msg.content, str) and msg.content.strip():
                    assistant_msg = msg.content
            elif msg.type == "tool":
                try:
                    tool_result = json.loads(msg.content) if isinstance(msg.content, str) else {}
                    if isinstance(tool_result, dict):
                        if tool_result.get("success"):
                            output = tool_result.get("stdout", "")
                            artifacts.update(tool_result.get("artifacts", {}))
                            error = None
                        elif tool_result.get("success") is False:
                            error = tool_result.get("error")
                except (json.JSONDecodeError, TypeError):
                    pass

    # Strip inline image references — plots render via A2UI PlotViewer
    assistant_msg = re.sub(
        r"!\[.*?\]\((attachment://[^)]+|data:image/[^)]+)\)\s*", "", assistant_msg
    ).strip()

    return {
        "text": assistant_msg,
        "code": code,
        "output": output,
        "artifacts": artifacts,
        "error": error,
    }


@app.get("/")
def health():
    return {"status": "running", "protocol": "a2ui+a2a"}


@app.get("/.well-known/agent-card.json")
def agent_card():
    host = os.getenv("HOST", "localhost")
    port = int(os.getenv("PORT", "8000"))
    return {
        "name": "Analytical Chatbot",
        "description": "AI-powered data analysis assistant with code execution and visualization.",
        "url": f"http://{host}:{port}",
        "version": "2.0.0",
        "capabilities": {
            "streaming": True,
            "extensions": [{
                "uri": "https://a2ui.org/a2a-extension/a2ui/v0.8",
                "params": {
                    "supportedCatalogIds": [
                        "https://a2ui.org/specification/v0_8/standard_catalog_definition.json",
                        "analytical-chatbot-custom",
                    ],
                    "acceptsInlineCatalogs": False,
                },
            }],
        },
        "skills": [
            {"id": "data-analysis", "name": "Data Analysis", "description": "Analyze data files and databases"},
            {"id": "visualization", "name": "Visualization", "description": "Create charts with Altair"},
            {"id": "database-query", "name": "Database Query", "description": "Query DuckDB tables"},
        ],
    }


TOOL_STATUS = {
    "get_database_schema": "Inspecting database schema...",
    "query_database": "Running SQL query...",
    "execute_python": "Running code in sandbox...",
}


async def _invoke_agent(session: dict[str, Any], user_text: str) -> dict[str, Any]:
    """Run the agent synchronously and return extracted results (for message/send)."""
    input_messages = list(session["history"])
    input_messages.append({"role": "user", "content": user_text})
    kernel = session["kernel"]

    def _run():
        set_kernel(kernel)
        try:
            return agent.invoke({"messages": input_messages})
        finally:
            set_kernel(None)

    result = await asyncio.get_event_loop().run_in_executor(None, _run)
    extracted = extract_agent_result(result.get("messages", []))

    session["history"].append({"role": "user", "content": user_text})
    session["history"].append({"role": "assistant", "content": extracted["text"]})

    return extracted


def _stream_agent_to_queue(
    q: queue.Queue,
    session: dict[str, Any],
    user_text: str,
) -> None:
    """Run agent.stream() in a worker thread and push events to a queue."""
    input_messages = list(session["history"])
    input_messages.append({"role": "user", "content": user_text})
    kernel = session["kernel"]

    set_kernel(kernel)
    try:
        all_messages: list = []
        for chunk in agent.stream({"messages": input_messages}):
            # LangGraph stream chunks: {"node_name": {"messages": [...]}}
            for node_name, state_update in chunk.items():
                msgs = state_update.get("messages", [])
                for msg in msgs:
                    all_messages.append(msg)

                    # Detect tool calls -> progress
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            name = tc.get("name", "")
                            status = TOOL_STATUS.get(name, f"Using {name}...")
                            q.put(("progress", status))

                    # Detect tool result -> progress
                    if hasattr(msg, "type") and msg.type == "tool":
                        tool_name = getattr(msg, "name", "")
                        if tool_name == "execute_python":
                            q.put(("progress", "Processing results..."))
                        elif tool_name == "get_database_schema":
                            q.put(("progress", "Analyzing schema..."))

                    # Detect AI response text being generated
                    if hasattr(msg, "type") and msg.type == "ai":
                        if isinstance(getattr(msg, "content", ""), str) and msg.content.strip():
                            if not (hasattr(msg, "tool_calls") and msg.tool_calls):
                                q.put(("progress", "Composing response..."))

        q.put(("done", all_messages))
    except Exception as e:
        q.put(("error", e))
    finally:
        set_kernel(None)


@app.post("/a2a/message/stream")
async def message_stream(request: Request):
    body = await request.json()
    message = body.get("message", {})
    context_id = message.get("contextId") or message.get("context_id") or str(uuid.uuid4())

    user_text = extract_user_text(message)
    if not user_text:
        raise HTTPException(status_code=400, detail="No text content in message")

    session = get_or_create_session(context_id)

    async def event_stream():
        try:
            q: queue.Queue = queue.Queue()
            loop = asyncio.get_event_loop()

            # Start streaming in background thread
            loop.run_in_executor(None, _stream_agent_to_queue, q, session, user_text)

            # Yield progress as SSE events until done
            yield f"data: {json.dumps({'progress': 'Thinking...'})}\n\n"

            all_messages = None
            while True:
                try:
                    kind, payload = await asyncio.wait_for(
                        loop.run_in_executor(None, q.get, True, 30.0),
                        timeout=60.0,
                    )
                except (asyncio.TimeoutError, queue.Empty):
                    yield f"data: {json.dumps({'progress': 'Still working...'})}\n\n"
                    continue

                if kind == "progress":
                    yield f"data: {json.dumps({'progress': payload})}\n\n"
                elif kind == "done":
                    all_messages = payload
                    break
                elif kind == "error":
                    raise payload

            extracted = extract_agent_result(all_messages or [])
            session["history"].append({"role": "user", "content": user_text})
            session["history"].append({"role": "assistant", "content": extracted["text"]})

            a2ui_messages = create_chat_surface(
                message_text=extracted["text"],
                code=extracted["code"],
                output=extracted["output"],
                artifacts=extracted["artifacts"],
                error=extracted["error"],
            )

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

            yield f"data: {json.dumps({'status': {'state': 'completed'}, 'contextId': context_id})}\n\n"

        except Exception as e:
            logger.exception("Agent execution error")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

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
    body = await request.json()
    message = body.get("message", {})
    context_id = message.get("contextId") or message.get("context_id") or str(uuid.uuid4())

    user_text = extract_user_text(message)
    if not user_text:
        raise HTTPException(status_code=400, detail="No text content in message")

    session = get_or_create_session(context_id)
    extracted = await _invoke_agent(session, user_text)

    a2ui_messages = create_chat_surface(
        message_text=extracted["text"],
        code=extracted["code"],
        output=extracted["output"],
        artifacts=extracted["artifacts"],
        error=extracted["error"],
    )

    return {
        "contextId": context_id,
        "status": {"state": "completed"},
        "parts": [
            {"root": {"data": msg, "metadata": {"mimeType": "application/json+a2ui"}}}
            for msg in a2ui_messages
        ],
    }


@app.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
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
    context_id = request.headers.get("X-Context-Id") or request.cookies.get("context_id")
    if not context_id or context_id not in sessions:
        return {"files": []}
    return {"files": sessions[context_id]["files"]}


@app.get("/database")
def database_info():
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
