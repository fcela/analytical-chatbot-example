"""A2UI-enabled A2A server for the analytical chatbot."""

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


@app.post("/a2a/message/stream")
async def message_stream(request: Request):
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

    async def event_stream():
        try:
            input_messages = list(session["history"])
            input_messages.append({"role": "user", "content": user_text})

            kernel = session["kernel"]

            def _run_agent():
                set_kernel(kernel)
                try:
                    return agent.invoke({"messages": input_messages})
                finally:
                    set_kernel(None)

            result = await asyncio.get_event_loop().run_in_executor(
                None, _run_agent
            )

            result_messages = result.get("messages", [])
            assistant_msg = ""
            code = None
            output = None
            artifacts = {}
            error = None

            for msg in result_messages:
                if hasattr(msg, "content") and hasattr(msg, "type"):
                    if msg.type == "ai":
                        assistant_msg = msg.content if isinstance(msg.content, str) else ""
                    elif msg.type == "tool":
                        try:
                            tool_result = json.loads(msg.content) if isinstance(msg.content, str) else {}
                            if isinstance(tool_result, dict):
                                if tool_result.get("success") is not None:
                                    if tool_result.get("success"):
                                        output = tool_result.get("stdout", "")
                                        artifacts.update(tool_result.get("artifacts", {}))
                                    else:
                                        error = tool_result.get("error")
                        except (json.JSONDecodeError, TypeError):
                            pass

            session["history"].append({"role": "user", "content": user_text})
            session["history"].append({"role": "assistant", "content": assistant_msg})

            a2ui_messages = create_chat_surface(
                message_text=assistant_msg,
                code=code,
                output=output,
                artifacts=artifacts,
                error=error,
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

    user_text = ""
    for part in message.get("parts", []):
        if "text" in part:
            user_text = part["text"]
        elif "root" in part and "text" in part["root"]:
            user_text = part["root"]["text"]

    if not user_text:
        raise HTTPException(status_code=400, detail="No text content in message")

    session = get_or_create_session(context_id)

    input_messages = list(session["history"])
    input_messages.append({"role": "user", "content": user_text})

    kernel = session["kernel"]

    def _run_agent():
        set_kernel(kernel)
        try:
            return agent.invoke({"messages": input_messages})
        finally:
            set_kernel(None)

    result = await asyncio.get_event_loop().run_in_executor(
        None, _run_agent
    )

    result_messages = result.get("messages", [])
    assistant_msg = ""
    artifacts = {}

    for msg in result_messages:
        if hasattr(msg, "content") and hasattr(msg, "type"):
            if msg.type == "ai":
                assistant_msg = msg.content if isinstance(msg.content, str) else ""
            elif msg.type == "tool":
                try:
                    tool_result = json.loads(msg.content) if isinstance(msg.content, str) else {}
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
