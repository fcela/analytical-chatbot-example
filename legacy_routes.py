"""Legacy REST API routes that translate to A2A protocol.

This module provides backwards-compatible REST endpoints for the frontend
while routing all requests through the A2A agent executor.
"""

import io
import json
import uuid
from typing import Any

from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Response
from pydantic import BaseModel

from a2a.types import (
    Message,
    Part,
    TextPart,
    DataPart,
    TaskState,
)
from a2a.server.request_handlers import DefaultRequestHandler

# MessageRole and SendMessageRequest may be in different locations depending on SDK version
try:
    from a2a.types import MessageRole
except ImportError:
    from enum import Enum
    class MessageRole(str, Enum):
        user = "user"
        agent = "agent"

try:
    from a2a.types import SendMessageRequest, MessageSendParams
except ImportError:
    # Create a simple wrapper if not available
    from pydantic import BaseModel as PydanticBaseModel
    class SendMessageRequest(PydanticBaseModel):
        message: Message

    class MessageSendParams(PydanticBaseModel):
        message: Message

# Try to import polars for file handling
try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    pl = None
    HAS_POLARS = False

# Try to import database utilities
try:
    from utils.database import get_connection, get_table_info, HAS_DUCKDB
except ImportError:
    HAS_DUCKDB = False
    def get_table_info():
        return {}

from utils.call_llm import get_provider_info


# --- Data Models ---

class ChatRequest(BaseModel):
    message: str


class FileInfo(BaseModel):
    filename: str
    rows: int
    columns: int


class FileListResponse(BaseModel):
    files: list[FileInfo]


# --- Session Management ---

# In-memory store for session data (Files + History)
# Structure: { session_id: { "files": {name: df}, "context_id": str } }
session_store: dict[str, dict[str, Any]] = {}


def get_session_id(request: Request, response: Response) -> str:
    """Retrieve or create a session ID via cookie."""
    session_id = request.cookies.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        response.set_cookie(key="session_id", value=session_id, httponly=True)

    if session_id not in session_store:
        session_store[session_id] = {
            "files": {},
            "context_id": str(uuid.uuid4()),  # A2A context for multi-turn
        }

    return session_id


def create_legacy_router(request_handler: DefaultRequestHandler) -> APIRouter:
    """Create a FastAPI router with legacy REST endpoints.

    Args:
        request_handler: The A2A DefaultRequestHandler to route requests through.

    Returns:
        A FastAPI APIRouter with legacy endpoints.
    """
    router = APIRouter()

    @router.get("/")
    def health_check():
        return {"status": "running", "backend": "A2A", "database": HAS_DUCKDB}

    @router.get("/llm")
    def llm_info():
        """Get information about the current LLM provider configuration."""
        return get_provider_info()

    @router.get("/database")
    def database_info():
        """Get information about the available database tables."""
        if not HAS_DUCKDB:
            return {"available": False, "tables": {}, "saved_tables": []}

        get_connection()
        all_tables = get_table_info()
        builtin = {k: v for k, v in all_tables.items() if not k.startswith("saved_")}
        saved = {k: v for k, v in all_tables.items() if k.startswith("saved_")}

        return {
            "available": True,
            "tables": builtin,
            "saved_tables": saved
        }

    @router.post("/chat")
    async def chat_endpoint(req: ChatRequest, request: Request, response: Response):
        """Chat endpoint that routes through A2A protocol."""
        session_id = get_session_id(request, response)
        session_data = session_store[session_id]

        try:
            # Store files in the agent executor's session
            # We need to pass file info through the A2A message
            file_info = {}
            for fname, df in session_data["files"].items():
                if hasattr(df, 'schema'):  # Polars
                    file_info[fname] = {
                        "columns": df.columns,
                        "schema": str(df.schema),
                        "shape": [df.height, df.width],
                        "sample": df.head(2).to_dicts()
                    }
                elif hasattr(df, 'columns'):  # Pandas
                    file_info[fname] = {
                        "columns": list(df.columns),
                        "dtypes": str(df.dtypes.to_dict()),
                        "shape": list(df.shape),
                        "sample": df.head(2).to_dict()
                    }

            # Create A2A message with file context embedded
            message_text = req.message

            # Create the A2A message
            parts = [Part(root=TextPart(text=message_text))]

            # Add file data as a DataPart if files exist
            if file_info:
                parts.append(Part(root=DataPart(
                    data={"uploaded_files": file_info},
                    mimeType="application/json"
                )))

            a2a_message = Message(
                messageId=str(uuid.uuid4()),
                role=MessageRole.user,
                parts=parts,
                contextId=session_data["context_id"],
            )

            # Create SendMessageRequest
            try:
                # Try JSON-RPC structure first
                send_request = SendMessageRequest(
                    id=str(uuid.uuid4()),
                    params=MessageSendParams(message=a2a_message)
                )
            except Exception:
                # Fallback to simple structure
                send_request = SendMessageRequest(message=a2a_message)

            # We need to inject the actual DataFrames into the executor
            # Get the executor from the handler and set the files
            executor = request_handler.agent_executor
            if hasattr(executor, '_sessions'):
                if session_data["context_id"] not in executor._sessions:
                    executor._sessions[session_data["context_id"]] = {
                        "files": {},
                        "history": []
                    }
                executor._sessions[session_data["context_id"]]["files"] = session_data["files"]

            # Call the A2A handler
            # In the current version of the SDK, on_message_send expects MessageSendParams, not SendMessageRequest
            # Extract params from send_request or use MessageSendParams directly
            params = getattr(send_request, "params", send_request)
            if not isinstance(params, MessageSendParams) and hasattr(send_request, "message"):
                # Fallback for dummy SendMessageRequest
                params = MessageSendParams(message=send_request.message)

            result = await request_handler.on_message_send(
                params=params,
                context=None,  # No auth context needed for local calls
            )

            # Extract response from A2A result
            if hasattr(result, 'error') and result.error:
                raise HTTPException(status_code=500, detail=str(result.error))

            # Parse the A2A response back to legacy format
            response_data = {
                "message": "",
                "code": None,
                "output": None,
                "plots": [],
                "tables": [],
                "html": [],
                "error": None
            }

            # Handle Task response (result is the Task/Message directly)
            task_or_message = result
            
            print(f"DEBUG: Result type: {type(task_or_message)}")
            print(f"DEBUG: Result dir: {dir(task_or_message)}")
            
            # Refetch task from store to ensure we have the latest artifacts
            # The result might be a Message (which has task_id but no artifacts) or a Task
            task_id = getattr(task_or_message, 'id', None) # If Task
            
            if not task_id:
                task_id = getattr(task_or_message, 'task_id', None) # If Message
                # Also check camelCase alias just in case
                if not task_id:
                    task_id = getattr(task_or_message, 'taskId', None)

            if task_id and hasattr(request_handler, 'task_store'):
                try:
                    latest_task = await request_handler.task_store.get(task_id)
                    if latest_task:
                        task_or_message = latest_task
                except Exception as e:
                    print(f"Warning: Failed to refetch task: {e}")

            # If it's a Task, extract from history and artifacts
            if hasattr(task_or_message, 'history'):
                for msg in task_or_message.history or []:
                    # Get role as string for comparison
                    role = msg.role
                    if hasattr(role, 'value'):
                        role = role.value
                    
                    if role == "agent":
                        for part in msg.parts or []:
                            part_data = part.root if hasattr(part, 'root') else part
                            if hasattr(part_data, 'text'):
                                response_data["message"] = part_data.text

            # Extract artifacts
            if hasattr(task_or_message, 'artifacts'):
                for artifact in task_or_message.artifacts or []:
                    name = getattr(artifact, "name", "")
                    for part in artifact.parts or []:
                        part_data = part.root if hasattr(part, 'root') else part

                        if name == "generated_code" and hasattr(part_data, 'text'):
                            response_data["code"] = part_data.text
                        elif name == "execution_output" and hasattr(part_data, 'text'):
                            response_data["output"] = part_data.text
                        elif name and name.startswith("plot_") and hasattr(part_data, 'data'):
                            response_data["plots"].append(part_data.data.get("image_base64", ""))
                        elif name and name.startswith("table_") and hasattr(part_data, 'data'):
                            # Try to find table data in the dict
                            table_val = part_data.data.get("html", part_data.data) if hasattr(part_data, 'data') else None
                            if table_val:
                                response_data["tables"].append(table_val)
                        elif name and name.startswith("html_") and hasattr(part_data, 'text'):
                            response_data["html"].append(part_data.text)
                        elif name == "error" and hasattr(part_data, 'text'):
                            response_data["error"] = part_data.text

            # Check task status for errors
            if hasattr(task_or_message, 'status'):
                status = task_or_message.status
                if hasattr(status, 'state') and status.state == TaskState.failed:
                    if hasattr(status, 'message'):
                        response_data["error"] = status.message.text if hasattr(status.message, 'text') else str(status.message)

            # If it's a direct Message response
            elif hasattr(task_or_message, 'parts'):
                for part in task_or_message.parts or []:
                    if hasattr(part, 'root') and isinstance(part.root, TextPart):
                        response_data["message"] = part.root.text
                    elif hasattr(part, 'text'):
                        response_data["message"] = part.text

            return {"response": response_data}

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/upload")
    async def upload_file(
        file: UploadFile = File(...),
        request: Request = None,
        response: Response = None
    ):
        """Upload a file (stored in session, available to A2A agent)."""
        if not HAS_POLARS:
            raise HTTPException(status_code=500, detail="Polars not installed")

        session_id = get_session_id(request, response)
        session_data = session_store[session_id]
        filename = file.filename

        if not filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        try:
            content = await file.read()

            if not content:
                raise HTTPException(status_code=400, detail="Empty file")

            if filename.endswith('.csv'):
                try:
                    df = pl.read_csv(io.BytesIO(content))
                except Exception:
                    df = pl.read_csv(io.BytesIO(content), encoding='latin-1')
            elif filename.endswith('.json'):
                json_data = json.loads(content.decode('utf-8'))
                if isinstance(json_data, list):
                    df = pl.DataFrame(json_data)
                elif isinstance(json_data, dict):
                    if all(isinstance(v, list) for v in json_data.values()):
                        df = pl.DataFrame(json_data)
                    else:
                        df = pl.DataFrame([json_data])
                else:
                    raise ValueError("JSON must be an array of records or an object")
            else:
                raise HTTPException(status_code=400, detail="Unsupported file type. Use .csv or .json")

            session_data["files"][filename] = df

            return {
                "filename": filename,
                "rows": df.height,
                "columns": df.width,
                "column_names": df.columns
            }
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error parsing file: {str(e)}")

    @router.get("/files", response_model=FileListResponse)
    def list_files(request: Request, response: Response):
        """List uploaded files in the session."""
        session_id = get_session_id(request, response)
        session_data = session_store[session_id]
        files_info = []

        for filename, df in session_data["files"].items():
            files_info.append({
                "filename": filename,
                "rows": df.height if hasattr(df, 'height') else len(df),
                "columns": df.width if hasattr(df, 'width') else len(df.columns)
            })

        return {"files": files_info}

    @router.delete("/upload/{filename}")
    def delete_file(filename: str, request: Request, response: Response):
        """Delete an uploaded file from the session."""
        session_id = get_session_id(request, response)
        session_data = session_store[session_id]
        if filename in session_data["files"]:
            del session_data["files"][filename]
        return {"status": "ok"}

    @router.post("/clear")
    def clear_session(request: Request, response: Response):
        """Clear the session (files and conversation context)."""
        session_id = get_session_id(request, response)

        # Clear local session
        if session_id in session_store:
            old_context_id = session_store[session_id].get("context_id")
            session_store[session_id] = {
                "files": {},
                "context_id": str(uuid.uuid4()),  # New context for fresh conversation
            }

            # Also clear the executor's session
            executor = request_handler.agent_executor
            if hasattr(executor, '_sessions') and old_context_id in executor._sessions:
                del executor._sessions[old_context_id]

        return {"status": "ok"}

    return router
