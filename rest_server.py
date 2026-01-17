"""
REST API Server (BFF) for the Analytical Chatbot.
Routes user requests to the A2A Backend via gRPC.
"""

import os
import uuid
import base64
import io
import json
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import polars as pl
import grpc.aio

# A2A SDK
from a2a.client.legacy_grpc import A2AGrpcClient
from a2a.grpc import a2a_pb2_grpc
from a2a.client import minimal_agent_card
from a2a.types import (
    Message,
    Part,
    TextPart,
    DataPart,
    MessageSendParams,
    TaskState,
)

try:
    from a2a.types import MessageRole
except ImportError:
    from enum import Enum
    class MessageRole(str, Enum):
        user = "user"
        agent = "agent"

# --- Configuration ---
BACKEND_HOST = os.getenv("BACKEND_HOST", "localhost")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "50051"))

# --- App Setup ---
app = FastAPI(title="Analytical Chatbot REST API")

@app.on_event("startup")
async def startup_event():
    print("REST API Server starting up...", flush=True)

@app.get("/")
def health_check():
    return {"status": "running", "backend": "BFF", "grpc_target": f"{BACKEND_HOST}:{BACKEND_PORT}"}

@app.get("/llm")
def llm_info():
    try:
        from utils.call_llm import get_provider_info
        return get_provider_info()
    except ImportError:
        return {"provider": "unknown"}

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex="https?://(localhost|127\.0\.0\.1)(:[0-9]+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Client Management ---
_client: Optional[A2AGrpcClient] = None

async def get_client():
    global _client
    if _client is None:
        print(f"Connecting to A2A Backend at {BACKEND_HOST}:{BACKEND_PORT}...")
        channel = grpc.aio.insecure_channel(f"{BACKEND_HOST}:{BACKEND_PORT}")
        stub = a2a_pb2_grpc.A2AServiceStub(channel)
        card = minimal_agent_card(url=f"grpc://{BACKEND_HOST}:{BACKEND_PORT}")
        _client = A2AGrpcClient(grpc_stub=stub, agent_card=card)
        # Manually initialize extensions to avoid AttributeError in _get_grpc_metadata
        if not hasattr(_client, 'extensions'):
            _client.extensions = None
    return _client

# --- Session Management ---
session_store: Dict[str, Dict[str, Any]] = {}

def get_session_id(request: Request, response: Response) -> str:
    session_id = request.cookies.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        response.set_cookie(key="session_id", value=session_id, httponly=True)

    if session_id not in session_store:
        session_store[session_id] = {
            "context_id": str(uuid.uuid4()),
            "files": [] 
        }
    return session_id

# --- Models ---
class ChatRequest(BaseModel):
    message: str

class FileInfo(BaseModel):
    filename: str
    rows: int
    columns: int

class FileListResponse(BaseModel):
    files: List[FileInfo]

# --- Routes ---

@app.post("/chat")
async def chat_endpoint(
    req: ChatRequest, 
    session_id: str = Depends(get_session_id),
    client: A2AGrpcClient = Depends(get_client)
):
    session = session_store[session_id]
    context_id = session["context_id"]

    try:
        msg = Message(
            message_id=str(uuid.uuid4()),
            role=MessageRole.user,
            parts=[Part(root=TextPart(text=req.message))],
            context_id=context_id
        )
        
        print(f"BFF DEBUG: Sending streaming request for context {context_id}...", flush=True)
        
        response_data = {
            "message": "",
            "code": None,
            "output": None,
            "artifacts": {}, # Map ID -> {type, content}
            "error": None
        }
        
        # Use streaming to handle events as they come
        try:
            async for event in client.send_message_streaming(MessageSendParams(message=msg)):
                
                # 1. Handle Text Message
                if hasattr(event, 'role') and hasattr(event, 'parts'):
                    role = event.role
                    if hasattr(role, 'value'): role = role.value
                    if role == "agent":
                        for part in event.parts or []:
                            part_data = part.root if hasattr(part, 'root') else part
                            if hasattr(part_data, 'text'):
                                response_data["message"] = part_data.text
                                
                # 2. Handle Artifacts
                artifact = getattr(event, 'artifact', None)
                if artifact:
                    name = getattr(artifact, "name", "")
                    # Extract content
                    content = None
                    art_type = "unknown"
                    
                    for part in artifact.parts or []:
                        part_data = part.root if hasattr(part, 'root') else part
                        
                        if hasattr(part_data, 'text'):
                            content = part_data.text
                            # Infer type from name if possible, or context
                            if name == "generated_code":
                                response_data["code"] = content
                                art_type = "code"
                            elif name == "execution_output":
                                response_data["output"] = content
                                art_type = "output"
                            elif name == "error":
                                response_data["error"] = content
                                art_type = "error"
                            elif name.startswith("html_"):
                                art_type = "html"
                            elif name.startswith("mermaid_"):
                                art_type = "mermaid"
                            elif name.startswith("markdown_") or name.startswith("md_"):
                                art_type = "markdown"
                            
                        elif hasattr(part_data, 'data'):
                            if name.startswith("plot_"):
                                content = part_data.data.get("image_base64")
                                art_type = "svg" # or image
                            elif name.startswith("table_"):
                                content = part_data.data.get("html", part_data.data)
                                art_type = "table"
                    
                    if content and name not in ["generated_code", "execution_output", "error"]:
                        response_data["artifacts"][name] = {
                            "type": art_type,
                            "content": content
                        }
                            
                # 3. Handle Task object (final result)
                if hasattr(event, 'artifacts') and event.artifacts:
                    for art in event.artifacts:
                        name = getattr(art, "name", "")
                        # Reuse extraction logic (simplified here for brevity as streaming usually catches it)
                        pass
        
        except Exception as stream_err:
            print(f"BFF DEBUG: Streaming error: {stream_err}", flush=True)
            raise stream_err
        
        return {"response": response_data}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    session_id: str = Depends(get_session_id),
    client: A2AGrpcClient = Depends(get_client)
):
    session = session_store[session_id]
    context_id = session["context_id"]
    filename = file.filename
    
    if not filename:
        raise HTTPException(status_code=400, detail="No filename")

    content = await file.read()
    
    try:
        if filename.endswith('.csv'):
            try:
                df = pl.read_csv(io.BytesIO(content))
            except:
                df = pl.read_csv(io.BytesIO(content), encoding='latin-1')
        elif filename.endswith('.json'):
            json_data = json.loads(content.decode('utf-8'))
            if isinstance(json_data, list):
                df = pl.DataFrame(json_data)
            elif isinstance(json_data, dict):
                 df = pl.DataFrame(json_data) if all(isinstance(v, list) for v in json_data.values()) else pl.DataFrame([json_data])
            else:
                raise ValueError("Invalid JSON structure")
        else:
            raise HTTPException(status_code=400, detail="Unsupported type")

        sink = io.BytesIO()
        df.write_ipc_stream(sink)
        binary_data = sink.getvalue()
        b64_data = base64.b64encode(binary_data).decode('utf-8')
        
        data_payload = {
            "filename": filename,
            "arrow_stream": b64_data,
            "mime_type": "application/vnd.apache.arrow.stream",
            "rows": df.height,
            "columns": df.width,
            "col_names": df.columns
        }
        
        msg = Message(
            message_id=str(uuid.uuid4()),
            role=MessageRole.user,
            parts=[Part(root=DataPart(data=data_payload))],
            context_id=context_id
        )
        
        await client.send_message(MessageSendParams(message=msg))
        
        session["files"].append({
            "filename": filename,
            "rows": df.height,
            "columns": df.width
        })
        
        return data_payload

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Processing failed: {e}")

@app.get("/files", response_model=FileListResponse)
def list_files(session_id: str = Depends(get_session_id)):
    return {"files": session_store[session_id]["files"]}

@app.post("/clear")
def clear_session(session_id: str = Depends(get_session_id)):
    if session_id in session_store:
        session_store[session_id] = {
            "context_id": str(uuid.uuid4()),
            "files": []
        }
    return {"status": "ok"}

@app.get("/database")
def database_info():
    try:
        from utils.database import get_table_info, HAS_DUCKDB
        if not HAS_DUCKDB: return {"available": False}
        from utils.database import get_connection
        get_connection()
        all_tables = get_table_info()
        builtin = {k: v for k, v in all_tables.items() if not k.startswith("saved_")}
        return {"available": True, "tables": builtin, "saved_tables": {}}
    except:
        return {"available": False}
