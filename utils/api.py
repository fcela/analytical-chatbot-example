"""
Legacy FastAPI Backend for Analytical Chatbot.

This module predates the A2A + BFF architecture and remains for reference
and backwards compatibility during development/testing.
"""

import os
import io
import json
import uuid
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Import existing logic
try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    pl = None
    HAS_POLARS = False

try:
    from utils.database import get_connection, get_table_info, list_saved_tables, HAS_DUCKDB
except ImportError:
    HAS_DUCKDB = False
    def get_table_info():
        return {}
    def list_saved_tables():
        return []

from utils.flow import run_chatbot
from utils.call_llm import get_provider_info

# --- Data Models ---

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: Dict[str, Any]

class FileInfo(BaseModel):
    filename: str
    rows: int
    columns: int

class FileListResponse(BaseModel):
    files: List[FileInfo]

# --- Session Management ---

# In-memory store for session data (Files + History)
# Structure: { session_id: { "files": {name: df}, "history": [] } }
session_store: Dict[str, Dict[str, Any]] = {}

def get_session_id(request: Request, response: Response) -> str:
    """
    Retrieve or create a session ID via cookie.
    """
    session_id = request.cookies.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        response.set_cookie(key="session_id", value=session_id, httponly=True)
    
    # Initialize store if empty
    if session_id not in session_store:
        session_store[session_id] = {
            "files": {},
            "history": []
        }
    
    return session_id

# --- App Setup ---

app = FastAPI(title="Analytical Chatbot API")

# Allow CORS for local React development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"], # React/Vite defaults
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"status": "running", "backend": "FastAPI", "database": HAS_DUCKDB}

@app.get("/llm")
def llm_info():
    """Get information about the current LLM provider configuration."""
    return get_provider_info()

@app.get("/database")
def database_info():
    """Get information about the available database tables."""
    if not HAS_DUCKDB:
        return {"available": False, "tables": {}, "saved_tables": []}

    # Initialize connection if needed
    get_connection()

    all_tables = get_table_info()
    builtin = {k: v for k, v in all_tables.items() if not k.startswith("saved_")}
    saved = {k: v for k, v in all_tables.items() if k.startswith("saved_")}

    return {
        "available": True,
        "tables": builtin,
        "saved_tables": saved
    }

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(
    req: ChatRequest, 
    session_id: str = Depends(get_session_id)
):
    session_data = session_store[session_id]
    
    try:
        # Run the existing flow logic
        result = run_chatbot(
            user_message=req.message,
            uploaded_files=session_data["files"],
            chat_history=session_data["history"]
        )

        # Update history in store
        session_data["history"] = result["chat_history"]

        return {"response": result["response"]}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
def upload_file(
    file: UploadFile = File(...),
    session_id: str = Depends(get_session_id)
):
    if not HAS_POLARS:
        raise HTTPException(status_code=500, detail="Polars not installed")

    session_data = session_store[session_id]
    filename = file.filename

    if not filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    try:
        content = file.file.read()

        if not content:
            raise HTTPException(status_code=400, detail="Empty file")

        if filename.endswith('.csv'):
            # Try UTF-8 first, fall back to latin-1
            try:
                df = pl.read_csv(io.BytesIO(content))
            except Exception:
                df = pl.read_csv(io.BytesIO(content), encoding='latin-1')
        elif filename.endswith('.json'):
            # Parse JSON manually since Polars expects NDJSON by default
            json_data = json.loads(content.decode('utf-8'))
            # Handle both array of records and single object
            if isinstance(json_data, list):
                df = pl.DataFrame(json_data)
            elif isinstance(json_data, dict):
                # Check if it's a columnar format {"col1": [...], "col2": [...]}
                if all(isinstance(v, list) for v in json_data.values()):
                    df = pl.DataFrame(json_data)
                else:
                    # Single record, wrap in list
                    df = pl.DataFrame([json_data])
            else:
                raise ValueError("JSON must be an array of records or an object")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Use .csv or .json")

        # Store in session
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

@app.get("/files", response_model=FileListResponse)
def list_files(session_id: str = Depends(get_session_id)):
    session_data = session_store[session_id]
    files_info = []

    for filename, df in session_data["files"].items():
        files_info.append({
            "filename": filename,
            "rows": df.height,
            "columns": df.width
        })

    return {"files": files_info}

@app.delete("/upload/{filename}")
def delete_file(filename: str, session_id: str = Depends(get_session_id)):
    session_data = session_store[session_id]
    if filename in session_data["files"]:
        del session_data["files"][filename]
    return {"status": "ok"}

@app.post("/clear")
def clear_session(session_id: str = Depends(get_session_id)):
    if session_id in session_store:
        session_store[session_id] = {
            "files": {},
            "history": []
        }
    return {"status": "ok"}
