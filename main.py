"""Entrypoint script to run the FastAPI backend (and optionally serve built frontend).

Usage:
  python main.py    # runs uvicorn programmatically

Or use make target:
  make start-backend
"""

import os
import uvicorn

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "true").lower() in ("1", "true", "yes")

    uvicorn.run("api:app", host=host, port=port, reload=reload)
