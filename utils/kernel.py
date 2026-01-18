"""
Kernel process logic for the analytical chatbot sandbox.
This module runs INSIDE the isolated subprocess.
"""

import sys
import io
import traceback
import json
from decimal import Decimal
import base64
import re
import uuid
from typing import Any
from contextlib import redirect_stdout, redirect_stderr

# Imports for the sandbox environment
import math
import statistics

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    pd = None
    HAS_PANDAS = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

try:
    import altair as alt
    HAS_ALTAIR = True
except ImportError:
    alt = None
    HAS_ALTAIR = False

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    pl = None
    HAS_POLARS = False

try:
    import vl_convert as vlc
    HAS_VL_CONVERT = True
except ImportError:
    vlc = None
    HAS_VL_CONVERT = False

try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    duckdb = None
    HAS_DUCKDB = False

# We need to redefine safety checks here or move them to a common module.
FORBIDDEN_PATTERNS = [
    'import os', 'import sys', 'import subprocess', 'import socket', 'import requests',
    'import urllib', 'import shutil', 'import pickle', '__import__', 'exec(', 'eval(',
    'open(', 'input(', '__builtins__', '__globals__', 'sys.modules'
]

def check_safety(code: str) -> tuple[bool, str]:
    """Lightweight guardrail to block obviously dangerous operations."""
    for pattern in FORBIDDEN_PATTERNS:
        if pattern in code: 
            return False, f"Forbidden pattern: {pattern}"
    return True, ""

# DuckDB helper functions for the kernel
# We need to re-implement or import these so they work within the worker process.
# Since we want a fresh DuckDB instance for each kernel (sandbox), we can't share the connection object.
# We will use utils.database to get a connection, which will initialize sample data in this process.

_db_conn = None

def _sanitize_vl_spec(value: Any) -> Any:
    """Convert non-JSON types (e.g., Decimal) in Vega-Lite specs to safe primitives."""
    if isinstance(value, Decimal):
        return float(value)
    if HAS_NUMPY:
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
    if isinstance(value, dict):
        return {k: _sanitize_vl_spec(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_vl_spec(v) for v in value]
    if isinstance(value, tuple):
        return [_sanitize_vl_spec(v) for v in value]
    return value

def _is_safe_artifact_id(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_]+", value or ""))

_MERMAID_PREFIXES = (
    "graph",
    "flowchart",
    "sequenceDiagram",
    "classDiagram",
    "stateDiagram",
    "erDiagram",
    "journey",
    "gantt",
    "pie",
    "mindmap",
    "timeline",
    "quadrantChart",
    "requirementDiagram",
    "gitGraph",
)


def _looks_like_mermaid(text: str) -> bool:
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    if stripped.startswith("```mermaid"):
        stripped = stripped[len("```mermaid"):].strip()
        if stripped.endswith("```"):
            stripped = stripped[:-3].strip()
    for line in stripped.splitlines():
        line = line.strip()
        if not line:
            continue
        return any(line.startswith(prefix) for prefix in _MERMAID_PREFIXES)
    return False

def get_db_conn():
    global _db_conn
    if _db_conn is None and HAS_DUCKDB:
        from utils.database import get_connection
        # This creates a new in-memory DB and populates it
        _db_conn = get_connection()
    return _db_conn

def query_db(sql: str):
    conn = get_db_conn()
    if conn is None:
        raise RuntimeError("Database not available")
    
    # Execute and return Polars DataFrame
    # If Polars is available, DuckDB can output to it directly
    if HAS_POLARS:
        return conn.execute(sql).pl()
    else:
        return conn.execute(sql).df()

def save_table(df, name: str, replace: bool = True):
    from utils.database import save_table as _save_table
    return _save_table(df, name, replace)

def load_table(name: str):
    from utils.database import load_table as _load_table
    return _load_table(name)

def list_saved_tables():
    from utils.database import list_saved_tables as _list_saved
    return _list_saved()

def delete_table(name: str):
    from utils.database import delete_table as _delete_table
    return _delete_table(name)


class Markdown:
    """Mock for IPython.display.Markdown."""
    def __init__(self, data):
        self.data = data
    def __str__(self):
        return str(self.data)
    def _repr_markdown_(self):
        return str(self.data)

class KernelWorker:
    def __init__(self, pipe):
        self.pipe = pipe
        self.namespace = {}
        self.artifacts = {}
        self.seen_plots = set()
        self._init_namespace()

    def _init_namespace(self):
        """Initialize the execution namespace."""
        self.namespace = {
            'math': math,
            'statistics': statistics,
            'json': json,
            'abs': abs, 'round': round, 'len': len, 'print': print,
            'list': list, 'dict': dict, 'set': set, 'tuple': tuple, 'str': str, 'int': int, 'float': float, 'bool': bool,
            'range': range, 'enumerate': enumerate, 'zip': zip, 'map': map, 'filter': filter, 'sorted': sorted,
            'Markdown': Markdown,
        }
        
        if HAS_PANDAS: self.namespace['pd'] = pd; self.namespace['pandas'] = pd
        if HAS_NUMPY: self.namespace['np'] = np; self.namespace['numpy'] = np
        if HAS_ALTAIR: self.namespace['alt'] = alt; self.namespace['altair'] = alt
        if HAS_POLARS: self.namespace['pl'] = pl; self.namespace['polars'] = pl
        
        if HAS_DUCKDB:
            self.namespace['query_db'] = query_db
            self.namespace['save_table'] = save_table
            self.namespace['load_table'] = load_table
            self.namespace['list_saved_tables'] = list_saved_tables
            self.namespace['delete_table'] = delete_table
        
        # Add helper functions
        self.namespace['show_table'] = self.show_table
        self.namespace['display'] = self.display
        self.namespace['show_html'] = self.show_html
        self.namespace['print_md'] = self.print_md
        self.namespace['artifacts'] = self.artifacts

    def _gen_id(self, prefix):
        return f"{prefix}_{str(uuid.uuid4())[:8]}"

    def print_md(self, obj):
        """Prints an object as Markdown to stdout."""
        if (HAS_PANDAS and isinstance(obj, pd.DataFrame)) or \
           (HAS_POLARS and isinstance(obj, pl.DataFrame)):
            try:
                # Use tabulate if available (pandas.to_markdown uses it)
                if HAS_POLARS and isinstance(obj, pl.DataFrame):
                    print(obj.to_pandas().to_markdown(index=False))
                else:
                    print(obj.to_markdown(index=False))
            except Exception:
                # Fallback to string representation if tabulate fails
                print(str(obj))
        else:
            print(str(obj))

    def display(self, obj, type=None, label=None):
        """
        Generic display function to render objects as artifacts.
        Registers the artifact and prints the reference tag.
        """
        art_id = label if (label and _is_safe_artifact_id(label)) else None
        label_text = label if (label and not _is_safe_artifact_id(label)) else None
        
        # Handle Markdown objects
        if isinstance(obj, Markdown):
            obj = obj.data
            if type is None:
                type = "markdown"

        # Determine type if not provided so callers can just call display(obj).
        if type is None:
            if HAS_ALTAIR and isinstance(obj, alt.TopLevelMixin):
                type = "plot"
            elif (HAS_PANDAS and isinstance(obj, pd.DataFrame)) or \
                 (HAS_POLARS and isinstance(obj, pl.DataFrame)):
                type = "table"
            elif hasattr(obj, '_repr_html_'):
                type = "html"
            else:
                type = "text"
        
        # Automatically tag Mermaid strings so the UI renders them.
        if (type == "text" or type == "markdown") and _looks_like_mermaid(obj):
            type = "mermaid"

        content = None
        tag_type = type

        if type == "plot":
            # Convert Altair to SVG
            if HAS_ALTAIR and HAS_VL_CONVERT and isinstance(obj, alt.TopLevelMixin):
                try:
                    vl_spec = _sanitize_vl_spec(obj.to_dict())
                    svg_data = vlc.vegalite_to_svg(vl_spec=vl_spec)
                    content = base64.b64encode(svg_data.encode('utf-8')).decode('utf-8')
                    self.seen_plots.add(id(obj))
                except Exception as e:
                    print(f"Error converting plot: {e}")
                    return
            else:
                # Fallback or other plot types
                print(f"Unsupported plot object: {type(obj)}")
                return

        elif type == "table":
            # Table conversion logic
            if HAS_POLARS and isinstance(obj, pl.DataFrame):
                df_html = obj.to_pandas()
            elif hasattr(obj, 'to_pandas'):
                df_html = obj.to_pandas()
            else:
                df_html = obj
                
            if hasattr(df_html, 'to_html'):
                content = df_html.to_html(index=False, classes='data-table', border=0, na_rep='-')
                if label: 
                    # If label is provided, maybe add it as title in HTML?
                    # For now, we use label as ID, so maybe we want a separate title param.
                    # Reusing existing pattern:
                    content = f'<div class="table-title">{label}</div>\n{content}'
                    art_id = self._gen_id("table") # Generate random ID to avoid collision if label is not unique or safe
            else:
                content = str(obj)

        elif type == "html":
             content = obj._repr_html_() if hasattr(obj, '_repr_html_') else str(obj)
        
        elif type == "mermaid":
            content = str(obj)
        
        elif type == "markdown" or type == "md":
            content = str(obj)
            tag_type = "markdown"

        else:
            content = str(obj)

        if content:
            # Ensure ID is safe
            if type == "plot" and not art_id:
                art_id = self._gen_id("plot")
            elif type == "table" and not art_id:
                art_id = self._gen_id("table")
            elif not art_id:
                art_id = self._gen_id(type)

            # Store artifacts and emit a tag that the LLM response should include.
            self.artifacts[art_id] = {"type": tag_type, "content": content}
            if label_text and type != "table":
                print(label_text)
            print(f"\n```{tag_type} {art_id}```\n")

    def show_table(self, df, title: str = None):
        """Legacy wrapper for display(df, type='table')."""
        self.display(df, type='table', label=title)

    def show_html(self, content: str):
        """Legacy wrapper for display(content, type='html')."""
        self.display(content, type='html')

    def capture_plots(self):
        """Capture all Altair plots from namespace that haven't been displayed."""
        output_tags = []
        if HAS_ALTAIR and HAS_VL_CONVERT:
            for name, val in list(self.namespace.items()):
                if name.startswith('_'): continue
                if isinstance(val, alt.TopLevelMixin):
                    if id(val) in self.seen_plots:
                        continue
                        
                    try:
                        vl_spec = _sanitize_vl_spec(val.to_dict())
                        svg_data = vlc.vegalite_to_svg(vl_spec=vl_spec)
                        b64_svg = base64.b64encode(svg_data.encode('utf-8')).decode('utf-8')
                        
                        art_id = self._gen_id("plot")
                        self.artifacts[art_id] = {"type": "plot", "content": b64_svg}
                        self.seen_plots.add(id(val))
                        output_tags.append(f"\n```plot {art_id}```\n") 
                    except Exception:
                        pass
        return "".join(output_tags)

    def run(self):
        """Main loop waiting for commands."""
        while True:
            try:
                cmd = self.pipe.recv()
                if cmd['type'] == 'execute':
                    self.execute_code(cmd['code'])
                elif cmd['type'] == 'load_file':
                    self.load_file(cmd['filename'], cmd['arrow_data'])
                elif cmd['type'] == 'terminate':
                    break
            except EOFError:
                break
            except Exception as e:
                self.pipe.send({'status': 'fatal_error', 'error': str(e)})

    def execute_code(self, code):
        # Reset per-execution artifact state while keeping the kernel namespace.
        self.artifacts = {} 
        self.namespace['artifacts'] = self.artifacts
        pre_existing_plots = set()
        if HAS_ALTAIR:
            for val in self.namespace.values():
                if isinstance(val, alt.TopLevelMixin):
                    pre_existing_plots.add(id(val))
        self.seen_plots = pre_existing_plots
        
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        
        is_safe, error_msg = check_safety(code)
        if not is_safe:
            self.pipe.send({'status': 'error', 'error': error_msg, 'stdout': '', 'stderr': ''})
            return

        success = False
        error = None
        
        try:
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                exec(code, self.namespace)
            
            plot_tags = self.capture_plots()
            if plot_tags:
                stdout_buffer.write(plot_tags)
                
            success = True
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            traceback.print_exc(file=stderr_buffer)

        self.pipe.send({
            'status': 'ok' if success else 'error',
            'stdout': stdout_buffer.getvalue(),
            'stderr': stderr_buffer.getvalue(),
            'artifacts': self.artifacts,
            'error': error
        })

    def load_file(self, filename, arrow_data):
        try:
            if HAS_POLARS:
                df = pl.read_ipc_stream(io.BytesIO(arrow_data))
                var_name = filename.rsplit('.', 1)[0].replace(' ', '_').replace('-', '_')
                self.namespace[var_name] = df
                self.pipe.send({'status': 'ok', 'message': f"Loaded {var_name}"})
            else:
                self.pipe.send({'status': 'error', 'error': "Polars not installed"})
        except Exception as e:
            self.pipe.send({'status': 'error', 'error': str(e)})

def run_worker(pipe):
    worker = KernelWorker(pipe)
    worker.run()
