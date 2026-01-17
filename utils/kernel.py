"""
Kernel process logic for the analytical chatbot sandbox.
This module runs INSIDE the isolated subprocess.
"""

import sys
import io
import traceback
import json
import base64
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
    for pattern in FORBIDDEN_PATTERNS:
        if pattern in code: 
            return False, f"Forbidden pattern: {pattern}"
    return True, ""

# DuckDB helper functions for the kernel
# We need to re-implement or import these so they work within the worker process.
# Since we want a fresh DuckDB instance for each kernel (sandbox), we can't share the connection object.
# We will use utils.database to get a connection, which will initialize sample data in this process.

_db_conn = None

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


class KernelWorker:
    def __init__(self, pipe):
        self.pipe = pipe
        self.namespace = {}
        self.artifacts = {}
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
        self.namespace['display'] = self.show_table
        self.namespace['show_html'] = self.show_html
        self.namespace['artifacts'] = self.artifacts

    def _gen_id(self, prefix):
        return f"{prefix}_{str(uuid.uuid4())[:8]}"

    def show_table(self, df, title: str = None):
        if HAS_POLARS and isinstance(df, pl.DataFrame):
            df_html = df.to_pandas()
        elif hasattr(df, 'to_pandas'):
            df_html = df.to_pandas()
        else:
            df_html = df
            
        if hasattr(df_html, 'to_html'):
            html = df_html.to_html(index=False, classes='data-table', border=0, na_rep='-')
            if title: html = f'<div class="table-title">{title}</div>\n{html}'
            
            art_id = self._gen_id("table")
            self.artifacts[art_id] = {"type": "table", "content": html}
            print(f"\n```table {art_id}```\n")

    def show_html(self, content: str):
        art_id = self._gen_id("html")
        self.artifacts[art_id] = {"type": "html", "content": str(content)}
        print(f"\n```html {art_id}```\n")

    def capture_plots(self):
        """Capture Altair plots from namespace."""
        if HAS_ALTAIR and HAS_VL_CONVERT:
            for name, val in list(self.namespace.items()):
                if name.startswith('_'): continue
                if isinstance(val, alt.TopLevelMixin):
                    try:
                        svg_data = vlc.vegalite_to_svg(vl_spec=val.to_dict())
                        b64_svg = base64.b64encode(svg_data.encode('utf-8')).decode('utf-8')
                        
                        art_id = self._gen_id("plot")
                        self.artifacts[art_id] = {"type": "plot", "content": b64_svg}
                        return f"\n```plot {art_id}```\n" 
                    except Exception:
                        pass
        return ""

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
        self.artifacts = {} 
        self.namespace['artifacts'] = self.artifacts
        
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
            
            plot_tag = self.capture_plots()
            if plot_tag:
                stdout_buffer.write(plot_tag)
                
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
