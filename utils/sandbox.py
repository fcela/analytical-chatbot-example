"""Sandboxed Python code execution utility."""

import io
import sys
import json
import base64
import threading
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Any

# Safe imports that will be available in the sandbox
import math
import statistics

# Try to import optional data science libraries
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

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except ImportError:
    sm = None
    smf = None
    HAS_STATSMODELS = False

# DuckDB query function for sandbox
_db_connection = None

def _get_db_connection():
    """Get or create the DuckDB connection for sandbox use."""
    global _db_connection
    if _db_connection is None and HAS_DUCKDB:
        from utils.database import get_connection
        _db_connection = get_connection()
    return _db_connection

def query_db(sql: str):
    """
    Execute a SQL query against the DuckDB database.
    Returns a Polars DataFrame with Decimals cast to Float64 (for JSON serialization).
    """
    conn = _get_db_connection()
    if conn is None:
        raise RuntimeError("Database not available")
    
    df = conn.execute(sql).pl()
    
    # Auto-cast Decimal columns to Float64 to avoid JSON serialization errors in Altair
    if HAS_POLARS:
        decimal_cols = [
            col_name for col_name, dtype in df.schema.items() 
            if isinstance(dtype, pl.Decimal)
        ]
        
        if decimal_cols:
            df = df.with_columns([
                pl.col(col).cast(pl.Float64) for col in decimal_cols
            ])
            
    return df

def save_table(df, name: str, replace: bool = True) -> str:
    """
    Save a DataFrame to DuckDB for later use.
    Table will be named 'saved_{name}'.
    Returns confirmation message.
    """
    from utils.database import save_table as _save_table
    return _save_table(df, name, replace)

def load_table(name: str):
    """
    Load a previously saved table from DuckDB.
    Returns a Polars DataFrame.
    """
    from utils.database import load_table as _load_table
    return _load_table(name)

def list_saved_tables() -> list:
    """
    List all saved tables.
    Returns list of table names.
    """
    from utils.database import list_saved_tables as _list_saved
    return _list_saved()

def delete_table(name: str) -> str:
    """
    Delete a saved table.
    Returns confirmation message.
    """
    from utils.database import delete_table as _delete_table
    return _delete_table(name)


# Safe builtins - remove dangerous operations
SAFE_BUILTINS = {
    'abs': abs,
    'all': all,
    'any': any,
    'bool': bool,
    'chr': chr,
    'dict': dict,
    'divmod': divmod,
    'enumerate': enumerate,
    'filter': filter,
    'float': float,
    'format': format,
    'frozenset': frozenset,
    'hash': hash,
    'hex': hex,
    'int': int,
    'isinstance': isinstance,
    'issubclass': issubclass,
    'iter': iter,
    'len': len,
    'list': list,
    'map': map,
    'max': max,
    'min': min,
    'next': next,
    'oct': oct,
    'ord': ord,
    'pow': pow,
    'print': print,
    'range': range,
    'repr': repr,
    'reversed': reversed,
    'round': round,
    'set': set,
    'slice': slice,
    'sorted': sorted,
    'str': str,
    'sum': sum,
    'tuple': tuple,
    'type': type,
    'zip': zip,
    'True': True,
    'False': False,
    'None': None,
}

# Explicitly forbidden - things we never allow
FORBIDDEN_PATTERNS = [
    'import os',
    'import sys',
    'import subprocess',
    'import socket',
    'import requests',
    'import urllib',
    'import shutil',
    'import pickle',
    '__import__',
    'exec(',
    'eval(',
    'compile(',
    'open(',
    'file(',
    'input(',
    'raw_input(',
    'breakpoint(',
    '__builtins__',
    '__class__',
    '__bases__',
    '__subclasses__',
    '__mro__',
    '__globals__',
    '__code__',
    '__reduce__',
    'getattr',
    'setattr',
    'delattr',
    'globals(',
    'locals(',
    'vars(',
    'dir(',
]


def check_code_safety(code: str) -> tuple[bool, str]:
    """
    Check if code contains forbidden patterns.

    Args:
        code: The Python code to check

    Returns:
        Tuple of (is_safe, error_message)
    """
    code_lower = code.lower()
    for pattern in FORBIDDEN_PATTERNS:
        if pattern.lower() in code_lower:
            return False, f"Forbidden pattern detected: '{pattern}'"
    return True, ""


def _build_namespace(context: dict = None) -> dict:
    """Build the execution namespace with safe builtins and modules."""
    namespace = dict(SAFE_BUILTINS)

    # Add safe modules
    namespace['math'] = math
    namespace['statistics'] = statistics
    namespace['json'] = json

    if HAS_PANDAS:
        namespace['pd'] = pd
        namespace['pandas'] = pd

    if HAS_NUMPY:
        namespace['np'] = np
        namespace['numpy'] = np

    if HAS_ALTAIR:
        namespace['alt'] = alt
        namespace['altair'] = alt

    if HAS_POLARS:
        namespace['pl'] = pl
        namespace['polars'] = pl

    if HAS_STATSMODELS:
        namespace['sm'] = sm
        namespace['smf'] = smf
        namespace['statsmodels'] = sm

    # Add DuckDB query function and table persistence
    if HAS_DUCKDB:
        namespace['query_db'] = query_db
        namespace['sql'] = query_db  # Alias for convenience
        namespace['save_table'] = save_table
        namespace['load_table'] = load_table
        namespace['list_saved_tables'] = list_saved_tables
        namespace['delete_table'] = delete_table

    # Add context variables (uploaded data)
    if context:
        namespace.update(context)

    return namespace


def execute_sandboxed_code(
    code: str,
    context: dict[str, Any] = None,
    timeout_seconds: int = 30
) -> dict:
    """
    Execute Python code in a sandboxed environment.

    Args:
        code: Python code to execute
        context: Dictionary of variables to make available (e.g., uploaded DataFrames)
        timeout_seconds: Maximum execution time in seconds

    Returns:
        Dictionary with:
        - stdout: Captured standard output
        - stderr: Captured standard error
        - plots: List of base64-encoded plot images
        - tables: List of HTML table strings
        - success: Whether execution completed without error
        - error: Error message if any
    """
    result = {
        "stdout": "",
        "stderr": "",
        "plots": [],
        "tables": [],
        "html": [],
        "success": False,
        "error": None
    }

    # Table capture list (will be populated by show_table)
    captured_tables = []
    # HTML capture list (will be populated by show_html)
    captured_html = []

    def show_table(df, title: str = None):
        """
        Display a DataFrame as a nicely formatted HTML table.
        Works with both Polars and Pandas DataFrames.
        """
        # Convert Polars to Pandas for HTML rendering
        if HAS_POLARS and isinstance(df, pl.DataFrame):
            df_for_html = df.to_pandas()
        elif HAS_PANDAS and isinstance(df, pd.DataFrame):
            df_for_html = df
        else:
            # Try to convert to pandas if possible
            if hasattr(df, 'to_pandas'):
                df_for_html = df.to_pandas()
            else:
                print(str(df))
                return

        # Generate HTML table
        html = df_for_html.to_html(
            index=False,
            classes='data-table',
            border=0,
            na_rep='-'
        )

        # Add title if provided
        if title:
            html = f'<div class="table-title">{title}</div>\n{html}'

        captured_tables.append(html)

        # Also print a summary to stdout
        rows, cols = df_for_html.shape
        print(f"[Table: {rows} rows × {cols} columns]")

    def show_html(content: str):
        """
        Display raw HTML content (e.g. for dashboards).
        """
        captured_html.append(str(content))
        print(f"[HTML Output: {len(content)} chars]")

    # Safety check
    is_safe, error_msg = check_code_safety(code)
    if not is_safe:
        result["error"] = f"Security error: {error_msg}"
        return result

    # Build namespace
    namespace = _build_namespace(context)

    # Add show_table function to namespace
    namespace['show_table'] = show_table
    namespace['display'] = show_table  # Alias for familiarity
    namespace['show_html'] = show_html

    # Capture stdout and stderr
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    # Thread result container
    thread_result = {"exception": None, "completed": False}

    def run_code():
        try:
            compiled_code = compile(code, '<sandbox>', 'exec')
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                exec(compiled_code, namespace)
            thread_result["completed"] = True
        except Exception as e:
            thread_result["exception"] = e

    # Execute in thread with timeout
    thread = threading.Thread(target=run_code, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        result["error"] = f"Code execution timed out ({timeout_seconds} second limit)"
        return result

    if thread_result["exception"]:
        exc = thread_result["exception"]
        if isinstance(exc, SyntaxError):
            result["error"] = f"Syntax error: {exc}"
        else:
            result["error"] = f"{type(exc).__name__}: {exc}"
            result["stderr"] = traceback.format_exc()
    elif thread_result["completed"]:
        result["success"] = True

        # Capture Altair charts
        if HAS_ALTAIR and HAS_VL_CONVERT:
            for name, val in list(namespace.items()):
                if isinstance(val, alt.TopLevelMixin):
                    try:
                        # Capture as SVG for better quality
                        svg_data = vlc.vegalite_to_svg(vl_spec=val.to_dict())
                        # Encode as base64
                        result["plots"].append(base64.b64encode(svg_data.encode('utf-8')).decode('utf-8'))
                    except Exception as e:
                        stderr_buffer.write(f"\nWarning: Failed to render Altair chart '{name}': {e}\n")

    result["stdout"] = stdout_buffer.getvalue()
    if not result["stderr"]:
        result["stderr"] = stderr_buffer.getvalue()

    # Capture tables
    result["tables"] = captured_tables
    result["html"] = captured_html

    return result


if __name__ == "__main__":
    # Test cases
    print("Test 1: Simple calculation")
    result = execute_sandboxed_code("print(2 + 2)")
    print(f"  Success: {result['success']}, Output: {result['stdout'].strip()}")

    print("\nTest 2: Pandas operation")
    if HAS_PANDAS:
        code = """
df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
print(df.mean())
"""
        result = execute_sandboxed_code(code)
        print(f"  Success: {result['success']}")
        print(f"  Output: {result['stdout']}")
    else:
        print("  Skipped (pandas not installed)")

    print("\nTest 3: Altair chart")
    if HAS_ALTAIR and HAS_VL_CONVERT:
        code = """
df = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [1, 4, 9, 16, 25]})
chart = alt.Chart(df).mark_line().encode(x='x', y='y')
print("Chart created")
"""
        result = execute_sandboxed_code(code)
        print(f"  Success: {result['success']}, Plots generated: {len(result['plots'])}")
    else:
        print("  Skipped (altair or vl-convert not installed)")

    print("\nTest 4: Forbidden operation (should fail)")
    result = execute_sandboxed_code("import os; os.system('ls')")
    print(f"  Success: {result['success']}, Error: {result['error']}")

    print("\nTest 5: Context variables")
    if HAS_PANDAS:
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [10, 20, 30]})
        code = "print(data.describe())"
        result = execute_sandboxed_code(code, context={"data": df})
        print(f"  Success: {result['success']}")
        print(f"  Output: {result['stdout']}")
