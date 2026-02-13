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
    - Cast Decimal to Float64 before charting with Altair

    Args:
        code: Python code to execute. Must be valid Python.

    Returns:
        JSON string with keys: success, stdout, artifacts, error
    """
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
