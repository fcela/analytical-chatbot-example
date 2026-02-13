"""DuckDB database tools for the DeepAgent."""

from langchain_core.tools import tool


@tool
def query_database(sql: str) -> str:
    """Execute a SQL query against the DuckDB database.

    Available tables: employees, products, sales, customers.
    Returns results as a formatted string.

    Args:
        sql: SQL query to execute.

    Returns:
        Query results as a formatted table string, or error message.
    """
    try:
        from utils.database import execute_query
        result = execute_query(sql)
        return str(result)
    except Exception as e:
        return f"Query error: {e}"


@tool
def get_database_schema() -> str:
    """Get the schema of all available database tables.

    Returns column names, types, and row counts for each table.
    Use this before writing SQL queries to understand the data structure.

    Returns:
        Formatted schema description string.
    """
    try:
        from utils.database import get_schema_description
        return get_schema_description()
    except Exception as e:
        return f"Schema error: {e}"
