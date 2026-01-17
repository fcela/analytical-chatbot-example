"""DuckDB database utility for the analytical chatbot."""

import os
import threading
from typing import Optional

try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    duckdb = None
    HAS_DUCKDB = False

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    pl = None
    HAS_POLARS = False

# Database file path (in-memory for simplicity, can be changed to file-based)
DB_PATH = ":memory:"

# Global connection (per-process)
_connection: Optional["duckdb.DuckDBPyConnection"] = None
_connection_pid: Optional[int] = None
_lock = threading.Lock()


def get_connection() -> "duckdb.DuckDBPyConnection":
    """Get or create the DuckDB connection."""
    global _connection, _connection_pid
    
    current_pid = os.getpid()
    
    # Reset connection if we are in a different process (fork safety)
    if _connection is not None and _connection_pid != current_pid:
        _connection = None

    if _connection is None:
        with _lock:
            if _connection is None:
                if not HAS_DUCKDB:
                    raise RuntimeError("DuckDB is not installed")
                _connection = duckdb.connect(DB_PATH)
                _initialize_sample_data(_connection)
                _connection_pid = current_pid
    return _connection


def _initialize_sample_data(conn: "duckdb.DuckDBPyConnection"):
    """Populate the database with sample data."""

    # 1. Employees table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY,
            name VARCHAR,
            department VARCHAR,
            salary DECIMAL(10,2),
            hire_date DATE,
            is_manager BOOLEAN
        )
    """)

    conn.execute("""
        INSERT INTO employees VALUES
        (1, 'Alice Johnson', 'Engineering', 95000.00, '2020-03-15', false),
        (2, 'Bob Smith', 'Engineering', 120000.00, '2018-07-22', true),
        (3, 'Carol Williams', 'Marketing', 75000.00, '2021-01-10', false),
        (4, 'David Brown', 'Marketing', 85000.00, '2019-11-05', true),
        (5, 'Eva Martinez', 'Sales', 65000.00, '2022-02-28', false),
        (6, 'Frank Lee', 'Sales', 90000.00, '2017-09-14', true),
        (7, 'Grace Kim', 'Engineering', 105000.00, '2019-04-01', false),
        (8, 'Henry Wilson', 'HR', 70000.00, '2020-08-20', false),
        (9, 'Iris Chen', 'HR', 80000.00, '2018-12-03', true),
        (10, 'Jack Taylor', 'Engineering', 88000.00, '2021-06-15', false)
    """)

    # 2. Products table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY,
            name VARCHAR,
            category VARCHAR,
            price DECIMAL(10,2),
            stock_quantity INTEGER,
            supplier VARCHAR
        )
    """)

    conn.execute("""
        INSERT INTO products VALUES
        (1, 'Laptop Pro', 'Electronics', 1299.99, 50, 'TechCorp'),
        (2, 'Wireless Mouse', 'Electronics', 29.99, 200, 'TechCorp'),
        (3, 'Office Chair', 'Furniture', 249.99, 30, 'ComfortPlus'),
        (4, 'Standing Desk', 'Furniture', 599.99, 15, 'ComfortPlus'),
        (5, 'Monitor 27"', 'Electronics', 349.99, 75, 'DisplayMax'),
        (6, 'Keyboard Mechanical', 'Electronics', 89.99, 120, 'TechCorp'),
        (7, 'Desk Lamp', 'Furniture', 45.99, 80, 'LightWorks'),
        (8, 'Webcam HD', 'Electronics', 79.99, 60, 'DisplayMax'),
        (9, 'Notebook Pack', 'Supplies', 12.99, 500, 'OfficeMart'),
        (10, 'Pen Set', 'Supplies', 8.99, 300, 'OfficeMart')
    """)

    # 3. Sales table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sales (
            id INTEGER PRIMARY KEY,
            product_id INTEGER,
            employee_id INTEGER,
            quantity INTEGER,
            sale_date DATE,
            total_amount DECIMAL(10,2),
            region VARCHAR
        )
    """)

    conn.execute("""
        INSERT INTO sales VALUES
        (1, 1, 5, 2, '2024-01-15', 2599.98, 'North'),
        (2, 2, 5, 10, '2024-01-16', 299.90, 'North'),
        (3, 3, 6, 5, '2024-01-17', 1249.95, 'South'),
        (4, 5, 5, 3, '2024-01-18', 1049.97, 'North'),
        (5, 1, 6, 1, '2024-01-20', 1299.99, 'South'),
        (6, 4, 6, 2, '2024-01-22', 1199.98, 'South'),
        (7, 6, 5, 8, '2024-01-25', 719.92, 'East'),
        (8, 8, 6, 4, '2024-01-28', 319.96, 'West'),
        (9, 9, 5, 50, '2024-02-01', 649.50, 'North'),
        (10, 10, 6, 30, '2024-02-03', 269.70, 'South'),
        (11, 1, 5, 3, '2024-02-05', 3899.97, 'East'),
        (12, 2, 6, 15, '2024-02-08', 449.85, 'West'),
        (13, 5, 5, 5, '2024-02-10', 1749.95, 'North'),
        (14, 7, 6, 10, '2024-02-12', 459.90, 'South'),
        (15, 3, 5, 3, '2024-02-15', 749.97, 'East')
    """)

    # 4. Customers table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            id INTEGER PRIMARY KEY,
            name VARCHAR,
            email VARCHAR,
            city VARCHAR,
            country VARCHAR,
            signup_date DATE,
            total_purchases DECIMAL(10,2)
        )
    """)

    conn.execute("""
        INSERT INTO customers VALUES
        (1, 'Acme Corp', 'contact@acme.com', 'New York', 'USA', '2023-01-15', 15420.50),
        (2, 'GlobalTech', 'info@globaltech.com', 'London', 'UK', '2023-02-20', 8750.00),
        (3, 'StartupXYZ', 'hello@startupxyz.com', 'San Francisco', 'USA', '2023-03-10', 3200.75),
        (4, 'MegaStore', 'sales@megastore.com', 'Toronto', 'Canada', '2023-04-05', 22100.00),
        (5, 'EuroServices', 'contact@euroservices.eu', 'Berlin', 'Germany', '2023-05-18', 6890.25),
        (6, 'AsiaTrading', 'info@asiatrading.com', 'Tokyo', 'Japan', '2023-06-22', 12500.00),
        (7, 'LocalBiz', 'owner@localbiz.com', 'Chicago', 'USA', '2023-07-30', 1850.50),
        (8, 'TechStartup', 'team@techstartup.io', 'Austin', 'USA', '2023-08-14', 4200.00),
        (9, 'RetailKing', 'orders@retailking.com', 'Miami', 'USA', '2023-09-25', 9800.75),
        (10, 'ConsultPro', 'info@consultpro.com', 'Sydney', 'Australia', '2023-10-08', 5600.00)
    """)

    print("Sample data initialized in DuckDB:")
    print("  - employees (10 rows)")
    print("  - products (10 rows)")
    print("  - sales (15 rows)")
    print("  - customers (10 rows)")


def execute_query(sql: str) -> "pl.DataFrame":
    """
    Execute a SQL query and return results as a Polars DataFrame.

    Args:
        sql: SQL query string

    Returns:
        Polars DataFrame with query results
    """
    if not HAS_DUCKDB:
        raise RuntimeError("DuckDB is not installed")
    if not HAS_POLARS:
        raise RuntimeError("Polars is not installed")

    conn = get_connection()
    result = conn.execute(sql).pl()
    return result


def get_table_info() -> dict:
    """Get information about available tables."""
    conn = get_connection()

    tables = conn.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'main'
    """).fetchall()

    table_info = {}
    for (table_name,) in tables:
        columns = conn.execute(f"""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = '{table_name}'
        """).fetchall()

        row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

        table_info[table_name] = {
            "columns": {col: dtype for col, dtype in columns},
            "row_count": row_count
        }

    return table_info


def get_schema_description() -> str:
    """Get a formatted description of the database schema for LLM prompts."""
    info = get_table_info()

    # Separate built-in and saved tables
    builtin_tables = {k: v for k, v in info.items() if not k.startswith("saved_")}
    saved_tables = {k: v for k, v in info.items() if k.startswith("saved_")}

    lines = ["Available database tables:"]

    # Built-in tables
    for table_name, details in builtin_tables.items():
        lines.append(f"\n{table_name} ({details['row_count']} rows):")
        for col, dtype in details["columns"].items():
            lines.append(f"  - {col}: {dtype}")

    # Saved tables (from previous analysis)
    if saved_tables:
        lines.append("\n\nSaved tables (from previous analysis):")
        for table_name, details in saved_tables.items():
            lines.append(f"\n{table_name} ({details['row_count']} rows):")
            for col, dtype in details["columns"].items():
                lines.append(f"  - {col}: {dtype}")

    return "\n".join(lines)


def save_table(df: "pl.DataFrame", name: str, replace: bool = True) -> str:
    """
    Save a Polars DataFrame as a table in DuckDB.

    Args:
        df: Polars DataFrame to save
        name: Table name (will be prefixed with 'saved_' for safety)
        replace: If True, replace existing table; if False, raise error if exists

    Returns:
        Confirmation message
    """
    if not HAS_DUCKDB or not HAS_POLARS:
        raise RuntimeError("DuckDB or Polars not available")

    # Sanitize table name - only allow alphanumeric and underscore
    import re
    safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    table_name = f"saved_{safe_name}"

    conn = get_connection()

    if replace:
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")

    # Register the Polars DataFrame and create table from it
    conn.register("_temp_df", df)
    conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM _temp_df")
    conn.unregister("_temp_df")

    row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    return f"Saved {row_count} rows to table '{table_name}'"


def load_table(name: str) -> "pl.DataFrame":
    """
    Load a saved table from DuckDB as a Polars DataFrame.

    Args:
        name: Table name (with or without 'saved_' prefix)

    Returns:
        Polars DataFrame
    """
    if not HAS_DUCKDB:
        raise RuntimeError("DuckDB not available")

    conn = get_connection()

    # Handle both with and without prefix
    table_name = name if name.startswith("saved_") else f"saved_{name}"

    return conn.execute(f"SELECT * FROM {table_name}").pl()


def list_saved_tables() -> list:
    """
    List all user-saved tables (those with 'saved_' prefix).

    Returns:
        List of table names
    """
    if not HAS_DUCKDB:
        return []

    conn = get_connection()
    tables = conn.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'main' AND table_name LIKE 'saved_%'
    """).fetchall()

    return [t[0] for t in tables]


def delete_table(name: str) -> str:
    """
    Delete a saved table.

    Args:
        name: Table name (with or without 'saved_' prefix)

    Returns:
        Confirmation message
    """
    if not HAS_DUCKDB:
        raise RuntimeError("DuckDB not available")

    conn = get_connection()
    table_name = name if name.startswith("saved_") else f"saved_{name}"

    conn.execute(f"DROP TABLE IF EXISTS {table_name}")
    return f"Deleted table '{table_name}'"


if __name__ == "__main__":
    # Test the database
    print("Testing DuckDB connection...\n")

    conn = get_connection()

    print("\nSchema:")
    print(get_schema_description())

    print("\n\nSample queries:")

    print("\n1. Top 5 highest paid employees:")
    result = execute_query("""
        SELECT name, department, salary
        FROM employees
        ORDER BY salary DESC
        LIMIT 5
    """)
    print(result)

    print("\n2. Total sales by region:")
    result = execute_query("""
        SELECT region, SUM(total_amount) as total_sales, COUNT(*) as num_orders
        FROM sales
        GROUP BY region
        ORDER BY total_sales DESC
    """)
    print(result)

    print("\n3. Products with low stock:")
    result = execute_query("""
        SELECT name, category, stock_quantity
        FROM products
        WHERE stock_quantity < 50
        ORDER BY stock_quantity
    """)
    print(result)
