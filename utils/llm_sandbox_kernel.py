"""
Docker-based sandbox using the llm-sandbox library.

This module provides a sandbox implementation that runs code in isolated
Docker containers, with fallback to the multiprocessing sandbox when
Docker is unavailable.
"""

import base64
import json
import logging
import os
import re
import tempfile
import uuid
from typing import Any, Dict, Optional

from llm_sandbox import SandboxSession, SandboxBackend

from utils.sandbox_interface import SandboxInterface

logger = logging.getLogger(__name__)


# Helper code to inject into the sandbox for display functions and artifact handling
SANDBOX_INIT_CODE = '''
import sys
import io
import json
import math
import statistics
import base64
import uuid
import re
from decimal import Decimal

# Data analysis libraries
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
    import polars as pl
    HAS_POLARS = True
except ImportError:
    pl = None
    HAS_POLARS = False

try:
    import altair as alt
    HAS_ALTAIR = True
except ImportError:
    alt = None
    HAS_ALTAIR = False

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

# Global artifact storage
_artifacts = {}
_seen_plots = set()

def _gen_id(prefix):
    return f"{prefix}_{str(uuid.uuid4())[:8]}"

def _is_safe_artifact_id(value):
    return bool(re.fullmatch(r"[A-Za-z0-9_]+", value or ""))

def _sanitize_vl_spec(value):
    """Convert non-JSON types in Vega-Lite specs to safe primitives."""
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

_MERMAID_PREFIXES = (
    "graph", "flowchart", "sequenceDiagram", "classDiagram",
    "stateDiagram", "erDiagram", "journey", "gantt", "pie",
    "mindmap", "timeline", "quadrantChart", "requirementDiagram", "gitGraph",
)

def _looks_like_mermaid(text):
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

class Markdown:
    """Mock for IPython.display.Markdown."""
    def __init__(self, data):
        self.data = data
    def __str__(self):
        return str(self.data)
    def _repr_markdown_(self):
        return str(self.data)

def display(obj, type=None, label=None):
    """Generic display function to render objects as artifacts."""
    global _artifacts, _seen_plots

    art_id = label if (label and _is_safe_artifact_id(label)) else None
    label_text = label if (label and not _is_safe_artifact_id(label)) else None

    if isinstance(obj, Markdown):
        obj = obj.data
        if type is None:
            type = "markdown"

    if type is None:
        if HAS_ALTAIR and isinstance(obj, alt.TopLevelMixin):
            type = "plot"
        elif (HAS_PANDAS and isinstance(obj, pd.DataFrame)) or \\
             (HAS_POLARS and isinstance(obj, pl.DataFrame)):
            type = "table"
        elif hasattr(obj, '_repr_html_'):
            type = "html"
        else:
            type = "text"

    if (type == "text" or type == "markdown") and _looks_like_mermaid(obj):
        type = "mermaid"

    content = None
    tag_type = type

    if type == "plot":
        if HAS_ALTAIR and HAS_VL_CONVERT and isinstance(obj, alt.TopLevelMixin):
            try:
                vl_spec = _sanitize_vl_spec(obj.to_dict())
                svg_data = vlc.vegalite_to_svg(vl_spec=vl_spec)
                content = base64.b64encode(svg_data.encode('utf-8')).decode('utf-8')
                _seen_plots.add(id(obj))
            except Exception as e:
                print(f"Error converting plot: {e}")
                return
        else:
            print(f"Unsupported plot object: {type(obj)}")
            return

    elif type == "table":
        if HAS_POLARS and isinstance(obj, pl.DataFrame):
            df_html = obj.to_pandas()
        elif hasattr(obj, 'to_pandas'):
            df_html = obj.to_pandas()
        else:
            df_html = obj

        if hasattr(df_html, 'to_html'):
            content = df_html.to_html(index=False, classes='data-table', border=0, na_rep='-')
            if label:
                content = f'<div class="table-title">{label}</div>\\n{content}'
                art_id = _gen_id("table")
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
        if type == "plot" and not art_id:
            art_id = _gen_id("plot")
        elif type == "table" and not art_id:
            art_id = _gen_id("table")
        elif not art_id:
            art_id = _gen_id(type)

        _artifacts[art_id] = {"type": tag_type, "content": content}
        if label_text and type != "table":
            print(label_text)
        print(f"\\n```{tag_type} {art_id}```\\n")

def show_table(df, title=None):
    """Display a DataFrame as a table artifact."""
    display(df, type='table', label=title)

def show_html(content):
    """Display HTML content as an artifact."""
    display(content, type='html')

def print_md(obj):
    """Print object as markdown and create artifact."""
    if (HAS_PANDAS and isinstance(obj, pd.DataFrame)) or \\
       (HAS_POLARS and isinstance(obj, pl.DataFrame)):
        try:
            if HAS_POLARS and isinstance(obj, pl.DataFrame):
                md_content = obj.to_pandas().to_markdown(index=False)
            else:
                md_content = obj.to_markdown(index=False)
            print(md_content)
            display(md_content, type='markdown')
        except Exception:
            content = str(obj)
            print(content)
            display(content, type='markdown')
    else:
        content = str(obj)
        print(content)
        display(content, type='markdown')

# Safe describe function that works in Docker (polars describe crashes in some container environments)
def safe_describe(df):
    """Return descriptive statistics for a DataFrame, using pandas to avoid polars crashes."""
    if HAS_POLARS and isinstance(df, pl.DataFrame):
        return df.to_pandas().describe()
    elif HAS_PANDAS and isinstance(df, pd.DataFrame):
        return df.describe()
    else:
        raise TypeError("Input must be a pandas or polars DataFrame")

# Monkey-patch polars DataFrame.describe to avoid crashes in Docker
if HAS_POLARS:
    _original_pl_describe = pl.DataFrame.describe
    def _safe_pl_describe(self, *args, **kwargs):
        """Safe describe that uses pandas to avoid segfaults in Docker containers."""
        try:
            return self.to_pandas().describe()
        except Exception:
            # If pandas conversion fails, try original (may crash)
            return _original_pl_describe(self, *args, **kwargs)
    pl.DataFrame.describe = _safe_pl_describe

def _capture_plots():
    """Capture all undisplayed Altair plots from globals."""
    global _seen_plots, _artifacts
    if HAS_ALTAIR and HAS_VL_CONVERT:
        for name, val in list(globals().items()):
            if name.startswith('_'):
                continue
            if isinstance(val, alt.TopLevelMixin):
                if id(val) in _seen_plots:
                    continue
                try:
                    vl_spec = _sanitize_vl_spec(val.to_dict())
                    svg_data = vlc.vegalite_to_svg(vl_spec=vl_spec)
                    b64_svg = base64.b64encode(svg_data.encode('utf-8')).decode('utf-8')
                    art_id = _gen_id("plot")
                    _artifacts[art_id] = {"type": "plot", "content": b64_svg}
                    _seen_plots.add(id(val))
                    print(f"\\n```plot {art_id}```\\n")
                except Exception:
                    pass

# Database setup with sample data
_db_conn = None

def _init_database():
    global _db_conn
    if not HAS_DUCKDB:
        return

    _db_conn = duckdb.connect(':memory:')

    # Create sample tables
    _db_conn.execute("""
        CREATE TABLE employees (
            id INTEGER PRIMARY KEY,
            name VARCHAR,
            department VARCHAR,
            salary DECIMAL(10,2),
            hire_date DATE,
            is_manager BOOLEAN
        )
    """)

    _db_conn.execute("""
        INSERT INTO employees VALUES
        (1, 'Alice Johnson', 'Engineering', 95000, '2020-03-15', true),
        (2, 'Bob Smith', 'Engineering', 85000, '2021-06-01', false),
        (3, 'Carol Williams', 'Sales', 75000, '2019-11-20', true),
        (4, 'David Brown', 'Sales', 65000, '2022-01-10', false),
        (5, 'Eva Martinez', 'Marketing', 70000, '2021-08-05', false),
        (6, 'Frank Lee', 'Engineering', 90000, '2020-09-12', false),
        (7, 'Grace Kim', 'HR', 60000, '2022-04-18', true),
        (8, 'Henry Wilson', 'Marketing', 72000, '2021-02-28', true),
        (9, 'Iris Chen', 'Engineering', 88000, '2020-12-01', false),
        (10, 'Jack Davis', 'Sales', 68000, '2022-07-22', false)
    """)

    _db_conn.execute("""
        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name VARCHAR,
            category VARCHAR,
            price DECIMAL(10,2),
            stock_quantity INTEGER,
            supplier VARCHAR
        )
    """)

    _db_conn.execute("""
        INSERT INTO products VALUES
        (1, 'Laptop Pro', 'Electronics', 1299.99, 50, 'TechCorp'),
        (2, 'Wireless Mouse', 'Electronics', 29.99, 200, 'TechCorp'),
        (3, 'Office Chair', 'Furniture', 249.99, 75, 'ComfortPlus'),
        (4, 'Standing Desk', 'Furniture', 599.99, 30, 'ComfortPlus'),
        (5, 'Monitor 27"', 'Electronics', 399.99, 100, 'DisplayMax'),
        (6, 'Keyboard Mechanical', 'Electronics', 149.99, 150, 'TechCorp'),
        (7, 'Desk Lamp', 'Furniture', 45.99, 200, 'LightWorks'),
        (8, 'Webcam HD', 'Electronics', 79.99, 120, 'TechCorp'),
        (9, 'Notebook Pack', 'Supplies', 12.99, 500, 'OfficeMart'),
        (10, 'Pen Set', 'Supplies', 8.99, 1000, 'OfficeMart')
    """)

    _db_conn.execute("""
        CREATE TABLE sales (
            id INTEGER PRIMARY KEY,
            product_id INTEGER,
            employee_id INTEGER,
            quantity INTEGER,
            sale_date DATE,
            total_amount DECIMAL(10,2),
            region VARCHAR
        )
    """)

    _db_conn.execute("""
        INSERT INTO sales VALUES
        (1, 1, 3, 2, '2024-01-15', 2599.98, 'North'),
        (2, 2, 4, 10, '2024-01-16', 299.90, 'South'),
        (3, 3, 3, 5, '2024-01-17', 1249.95, 'North'),
        (4, 5, 10, 3, '2024-01-18', 1199.97, 'East'),
        (5, 6, 4, 8, '2024-01-19', 1199.92, 'South'),
        (6, 1, 10, 1, '2024-01-20', 1299.99, 'East'),
        (7, 4, 3, 2, '2024-01-21', 1199.98, 'North'),
        (8, 7, 4, 15, '2024-01-22', 689.85, 'South'),
        (9, 8, 10, 6, '2024-01-23', 479.94, 'East'),
        (10, 9, 3, 50, '2024-01-24', 649.50, 'North'),
        (11, 2, 4, 20, '2024-01-25', 599.80, 'South'),
        (12, 5, 3, 4, '2024-01-26', 1599.96, 'North'),
        (13, 6, 10, 5, '2024-01-27', 749.95, 'East'),
        (14, 10, 4, 100, '2024-01-28', 899.00, 'South'),
        (15, 1, 3, 3, '2024-01-29', 3899.97, 'North')
    """)

    _db_conn.execute("""
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY,
            name VARCHAR,
            email VARCHAR,
            city VARCHAR,
            country VARCHAR,
            signup_date DATE,
            total_purchases DECIMAL(10,2)
        )
    """)

    _db_conn.execute("""
        INSERT INTO customers VALUES
        (1, 'Acme Corp', 'contact@acme.com', 'New York', 'USA', '2023-01-10', 15000.00),
        (2, 'GlobalTech', 'info@globaltech.com', 'London', 'UK', '2023-02-15', 22000.00),
        (3, 'StartupXYZ', 'hello@startupxyz.com', 'San Francisco', 'USA', '2023-03-20', 8500.00),
        (4, 'MegaStore', 'sales@megastore.com', 'Toronto', 'Canada', '2023-04-05', 31000.00),
        (5, 'TechGiant', 'support@techgiant.com', 'Berlin', 'Germany', '2023-05-12', 18000.00),
        (6, 'LocalShop', 'owner@localshop.com', 'Chicago', 'USA', '2023-06-18', 5200.00),
        (7, 'BigRetail', 'orders@bigretail.com', 'Sydney', 'Australia', '2023-07-22', 27000.00),
        (8, 'SmallBiz', 'contact@smallbiz.com', 'Paris', 'France', '2023-08-30', 4100.00),
        (9, 'EnterpriseInc', 'procurement@enterprise.com', 'Tokyo', 'Japan', '2023-09-14', 45000.00),
        (10, 'QuickMart', 'buy@quickmart.com', 'Miami', 'USA', '2023-10-25', 12000.00)
    """)

_init_database()

def query_db(sql):
    """Execute SQL query and return result as DataFrame."""
    if _db_conn is None:
        raise RuntimeError("Database not available")
    if HAS_POLARS:
        try:
            df = _db_conn.execute(sql).pl()
            # Cast Decimal columns to Float64 to avoid segfaults in describe() and other operations
            for col in df.columns:
                if df[col].dtype == pl.Decimal:
                    df = df.with_columns(pl.col(col).cast(pl.Float64))
                elif str(df[col].dtype).startswith("Decimal"):
                    df = df.with_columns(pl.col(col).cast(pl.Float64))
            return df
        except Exception:
            # Fall back to pandas if polars conversion fails (e.g., missing pyarrow)
            return _db_conn.execute(sql).df()
    else:
        return _db_conn.execute(sql).df()

_saved_tables = {}

def save_table(df, name, replace=True):
    """Save a DataFrame for later use."""
    _saved_tables[name] = df
    return True

def load_table(name):
    """Load a previously saved DataFrame."""
    if name not in _saved_tables:
        raise KeyError(f"Table '{name}' not found")
    return _saved_tables[name]

def list_saved_tables():
    """List all saved tables."""
    return list(_saved_tables.keys())

def delete_table(name):
    """Delete a saved table."""
    if name in _saved_tables:
        del _saved_tables[name]
        return True
    return False

print("Sandbox environment initialized")
'''

# Post-execution code to capture artifacts
EXECUTION_WRAPPER_POST = '''
# User code ends here

# Capture any undisplayed plots
_capture_plots()

# Output artifacts as JSON marker
print("##ARTIFACTS_START##")
print(json.dumps(_artifacts))
print("##ARTIFACTS_END##")
'''


def _parse_artifacts_from_output(stdout: str) -> tuple[dict, str]:
    """Parse artifacts JSON from stdout and return clean stdout.

    Returns:
        Tuple of (artifacts_dict, clean_stdout)
    """
    artifacts = {}
    clean_stdout = stdout

    # Extract artifacts JSON block
    match = re.search(r'##ARTIFACTS_START##\n(.*?)\n##ARTIFACTS_END##', stdout, re.DOTALL)
    if match:
        try:
            artifacts = json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
        clean_stdout = stdout[:match.start()] + stdout[match.end():]

    return artifacts, clean_stdout.strip()


class LLMSandboxKernel(SandboxInterface):
    """Docker-based sandbox implementation using llm-sandbox library."""

    def __init__(self, timeout_seconds: int = 30, image: Optional[str] = None):
        """Initialize the Docker sandbox.

        Args:
            timeout_seconds: Execution timeout in seconds.
            image: Custom Docker image to use (optional).
        """
        logger.info("[DOCKER-SANDBOX] Initializing LLMSandboxKernel...")
        logger.info(f"[DOCKER-SANDBOX]   timeout_seconds: {timeout_seconds}")
        logger.info(f"[DOCKER-SANDBOX]   custom_image: {image or '(using default)'}")
        self.timeout_seconds = timeout_seconds
        self.image = image
        self._session: Optional[SandboxSession] = None
        self._initialized = False
        self._execution_count = 0
        self._start()

    @property
    def backend_name(self) -> str:
        return "docker"

    def _start(self) -> None:
        """Start the Docker sandbox session."""
        logger.info("[DOCKER-SANDBOX] Starting Docker sandbox session...")
        if self._session:
            logger.info("[DOCKER-SANDBOX] Cleaning up previous session...")
            try:
                self._session.__exit__(None, None, None)
            except Exception as e:
                logger.warning(f"[DOCKER-SANDBOX] Error cleaning up previous session: {e}")

        # Create session with configuration
        session_kwargs = {
            "lang": "python",
            "backend": SandboxBackend.DOCKER,
            "verbose": False,
            "keep_template": True,  # Keep container for session persistence
        }

        if self.image:
            session_kwargs["image"] = self.image

        logger.info(f"[DOCKER-SANDBOX] Creating SandboxSession with: backend=DOCKER, lang=python, keep_template=True")
        if self.image:
            logger.info(f"[DOCKER-SANDBOX]   custom image: {self.image}")

        self._session = SandboxSession(**session_kwargs)
        logger.info("[DOCKER-SANDBOX] Entering Docker container context...")
        self._session.__enter__()
        self._initialized = False
        logger.info("[DOCKER-SANDBOX] Docker sandbox session started successfully!")

    def _ensure_initialized(self) -> None:
        """Ensure the sandbox environment is initialized."""
        if self._initialized:
            logger.debug("[DOCKER-SANDBOX] Environment already initialized")
            return

        logger.info("[DOCKER-SANDBOX] Initializing sandbox environment inside Docker container...")

        # Check if we should skip library installation (pre-built image)
        skip_install = os.environ.get("SANDBOX_SKIP_INSTALL", "").lower() == "true"
        if self.image and not skip_install:
            # Auto-detect pre-built images by name pattern
            prebuilt_patterns = ["analytical-chatbot-sandbox", "sandbox-prebuilt", "-sandbox:"]
            if any(pattern in self.image for pattern in prebuilt_patterns):
                skip_install = True
                logger.info(f"[DOCKER-SANDBOX] Detected pre-built image '{self.image}', skipping library installation")

        if skip_install:
            # Pre-built image: just run init code without installing libraries
            logger.info("[DOCKER-SANDBOX] Using pre-built image - skipping pip install (libraries already present)")
            try:
                result = self._session.run(SANDBOX_INIT_CODE)
                if result.exit_code != 0:
                    logger.warning(f"[DOCKER-SANDBOX] Initialization had errors: {result.stderr}")
                else:
                    logger.info("[DOCKER-SANDBOX] Environment initialized successfully (pre-built image)")
                self._initialized = True
            except Exception as e:
                logger.error(f"[DOCKER-SANDBOX] Initialization failed: {e}")
            return

        # Standard initialization: install libraries
        libraries = ["pandas", "numpy", "polars", "pyarrow", "altair", "vl-convert-python", "duckdb", "tabulate"]
        logger.info(f"[DOCKER-SANDBOX] Installing libraries in container: {', '.join(libraries)}")
        logger.info("[DOCKER-SANDBOX] (This may take 30-60 seconds on first run...)")

        try:
            result = self._session.run(SANDBOX_INIT_CODE, libraries=libraries)
            if result.exit_code != 0:
                logger.warning(f"[DOCKER-SANDBOX] Initialization had errors: {result.stderr}")
            else:
                logger.info("[DOCKER-SANDBOX] Libraries installed and environment initialized successfully")
            self._initialized = True
        except Exception as e:
            logger.warning(f"[DOCKER-SANDBOX] Failed to initialize with all libraries: {e}")
            # Try again without all libraries
            logger.info("[DOCKER-SANDBOX] Retrying initialization without external libraries...")
            try:
                result = self._session.run(SANDBOX_INIT_CODE)
                self._initialized = True
                logger.info("[DOCKER-SANDBOX] Basic initialization succeeded (some libraries may be missing)")
            except Exception as e2:
                logger.error(f"[DOCKER-SANDBOX] Basic initialization also failed: {e2}")

    def execute(self, code: str) -> Dict[str, Any]:
        """Execute code in the Docker sandbox.

        Args:
            code: Python code to execute.

        Returns:
            Dictionary with success, stdout, stderr, artifacts, and error.
        """
        self._execution_count += 1
        exec_id = self._execution_count

        logger.info(f"[DOCKER-SANDBOX] ========== Execution #{exec_id} ==========")
        logger.info(f"[DOCKER-SANDBOX] Executing code in Docker container...")
        logger.debug(f"[DOCKER-SANDBOX] Code preview (first 200 chars): {code[:200]}...")

        if not self._session:
            logger.info("[DOCKER-SANDBOX] No active session, starting new one...")
            self._start()

        self._ensure_initialized()

        # Prepend full init code + user code + artifact extraction
        # Each run() is a fresh interpreter, so we need all definitions every time
        wrapped_code = SANDBOX_INIT_CODE + "\n# User code starts here\n" + code + EXECUTION_WRAPPER_POST

        try:
            logger.info(f"[DOCKER-SANDBOX] Running code with timeout={self.timeout_seconds}s")
            result = self._session.run(wrapped_code, timeout=self.timeout_seconds)

            stdout = result.stdout or ""
            stderr = result.stderr or ""
            success = result.exit_code == 0

            # Extract artifacts from stdout
            artifacts, clean_stdout = _parse_artifacts_from_output(stdout)

            # Remove initialization message from output
            clean_stdout = clean_stdout.replace("Sandbox environment initialized\n", "")
            clean_stdout = clean_stdout.strip()

            logger.info(f"[DOCKER-SANDBOX] Execution #{exec_id} completed: success={success}, exit_code={result.exit_code}")
            if artifacts:
                logger.info(f"[DOCKER-SANDBOX] Artifacts generated: {list(artifacts.keys())}")
            if clean_stdout:
                logger.debug(f"[DOCKER-SANDBOX] Output preview: {clean_stdout[:200]}...")
            if stderr:
                logger.warning(f"[DOCKER-SANDBOX] Stderr: {stderr[:500]}...")
            logger.info(f"[DOCKER-SANDBOX] ========================================")

            return {
                "success": success,
                "stdout": clean_stdout,
                "stderr": stderr,
                "artifacts": artifacts,
                "error": None if success else f"Exit code: {result.exit_code}\n{stderr}"
            }

        except Exception as e:
            error_msg = str(e)
            logger.error(f"[DOCKER-SANDBOX] Execution #{exec_id} failed with exception: {error_msg}")
            # Handle timeout
            if "timeout" in error_msg.lower():
                logger.warning(f"[DOCKER-SANDBOX] Execution timed out after {self.timeout_seconds}s, restarting kernel...")
                self.restart()
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": "",
                    "artifacts": {},
                    "error": f"Execution timed out ({self.timeout_seconds}s). Kernel restarted."
                }

            return {
                "success": False,
                "stdout": "",
                "stderr": error_msg,
                "artifacts": {},
                "error": error_msg
            }

    def load_file(self, filename: str, arrow_data: bytes) -> bool:
        """Load Arrow data into the sandbox.

        Args:
            filename: Name of the file (used as variable name base).
            arrow_data: Binary Arrow IPC stream data.

        Returns:
            True if loading succeeded, False otherwise.
        """
        logger.info(f"[DOCKER-SANDBOX] Loading file into container: {filename} ({len(arrow_data)} bytes)")

        if not self._session:
            self._start()

        self._ensure_initialized()

        try:
            # Write Arrow data to a temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.arrow') as f:
                f.write(arrow_data)
                temp_path = f.name
            logger.debug(f"[DOCKER-SANDBOX] Wrote temp file: {temp_path}")

            # Copy file to container
            container_path = f"/tmp/{filename}"
            logger.info(f"[DOCKER-SANDBOX] Copying file to container: {container_path}")
            self._session.copy_to_runtime(temp_path, container_path)

            # Clean up local temp file
            os.unlink(temp_path)

            # Load in sandbox
            var_name = filename.rsplit('.', 1)[0].replace(' ', '_').replace('-', '_')
            load_code = f'''
import polars as pl
{var_name} = pl.read_ipc_stream("{container_path}")
print(f"Loaded {{len({var_name})}} rows into {var_name}")
'''
            result = self._session.run(load_code)
            success = result.exit_code == 0
            if success:
                logger.info(f"[DOCKER-SANDBOX] File loaded successfully as variable '{var_name}'")
            else:
                logger.error(f"[DOCKER-SANDBOX] Failed to load file: {result.stderr}")
            return success

        except Exception as e:
            logger.error(f"[DOCKER-SANDBOX] Error loading file: {e}")
            return False

    def restart(self) -> None:
        """Restart the sandbox (new container)."""
        logger.info("[DOCKER-SANDBOX] Restarting Docker sandbox (creating new container)...")
        self._execution_count = 0
        self._start()

    def terminate(self) -> None:
        """Terminate the sandbox session."""
        logger.info("[DOCKER-SANDBOX] Terminating Docker sandbox session...")
        if self._session:
            try:
                self._session.__exit__(None, None, None)
                logger.info("[DOCKER-SANDBOX] Session terminated successfully")
            except Exception as e:
                logger.warning(f"[DOCKER-SANDBOX] Error during termination: {e}")
            self._session = None
            self._initialized = False
        else:
            logger.debug("[DOCKER-SANDBOX] No active session to terminate")
