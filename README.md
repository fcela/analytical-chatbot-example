# Analytical Chatbot (PocketFlow + FastAPI + React)

A web-based chatbot built with PocketFlow that can analyze uploaded CSV/JSON files and execute Python code in a sandboxed environment.

## Architecture

- **Backend**: FastAPI (`api.py`) with PocketFlow for LLM orchestration
- **Frontend**: React + Vite + TypeScript
- **LLM Providers**: OpenAI, Anthropic (Claude), or Ollama (local)
- **Sandbox**: Thread-based Python execution with timeout and security restrictions

## Project Structure

```
chatbot/
├── api.py              # FastAPI endpoints
├── main.py             # Entry point (runs uvicorn)
├── nodes.py            # PocketFlow node definitions
├── flow.py             # Flow orchestration
├── utils/
│   ├── call_llm.py     # LLM wrapper with retry/mock support
│   ├── sandbox.py      # Sandboxed code execution
│   └── parse_code.py   # Code block extraction
├── frontend/
│   ├── src/
│   │   ├── App.tsx     # React app
│   │   ├── main.tsx    # Entry point
│   │   └── style.css   # Styles
│   ├── package.json
│   └── vite.config.ts
├── docs/
│   └── documentation.html  # Technical documentation
├── Makefile
└── requirements.txt
```

## Quickstart

### 1. Install dependencies

```bash
# Backend
pip install -r requirements.txt

# Frontend
cd frontend && npm install && cd ..
```

### 2. Set environment variables

Choose your LLM provider:

```bash
# Option A: OpenAI (default)
export LLM_PROVIDER='openai'
export OPENAI_API_KEY='your-key'

# Option B: Anthropic (Claude)
export LLM_PROVIDER='anthropic'
export ANTHROPIC_API_KEY='your-key'

# Option C: Ollama (local, no API key needed)
export LLM_PROVIDER='ollama'
export OLLAMA_MODEL='qwen3:4b'  # or any model you have installed

# Option D: Mock mode (for testing without any LLM)
export MOCK_LLM='true'
```

### 3. Start the servers

**Option A: Using Make**
```bash
make start-backend   # Terminal 1: FastAPI on port 8000
make start-frontend  # Terminal 2: Vite on port 5173
```

**Option B: Manual**
```bash
# Terminal 1
python main.py

# Terminal 2
cd frontend && npm run dev
```

### 4. Open the app

Navigate to http://localhost:5173 in your browser.

## Environment Variables

### LLM Provider Selection

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | Provider to use (`openai`, `anthropic`, `ollama`) | `openai` |
| `MOCK_LLM` | Force mock responses for testing | `false` |
| `LLM_TEMPERATURE` | Temperature for generation | `0.7` |
| `LLM_DEBUG` | Enable debug logging | `false` |

### Provider-Specific Variables

**OpenAI:**
| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `OPENAI_MODEL` | Model to use | `gpt-4o` |

**Anthropic (Claude):**
| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Anthropic API key | Required |
| `ANTHROPIC_MODEL` | Model to use | `claude-sonnet-4-20250514` |
| `ANTHROPIC_MAX_TOKENS` | Max tokens for response | `4096` |

**Ollama (Local):**
| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_MODEL` | Model to use | `qwen3:4b` |
| `LLM_TIMEOUT` | Timeout for generation (seconds) | `120` |

### Server Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `HOST` | Backend host | `0.0.0.0` |
| `PORT` | Backend port | `8000` |

## Features

- **Intent Classification**: Automatically detects if user wants conversation or code execution
- **Code Generation**: Generates Python code for data analysis tasks (preferring Polars and Altair)
- **Sandboxed Execution**: Runs code with security restrictions and timeout
- **Formatted Tables**: Display DataFrames as styled HTML tables using `show_table()`
- **Interactive Dashboards**: Create responsive HTML dashboards using `show_html()`
- **Mermaid Diagrams**: Support for rendering flowcharts, sequence diagrams, and more via Mermaid syntax
- **High-Quality Plots**: Renders Altair charts as crisp SVG vectors
- **File Upload**: Supports CSV and JSON file uploads
- **Built-in Database**: Query pre-loaded DuckDB tables (employees, products, sales, customers)
- **Session Management**: Cookie-based sessions with in-memory storage
- **Clean UI**: Markdown support and collapsible code/output blocks for a better chat experience

## Sandbox Environment

The sandbox provides a restricted execution environment with pre-loaded data science tools:
- **Polars** (`pl`): High-performance data manipulation (preferred)
- **Pandas** (`pd`): Traditional DataFrame library
- **NumPy** (`np`): Numerical computing
- **Altair** (`alt`): Declarative statistical visualization
- **Statsmodels** (`sm`, `smf`): Statistical modeling and econometrics
- **DuckDB**: SQL query interface via `query_db()` function
- **Helper Functions**: `show_table()` for HTML tables, `show_html()` for dashboards
- **Standard Libs**: `math`, `statistics`, `json`

Security restrictions prevent:
- File system access (`open`, `os`, `shutil`)
- Network access (`socket`, `requests`, `urllib`)
- Dangerous operations (`exec`, `eval`, `__import__`)
- Long-running code (30-second timeout)

## Development

See `docs/documentation.html` for detailed architecture documentation.
