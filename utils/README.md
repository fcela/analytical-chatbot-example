# Analytical Chatbot (A2A Agent + PocketFlow + React)

A web-based chatbot built with PocketFlow that can analyze uploaded CSV/JSON files and execute Python code in a sandboxed environment. Now available as an **A2A (Agent-to-Agent) compatible agent** that can communicate with other AI agents using Google's [A2A Protocol](https://-protocol.org/).

## Architecture

- **Backend**: A2A Protocol server with HTTP/JSON-RPC and gRPC interfaces
- **Agent Core**: PocketFlow for LLM orchestration
- **Frontend**: React + Vite + TypeScript (communicates via legacy REST → A2A translation layer)
- **Sandbox**: Thread-based Python execution with timeout and security restrictions

```
┌─────────────┐     ┌──────────────────────────────────────────────┐
│   Frontend  │     │              A2A Server                      │
│   (React)   │────▶│  ┌─────────────┐    ┌───────────────────┐   │
└─────────────┘     │  │ Legacy REST │───▶│  A2A Protocol     │   │
                    │  │  Endpoints  │    │  Request Handler  │   │
┌─────────────┐     │  └─────────────┘    └─────────┬─────────┘   │
│ A2A Client  │────▶│  ┌─────────────┐              │             │
│ (JSON-RPC)  │     │  │   /      │──────────────┤             │
└─────────────┘     │  └─────────────┘              ▼             │
                    │                    ┌───────────────────┐    │
┌─────────────┐     │                    │  AgentExecutor    │    │
│ gRPC Client │────▶│  ┌─────────────┐   │  (PocketFlow)     │    │
└─────────────┘     │  │   :50051    │──▶└───────────────────┘    │
                    │  └─────────────┘                            │
                    └──────────────────────────────────────────────┘
```

## A2A Protocol Support

This agent implements the [A2A Protocol](https://-protocol.org/) (v0.3), enabling:

- **Agent Discovery**: Agent card available at `/.well-known/agent-card.json`
- **Dual Transport**: Both HTTP/JSON-RPC and gRPC interfaces
- **Streaming**: Real-time task status updates
- **Multi-turn Conversations**: Context preserved across interactions
- **Rich Artifacts**: Code, visualizations, and structured data in responses

## Project Structure

```
analytical-chatbot-example/
├── main.py                 # Entry point (runs A2A server)
├── a2a_server.py           # A2A Protocol server setup (HTTP + gRPC)
├── agent_executor.py       # A2A AgentExecutor wrapping the PocketFlow
├── legacy_routes.py        # REST → A2A translation layer for frontend
├── examples/
│   └── a2a_client_example.py  # Example A2A client script
├── utils/
│   ├── api.py              # Standalone FastAPI server (backwards compat)
│   ├── nodes.py            # PocketFlow node definitions
│   ├── flow.py             # Flow orchestration
│   ├── call_llm.py         # LLM wrapper with retry/mock support
│   ├── sandbox.py          # Sandboxed code execution
│   ├── parse_code.py       # Code block extraction
│   ├── database.py         # DuckDB database utilities
│   ├── frontend/
│   │   ├── src/
│   │   │   ├── App.tsx     # React app
│   │   │   ├── main.tsx    # Entry point
│   │   │   └── style.css   # Styles
│   │   ├── package.json
│   │   └── vite.config.ts
│   ├── docs/
│   │   └── design.md       # Design documentation
│   ├── Makefile
│   └── requirements.txt
```

## Quickstart

### 1. Install dependencies

```bash
# Backend (from project root)
pip install -r utils/requirements.txt

# Frontend
cd utils/frontend && npm install && cd ../..
```

### 2. Set environment variables (optional)

```bash
export OPENAI_API_KEY='your-key'  # Optional: runs in mock mode without it
```

### 3. Start the servers

**Option A: From project root**
```bash
# Terminal 1: A2A server on port 8000 (HTTP) and 50051 (gRPC)
python main.py

# Terminal 2: Frontend on port 5173
cd utils/frontend && npm run dev
```

**Option B: Using Make (from utils/ directory)**
```bash
cd utils
make start-backend   # A2A server
make start-frontend  # Vite dev server
```

**Option C: Legacy FastAPI (backwards compatible)**
```bash
cd utils
make start-backend-legacy  # Runs the original FastAPI server
```

### 4. Open the app

- **Web UI**: Navigate to http://localhost:5173 in your browser
- **Agent Card**: View at http://localhost:8000/.well-known/agent-card.json
- **A2A Endpoint**: Send JSON-RPC requests to http://localhost:8000/
- **gRPC Endpoint**: Connect to localhost:50051

### 5. Try the A2A example client (optional)

```bash
# From the project root directory
python examples/_client_example.py
```

This demonstrates discovering the agent, sending messages, and handling artifacts via the A2A protocol.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | None (mock mode) |
| `OPENAI_MODEL` | Model to use | `gpt-4o` |
| `MOCK_LLM` | Force mock responses | `false` |
| `HOST` | Backend host | `0.0.0.0` |
| `PORT` | HTTP port | `8000` |
| `GRPC_PORT` | gRPC port | `50051` |
| `ENABLE_GRPC` | Enable gRPC server | `true` |
| `A2A_BASE_URL` | Override agent card URL | Auto-generated |

## Features

- **Intent Classification**: Automatically detects if user wants conversation or code execution
- **Code Generation**: Generates Python code for data analysis tasks (using Polars and Altair)
- **Sandboxed Execution**: Runs code with security restrictions and timeout
- **Interactive Dashboards**: Create responsive HTML dashboards using `show_html()`
- **Mermaid Diagrams**: Support for rendering flowcharts, sequence diagrams, and more via Mermaid syntax
- **High-Quality Plots**: Renders Altair charts as crisp SVG vectors
- **File Upload**: Supports CSV and JSON file uploads
- **Built-in Database**: Query pre-loaded DuckDB tables (employees, products, sales, customers)
- **Session Management**: Cookie-based sessions with in-memory storage
- **Clean UI**: Markdown support and collapsible code/output blocks for a better chat experience

## Sandbox Environment

The sandbox provides a restricted execution environment with pre-loaded data science tools:
- **Polars**: High-performance data manipulation
- **Altair**: Declarative statistical visualization (v6.0+)
- **DuckDB**: SQL query interface to built-in and saved tables
- **Standard Libs**: `math`, `statistics`, `json`

Security restrictions prevent:
- File system access (`open`, `os`, `shutil`)
- Network access (`socket`, `requests`, `urllib`)
- Dangerous operations (`exec`, `eval`, `__import__`)
- Long-running code (30-second timeout)

## A2A Integration

### How It Works

The application exposes three interfaces, all routing through the same A2A AgentExecutor:

| Interface | Endpoint | Use Case |
|-----------|----------|----------|
| **Legacy REST** | `/chat`, `/upload`, `/files` | React frontend compatibility |
| **JSON-RPC** | `/` | A2A protocol clients |
| **gRPC** | `:50051` | High-performance A2A clients |

**Request Flow Example:**

When the frontend sends a chat message:

```
1. Frontend POST /chat {"message": "What is the average salary?"}
          │
          ▼
2. legacy_routes.py translates to A2A Message:
   Message(
     role="user",
     parts=[TextPart(text="What is the average salary?")],
     contextId="session-uuid"
   )
          │
          ▼
3. A2A DefaultRequestHandler.on_message_send()
          │
          ▼
4. AnalyticalChatbotExecutor.execute()
   - Extracts text from A2A message
   - Calls run_chatbot() (PocketFlow)
   - Emits A2A events: TaskStatusUpdateEvent, TaskArtifactUpdateEvent
          │
          ▼
5. legacy_routes.py translates A2A response back to REST:
   {"response": {"message": "The average salary is $75,000", "code": "...", ...}}
```

### Agent Card

The agent advertises its capabilities via the [Agent Card](https://-protocol.org/latest/specification/#agent-card):

```bash
curl http://localhost:8000/.well-known/agent-card.json
```

```json
{
  "name": "Analytical Chatbot",
  "description": "An AI-powered analytical assistant...",
  "version": "1.0.0",
  "capabilities": {
    "streaming": true,
    "pushNotifications": false,
    "stateTransitionHistory": true
  },
  "skills": [
    {"id": "data-analysis", "name": "Data Analysis", ...},
    {"id": "code-generation", "name": "Code Generation", ...},
    {"id": "visualization", "name": "Data Visualization", ...},
    {"id": "database-query", "name": "Database Querying", ...},
    {"id": "conversation", "name": "Conversation", ...}
  ],
  "defaultInputModes": ["text"],
  "defaultOutputModes": ["text"]
}
```

## A2A Client Examples

### Example 1: Python Client (Full Workflow)

```python
"""
Example: Multi-turn conversation with the Analytical Chatbot via A2A.

This demonstrates:
1. Discovering the agent via its Agent Card
2. Sending messages and receiving responses
3. Handling artifacts (code, plots, tables)
4. Multi-turn conversations with context
"""

import httpx
import json

BASE_URL = "http://localhost:8000"

def send_a2a_message(text: str, context_id: str = None) -> dict:
    """Send a message via A2A JSON-RPC protocol."""
    payload = {
        "jsonrpc": "2.0",
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"text": text}],
                **({"contextId": context_id} if context_id else {})
            }
        },
        "id": 1
    }

    response = httpx.post(f"{BASE_URL}/", json=payload)
    return response.json()

def extract_response(result: dict) -> dict:
    """Extract useful information from A2A response."""
    extracted = {"message": "", "code": None, "artifacts": []}

    if "result" not in result:
        return extracted

    task = result["result"]

    # Extract message from history
    for msg in task.get("history", []):
        if msg.get("role") == "agent":
            for part in msg.get("parts", []):
                if "text" in part:
                    extracted["message"] = part["text"]

    # Extract artifacts
    for artifact in task.get("artifacts", []):
        extracted["artifacts"].append({
            "name": artifact.get("name"),
            "parts": artifact.get("parts", [])
        })
        if artifact.get("name") == "generated_code":
            for part in artifact.get("parts", []):
                if "text" in part:
                    extracted["code"] = part["text"]

    return extracted

# 1. Discover the agent
print("=== Discovering Agent ===")
agent_card = httpx.get(f"{BASE_URL}/.well-known/agent-card.json").json()
print(f"Agent: {agent_card['name']}")
print(f"Skills: {[s['name'] for s in agent_card['skills']]}")

# 2. Start a conversation
print("\n=== Conversation ===")
context_id = "example-session-001"

# First message: greeting
response = send_a2a_message("Hello! What can you help me with?", context_id)
extracted = extract_response(response)
print(f"User: Hello! What can you help me with?")
print(f"Agent: {extracted['message'][:200]}...")

# Second message: data analysis request
response = send_a2a_message(
    "Show me the top 5 highest paid employees from the database",
    context_id
)
extracted = extract_response(response)
print(f"\nUser: Show me the top 5 highest paid employees")
print(f"Agent: {extracted['message'][:200]}...")
if extracted["code"]:
    print(f"Code generated:\n{extracted['code'][:300]}...")

# Third message: follow-up (uses context)
response = send_a2a_message("Now create a bar chart of their salaries", context_id)
extracted = extract_response(response)
print(f"\nUser: Now create a bar chart of their salaries")
print(f"Agent: {extracted['message'][:200]}...")
print(f"Artifacts: {[a['name'] for a in extracted['artifacts']]}")
```

### Example 2: Using the a2a-sdk

```python
"""Using the official A2A Python SDK."""

from a2a.client import A2AClient
from a2a.types import Message, Part, TextPart

# Create client
client = A2AClient(base_url="http://localhost:8000")

# Get agent card
agent_card = client.get_agent_card()
print(f"Connected to: {agent_card.name}")

# Send message
message = Message(
    role="user",
    parts=[Part(root=TextPart(text="Calculate the total sales by product category"))]
)

# For streaming responses
async for event in client.send_message_stream(message):
    if hasattr(event, 'status'):
        print(f"Status: {event.status.state}")
    elif hasattr(event, 'artifact'):
        print(f"Artifact: {event.artifact.name}")
```

### Example 3: gRPC Client

```bash
# Install grpcurl: brew install grpcurl

# Get agent capabilities
grpcurl -plaintext localhost:50051 a2a.A2AService/GetAgentCard

# Send a message (streaming response)
grpcurl -plaintext -d '{
  "message": {
    "role": "user",
    "parts": [{"text": {"text": "What tables are available in the database?"}}]
  }
}' localhost:50051 a2a.A2AService/SendStreamingMessage
```

### Example 4: curl (JSON-RPC)

```bash
# Simple message
curl -X POST http://localhost:8000/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "message/send",
    "params": {
      "message": {
        "role": "user",
        "parts": [{"text": "Create a pie chart of sales by region"}]
      }
    },
    "id": 1
  }' | jq .

# With context for multi-turn
curl -X POST http://localhost:8000/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "message/send",
    "params": {
      "message": {
        "role": "user",
        "parts": [{"text": "Now filter it to only show Q4 data"}],
        "contextId": "my-session-123"
      }
    },
    "id": 2
  }' | jq .
```

## Integrating with Other A2A Agents

This agent can be called by other A2A-compatible agents. Example orchestrator:

```python
"""Example: Orchestrator agent that delegates to the Analytical Chatbot."""

from a2a.client import A2AClient

# Discover available agents
analytical_agent = A2AClient(base_url="http://localhost:8000")
card = analytical_agent.get_agent_card()

# Check if agent has the skill we need
has_visualization = any(s.id == "visualization" for s in card.skills)

if has_visualization:
    # Delegate visualization task
    result = analytical_agent.send_message(
        Message(
            role="user",
            parts=[Part(root=TextPart(
                text="Create a dashboard showing sales trends for 2024"
            ))]
        )
    )

    # Extract the generated visualization
    for artifact in result.artifacts:
        if artifact.name.startswith("html_"):
            dashboard_html = artifact.parts[0].text
            # Use the dashboard in your application
```

## Development

### Running Tests

```bash
# Test the A2A endpoints
python -c "
import httpx
r = httpx.get('http://localhost:8000/.well-known/agent-card.json')
print('Agent Card:', r.json()['name'])
"

# Test legacy endpoints (frontend compatibility)
python -c "
import httpx
r = httpx.get('http://localhost:8000/')
print('Health:', r.json())
"
```

### Adding New Skills

1. Add the skill logic to `nodes.py` (PocketFlow nodes)
2. Update the flow in `flow.py`
3. Add the skill to the AgentCard in `a2a_server.py`

See `docs/design.md` for detailed architecture documentation.
