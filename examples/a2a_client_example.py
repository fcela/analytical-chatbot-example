#!/usr/bin/env python3
"""
Example A2A Client for the Analytical Chatbot.

This script demonstrates how to interact with the Analytical Chatbot
using the A2A (Agent-to-Agent) protocol.

Prerequisites:
    1. Start the A2A server: python main.py
    2. Install httpx: pip install httpx

Usage:
    python examples/a2a_client_example.py
"""

import json
import sys

try:
    import httpx
except ImportError:
    print("Please install httpx: pip install httpx")
    sys.exit(1)


BASE_URL = "http://localhost:8000"


def get_agent_card() -> dict:
    """Fetch the agent's capability card."""
    response = httpx.get(f"{BASE_URL}/.well-known/agent-card.json")
    response.raise_for_status()
    return response.json()


def send_message(text: str, context_id: str = None) -> dict:
    """
    Send a message to the agent via A2A JSON-RPC protocol.

    Args:
        text: The message text to send.
        context_id: Optional context ID for multi-turn conversations.

    Returns:
        The JSON-RPC response containing the task result.
    """
    message = {
        "role": "user",
        "parts": [{"text": text}],
    }
    if context_id:
        message["contextId"] = context_id

    payload = {
        "jsonrpc": "2.0",
        "method": "message/send",
        "params": {"message": message},
        "id": 1
    }

    response = httpx.post(
        f"{BASE_URL}/",
        json=payload,
        timeout=60.0  # Allow time for LLM processing
    )
    response.raise_for_status()
    return response.json()


def extract_response_text(result: dict) -> str:
    """Extract the agent's text response from the A2A result."""
    if "error" in result:
        return f"Error: {result['error']}"

    if "result" not in result:
        return "No result in response"

    task = result["result"]

    # Look for agent messages in history
    for msg in task.get("history", []):
        if msg.get("role") == "agent":
            for part in msg.get("parts", []):
                if "text" in part:
                    return part["text"]

    return "No text response found"


def extract_artifacts(result: dict) -> list:
    """Extract artifacts (code, plots, tables) from the A2A result."""
    artifacts = []

    if "result" not in result:
        return artifacts

    task = result["result"]

    for artifact in task.get("artifacts", []):
        artifact_info = {
            "name": artifact.get("name", "unknown"),
            "id": artifact.get("artifactId", ""),
            "content": None
        }

        for part in artifact.get("parts", []):
            if "text" in part:
                artifact_info["content"] = part["text"]
            elif "data" in part:
                artifact_info["content"] = part["data"]

        artifacts.append(artifact_info)

    return artifacts


def print_separator(title: str = ""):
    """Print a visual separator."""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print('='*60)
    else:
        print('-'*60)


def main():
    """Run the example A2A client demonstration."""

    print_separator("A2A Client Example - Analytical Chatbot")

    # Step 1: Discover the agent
    print("\n[1] Fetching Agent Card...")
    try:
        agent_card = get_agent_card()
        print(f"    Agent Name: {agent_card['name']}")
        print(f"    Version: {agent_card['version']}")
        print(f"    Skills:")
        for skill in agent_card.get('skills', []):
            print(f"      - {skill['name']}: {skill['description'][:50]}...")
    except httpx.ConnectError:
        print("    ERROR: Could not connect to the server.")
        print("    Make sure the A2A server is running: python main.py")
        sys.exit(1)

    # Step 2: Simple conversation
    print_separator("Simple Conversation")

    print("\n[2] Sending greeting...")
    result = send_message("Hello! What can you do?")
    response_text = extract_response_text(result)
    print(f"    User: Hello! What can you do?")
    print(f"    Agent: {response_text[:300]}{'...' if len(response_text) > 300 else ''}")

    # Step 3: Database query
    print_separator("Database Query Example")

    print("\n[3] Querying database...")
    context_id = "example-session-001"

    result = send_message(
        "Show me the employees with salary over 80000 from the database",
        context_id
    )
    response_text = extract_response_text(result)
    artifacts = extract_artifacts(result)

    print(f"    User: Show me the employees with salary over 80000")
    print(f"    Agent: {response_text[:300]}{'...' if len(response_text) > 300 else ''}")

    if artifacts:
        print(f"\n    Artifacts generated:")
        for artifact in artifacts:
            print(f"      - {artifact['name']}")
            if artifact['name'] == 'generated_code' and artifact['content']:
                code_preview = artifact['content'][:200]
                print(f"        Code preview: {code_preview}...")

    # Step 4: Visualization request (multi-turn)
    print_separator("Visualization Example (Multi-turn)")

    print("\n[4] Requesting visualization (using same context)...")
    result = send_message(
        "Create a bar chart showing the salary distribution",
        context_id
    )
    response_text = extract_response_text(result)
    artifacts = extract_artifacts(result)

    print(f"    User: Create a bar chart showing the salary distribution")
    print(f"    Agent: {response_text[:300]}{'...' if len(response_text) > 300 else ''}")

    if artifacts:
        print(f"\n    Artifacts generated:")
        for artifact in artifacts:
            print(f"      - {artifact['name']}")

    # Step 5: Show raw response structure
    print_separator("Raw A2A Response Structure")

    print("\n[5] Example raw response (truncated):")
    result = send_message("What is 2 + 2?")
    # Pretty print with truncation
    result_str = json.dumps(result, indent=2)
    if len(result_str) > 1000:
        print(result_str[:1000] + "\n    ... (truncated)")
    else:
        print(result_str)

    print_separator("Example Complete")
    print("\nThe Analytical Chatbot is ready for A2A communication!")
    print("Try the other endpoints:")
    print(f"  - Agent Card: {BASE_URL}/.well-known/agent-card.json")
    print(f"  - JSON-RPC:   {BASE_URL}/")
    print(f"  - gRPC:       localhost:50051")
    print(f"  - Legacy REST: {BASE_URL}/chat (for frontend)")


if __name__ == "__main__":
    main()
