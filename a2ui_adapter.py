"""Translate agent responses into A2UI v0.8 JSONL messages.

Produces surfaceUpdate, dataModelUpdate, and beginRendering messages
for the A2UI React renderer. Messages are serialized as JSON lines.
"""

import json
import uuid
from typing import Any


def _make_id(prefix: str = "c") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def create_chat_surface(
    message_text: str,
    code: str | None = None,
    output: str | None = None,
    artifacts: dict[str, dict[str, Any]] | None = None,
    error: str | None = None,
    surface_id: str = "chat",
) -> list[dict]:
    """Build A2UI messages for a single chat response.

    Args:
        message_text: The agent's text response.
        code: Generated Python code (if any).
        output: Execution stdout (if any).
        artifacts: Dict of artifact_id -> {type, content}.
        error: Error message (if any).
        surface_id: The A2UI surface identifier.

    Returns:
        List of A2UI message dicts (surfaceUpdate, dataModelUpdate, beginRendering).
    """
    messages: list[dict] = []
    components: list[dict] = []
    data_contents: list[dict] = []
    children_ids: list[str] = []

    # Root container
    root_id = _make_id("root")

    # 1. Text message
    if message_text:
        text_id = _make_id("text")
        components.append({
            "id": text_id,
            "component": {
                "Text": {
                    "text": {"literalString": message_text},
                    "usageHint": "body",
                }
            }
        })
        children_ids.append(text_id)

    # 2. Artifacts (plots, tables, mermaid, html, markdown)
    if artifacts:
        for art_id, art_data in artifacts.items():
            art_type = art_data.get("type", "unknown")
            content = art_data.get("content", "")
            comp_id = _make_id(art_type)
            data_path = f"/artifacts/{art_id}"

            if art_type in ("plot", "svg"):
                components.append({
                    "id": comp_id,
                    "component": {
                        "PlotViewer": {
                            "data": {"path": data_path},
                        }
                    }
                })
                data_contents.append({
                    "key": art_id,
                    "valueString": content,
                })

            elif art_type == "table":
                components.append({
                    "id": comp_id,
                    "component": {
                        "DataTable": {
                            "html": {"path": data_path},
                        }
                    }
                })
                data_contents.append({
                    "key": art_id,
                    "valueString": content,
                })

            elif art_type == "mermaid":
                components.append({
                    "id": comp_id,
                    "component": {
                        "MermaidChart": {
                            "definition": {"path": data_path},
                        }
                    }
                })
                data_contents.append({
                    "key": art_id,
                    "valueString": content,
                })

            elif art_type == "html":
                components.append({
                    "id": comp_id,
                    "component": {
                        "HtmlViewer": {
                            "html": {"path": data_path},
                        }
                    }
                })
                data_contents.append({
                    "key": art_id,
                    "valueString": content,
                })

            elif art_type == "markdown":
                components.append({
                    "id": comp_id,
                    "component": {
                        "Text": {
                            "text": {"literalString": content},
                            "usageHint": "body",
                        }
                    }
                })

            else:
                continue

            children_ids.append(comp_id)

    # 3. Code block (collapsible)
    if code:
        code_id = _make_id("code")
        components.append({
            "id": code_id,
            "component": {
                "CodeBlock": {
                    "code": {"path": "/response/code"},
                    "language": {"literalString": "python"},
                }
            }
        })
        data_contents.append({"key": "code", "valueString": code})
        children_ids.append(code_id)

    # 4. Output block (collapsible)
    if output:
        output_id = _make_id("output")
        components.append({
            "id": output_id,
            "component": {
                "OutputBlock": {
                    "output": {"path": "/response/output"},
                }
            }
        })
        data_contents.append({"key": "output", "valueString": output})
        children_ids.append(output_id)

    # 5. Error block
    if error:
        error_id = _make_id("error")
        components.append({
            "id": error_id,
            "component": {
                "Text": {
                    "text": {"literalString": f"Error: {error}"},
                    "usageHint": "body",
                }
            }
        })
        children_ids.append(error_id)

    # Root column container
    components.append({
        "id": root_id,
        "component": {
            "Column": {
                "children": {"explicitList": children_ids},
                "alignment": "start",
            }
        }
    })

    # Build messages
    messages.append({
        "surfaceUpdate": {
            "surfaceId": surface_id,
            "components": components,
        }
    })

    if data_contents:
        # Separate artifact data from response data (code, output)
        artifact_data = [d for d in data_contents if d["key"] not in ("code", "output")]
        response_data = [d for d in data_contents if d["key"] in ("code", "output")]

        if artifact_data:
            messages.append({
                "dataModelUpdate": {
                    "surfaceId": surface_id,
                    "path": "/artifacts",
                    "contents": artifact_data,
                }
            })
        if response_data:
            messages.append({
                "dataModelUpdate": {
                    "surfaceId": surface_id,
                    "path": "/response",
                    "contents": response_data,
                }
            })

    messages.append({
        "beginRendering": {
            "surfaceId": surface_id,
            "root": root_id,
        }
    })

    return messages


def to_jsonl(messages: list[dict]) -> str:
    """Serialize A2UI messages to JSONL format."""
    return "\n".join(json.dumps(m, separators=(",", ":")) for m in messages)
