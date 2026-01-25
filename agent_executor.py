"""A2A Agent Executor for the Analytical Chatbot.

This module wraps the existing PocketFlow-based chatbot flow into an A2A-compatible
AgentExecutor that can handle A2A protocol requests.
"""

import asyncio
import logging
import uuid
import base64
import io
import polars as pl
from typing import Any

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TaskArtifactUpdateEvent,
    Artifact,
    Part,
    TextPart,
    DataPart,
    Message,
)
from a2a.utils.message import new_agent_text_message

from utils.flow import run_chatbot
from utils.sandbox_factory import create_sandbox

logger = logging.getLogger(__name__)

class AnalyticalChatbotExecutor(AgentExecutor):
    """A2A Agent Executor that wraps the analytical chatbot flow."""

    def __init__(self):
        """Initialize the executor with session storage for multi-turn conversations."""
        self._sessions: dict[str, dict[str, Any]] = {}

    def _get_session(self, context_id: str | None) -> dict[str, Any]:
        """Get or create a session for the given context ID."""
        if not context_id:
            context_id = "default"

        if context_id not in self._sessions:
            logger.info(f"[SESSION] Creating new session for context_id='{context_id}'")
            kernel = create_sandbox()
            logger.info(f"[SESSION] Session created with backend: {kernel.backend_name}")
            print(f"[Sandbox] Created new session with backend: {kernel.backend_name}")
            self._sessions[context_id] = {
                "files": {},  # Metadata only (actual data lives inside the kernel).
                "history": [],
                "kernel": kernel
            }
        else:
            logger.debug(f"[SESSION] Reusing existing session for context_id='{context_id}'")

        return self._sessions[context_id]

    def _process_incoming_message(self, context: RequestContext, session: dict[str, Any]) -> str:
        """Extract text content and process any file uploads (DataParts)."""
        message = context.message
        if not message or not message.parts:
            return ""

        text_parts = []
        for part in message.parts:
            part_data = part.root if hasattr(part, 'root') else part
            
            # Text Parts
            if isinstance(part_data, TextPart):
                text_parts.append(part_data.text)
            elif hasattr(part_data, 'text') and not hasattr(part_data, 'data'):
                text_parts.append(part_data.text)
            
            # Data Parts (File Uploads)
            elif isinstance(part_data, DataPart) or hasattr(part_data, 'data'):
                data = getattr(part_data, 'data', {})
                if "arrow_stream" in data:
                    try:
                        b64_data = data["arrow_stream"]
                        filename = data.get("filename", "uploaded_file.arrow")
                        
                        binary_data = base64.b64decode(b64_data)
                        
                        # Load directly into kernel
                        kernel = session["kernel"]
                        success = kernel.load_file(filename, binary_data)
                        
                        if success:
                            # Store metadata
                            session["files"][filename] = {"size": len(binary_data)} 
                            print(f"Success: Loaded file '{filename}' into kernel")
                        else:
                            print(f"Failed to load file '{filename}' into kernel")
                            
                    except Exception as e:
                        print(f"Error processing file upload: {e}")
                        import traceback
                        traceback.print_exc()

        return " ".join(text_parts)

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue
    ) -> None:
        """Execute the analytical chatbot flow for the incoming request."""
        # Get session data for multi-turn conversations.
        message = context.message
        context_id = getattr(message, "context_id", getattr(message, "contextId", None)) if message else None
        session = self._get_session(context_id)
        
        # Get task ID safely
        task_id = context.task_id

        # Process message: extract text and load files into the kernel.
        user_message = self._process_incoming_message(context, session)

        # If files were uploaded but no text message
        if not user_message and session.get("files"):
             user_message = "" 

        if not user_message and not session.get("files"):
            # No text, no files
            msg = new_agent_text_message("I didn't receive a message. How can I help you?")
            msg.task_id = task_id
            msg.context_id = context_id
            await event_queue.enqueue_event(msg)
            
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=task_id,
                    context_id=context_id,
                    status=TaskStatus(state=TaskState.completed),
                    final=True
                )
            )
            return

        # Send status update: working (streaming clients can show a spinner).
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=task_id,
                context_id=context_id,
                status=TaskStatus(state=TaskState.working),
                final=False
            )
        )

        try:
            if not user_message:
                files_msg = f"Received file(s). Ready for analysis."
                msg = new_agent_text_message(files_msg)
                msg.task_id = task_id
                msg.context_id = context_id
                await event_queue.enqueue_event(msg)
                
                await event_queue.enqueue_event(
                    TaskStatusUpdateEvent(
                        task_id=task_id,
                        context_id=context_id,
                        status=TaskStatus(state=TaskState.completed),
                        final=True
                    )
                )
                return

            # Run the chatbot flow on a thread pool to avoid blocking the event loop.
            # We pass the kernel, not the raw file data, so the code can access loaded tables.
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: run_chatbot(
                    user_message=user_message,
                    kernel=session["kernel"],
                    chat_history=session["history"]
                )
            )

            # Update session history
            session["history"] = result.get("chat_history", [])

            # Extract response data
            response = result.get("response", {})
            message_text = response.get("message", "")
            code = response.get("code")
            output = response.get("output")
            artifacts_dict = response.get("artifacts", {})
            error = response.get("error")

            # helper to send artifacts
            async def send_art(name, parts):
                print(f"DEBUG: Sending artifact {name}")
                await event_queue.enqueue_event(
                    TaskArtifactUpdateEvent(
                        task_id=task_id,
                        context_id=context_id,
                        artifact=Artifact(
                            artifact_id=f"{task_id}-{name}",
                            name=name,
                            parts=parts
                        )
                    )
                )

            # Send artifacts
            if code:
                await send_art("generated_code", [Part(root=TextPart(text=code))])

            if output:
                await send_art("execution_output", [Part(root=TextPart(text=output))])

            for art_id, art_data in artifacts_dict.items():
                art_type = art_data.get("type")
                content = art_data.get("content")
                
                if art_type == "plot":
                    await send_art(art_id, [Part(root=DataPart(data={"image_base64": content}))])
                elif art_type == "table":
                    await send_art(art_id, [Part(root=DataPart(data={"html": content}))])
                elif art_type == "html" or art_type == "mermaid" or art_type == "markdown":
                    await send_art(art_id, [Part(root=TextPart(text=content))])

            if error:
                await send_art("error", [Part(root=TextPart(text=error))])

            # Send message
            msg = new_agent_text_message(message_text)
            msg.task_id = task_id
            msg.context_id = context_id
            await event_queue.enqueue_event(msg)

            # Send final status
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=task_id,
                    context_id=context_id,
                    status=TaskStatus(state=TaskState.completed),
                    final=True
                )
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            msg = new_agent_text_message(f"Error: {str(e)}")
            msg.task_id = task_id
            msg.context_id = context_id
            await event_queue.enqueue_event(msg)
            
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=task_id,
                    context_id=context_id,
                    status=TaskStatus(state=TaskState.failed, message=new_agent_text_message(str(e))),
                    final=True
                )
            )

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue
    ) -> None:
        message = context.message
        context_id = getattr(message, "context_id", getattr(message, "contextId", None)) if message else None
        task_id = context.task_id
        
        # Kill the kernel if cancelled? Maybe not, keep session alive.
        # But we could interrupt execution if it supported it (send SIGINT).
        # self._sessions[context_id]["kernel"].interrupt()

        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=task_id,
                context_id=context_id,
                status=TaskStatus(state=TaskState.cancelled),
                final=True
            )
        )
