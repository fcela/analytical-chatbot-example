"""Runtime context for tool execution.

Provides per-request sandbox kernel access to tools without passing
the kernel through LangChain's tool interface.

Uses a simple global variable since agent.invoke() runs synchronously
within a single request context. Thread-local storage was unreliable
because run_in_executor and LangGraph may use different threads.
"""

from typing import Optional

from utils.sandbox_interface import SandboxInterface

_current_kernel: Optional[SandboxInterface] = None


def set_kernel(kernel: Optional[SandboxInterface]) -> None:
    """Set the sandbox kernel for the current request."""
    global _current_kernel
    _current_kernel = kernel


def get_kernel() -> Optional[SandboxInterface]:
    """Get the sandbox kernel for the current request."""
    return _current_kernel
