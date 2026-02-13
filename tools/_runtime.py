"""Runtime context for tool execution.

Provides per-request sandbox kernel access to tools without passing
the kernel through LangChain's tool interface.

Uses contextvars.ContextVar so each concurrent request (even when run
via run_in_executor) gets its own isolated kernel reference.
"""

from contextvars import ContextVar
from typing import Optional

from utils.sandbox_interface import SandboxInterface

_current_kernel: ContextVar[Optional[SandboxInterface]] = ContextVar(
    "_current_kernel", default=None
)


def set_kernel(kernel: Optional[SandboxInterface]) -> None:
    """Set the sandbox kernel for the current request."""
    _current_kernel.set(kernel)


def get_kernel() -> Optional[SandboxInterface]:
    """Get the sandbox kernel for the current request."""
    return _current_kernel.get()
