"""Runtime context for tool execution.

Provides per-session sandbox kernel access to tools without passing
the kernel through LangChain's tool interface.
"""

import threading
from typing import Optional

from utils.sandbox_interface import SandboxInterface

_thread_local = threading.local()


def set_kernel(kernel: Optional[SandboxInterface]) -> None:
    """Set the sandbox kernel for the current thread."""
    _thread_local.kernel = kernel


def get_kernel() -> Optional[SandboxInterface]:
    """Get the sandbox kernel for the current thread."""
    return getattr(_thread_local, "kernel", None)
