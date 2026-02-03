"""
Abstract interface for code execution sandboxes.

This module defines the contract that all sandbox implementations must follow,
enabling a pluggable backend architecture (multiprocessing, Docker, etc.).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class SandboxInterface(ABC):
    """Abstract base class for code execution sandboxes."""

    @abstractmethod
    def __init__(self, timeout_seconds: int = 30):
        """Initialize the sandbox with timeout configuration.

        Args:
            timeout_seconds: Maximum execution time before timeout.
        """
        pass

    @abstractmethod
    def execute(self, code: str) -> Dict[str, Any]:
        """Execute code in the sandbox.

        Args:
            code: Python code to execute.

        Returns:
            Dictionary with keys:
                - success (bool): Whether execution succeeded
                - stdout (str): Standard output
                - stderr (str): Standard error output
                - artifacts (dict): Generated artifacts {id: {type, content}}
                - error (str | None): Error message if failed
        """
        pass

    @abstractmethod
    def load_file(self, filename: str, arrow_data: bytes) -> bool:
        """Load Arrow-serialized data into the sandbox namespace.

        Args:
            filename: Name of the file (used as variable name base).
            arrow_data: Binary Arrow IPC stream data.

        Returns:
            True if loading succeeded, False otherwise.
        """
        pass

    @abstractmethod
    def restart(self) -> None:
        """Restart the sandbox, clearing all state."""
        pass

    @abstractmethod
    def terminate(self) -> None:
        """Terminate the sandbox and release resources."""
        pass

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return the backend identifier (e.g., 'multiprocessing', 'docker')."""
        pass
