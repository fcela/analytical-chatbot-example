"""
Factory for creating sandbox instances with auto-detection.

This module provides a factory function that automatically selects the best
available sandbox backend (Docker via llm-sandbox, or multiprocessing fallback).
"""

import logging
import os
import shutil
import subprocess
from typing import Optional

from utils.sandbox_interface import SandboxInterface

logger = logging.getLogger(__name__)

# Configure sandbox logger to show INFO level by default
logging.getLogger("sandbox").setLevel(logging.INFO)

# Cache Docker availability check
_docker_available: Optional[bool] = None
_llm_sandbox_available: Optional[bool] = None


def is_docker_available() -> bool:
    """Check if Docker is available and running.

    Returns:
        True if Docker daemon is accessible, False otherwise.
    """
    global _docker_available

    if _docker_available is not None:
        logger.debug(f"[SANDBOX] Docker availability (cached): {_docker_available}")
        return _docker_available

    try:
        # Check if docker binary exists
        docker_path = shutil.which("docker")
        if not docker_path:
            logger.info("[SANDBOX] Docker binary not found in PATH")
            _docker_available = False
            return False
        logger.debug(f"[SANDBOX] Docker binary found at: {docker_path}")

        # Check if Docker daemon is running
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5
        )
        _docker_available = result.returncode == 0
        if _docker_available:
            logger.info("[SANDBOX] Docker daemon is running and accessible")
        else:
            logger.warning("[SANDBOX] Docker daemon not running (docker info failed)")

    except subprocess.TimeoutExpired:
        logger.warning("[SANDBOX] Docker check timed out")
        _docker_available = False
    except FileNotFoundError:
        logger.warning("[SANDBOX] Docker binary not found")
        _docker_available = False
    except Exception as e:
        logger.warning(f"[SANDBOX] Docker check failed: {e}")
        _docker_available = False

    return _docker_available


def is_llm_sandbox_available() -> bool:
    """Check if llm-sandbox library is installed.

    Returns:
        True if llm_sandbox can be imported, False otherwise.
    """
    global _llm_sandbox_available

    if _llm_sandbox_available is not None:
        logger.debug(f"[SANDBOX] llm-sandbox availability (cached): {_llm_sandbox_available}")
        return _llm_sandbox_available

    try:
        import llm_sandbox
        _llm_sandbox_available = True
        logger.info(f"[SANDBOX] llm-sandbox library is installed (version: {getattr(llm_sandbox, '__version__', 'unknown')})")
    except ImportError as e:
        logger.warning(f"[SANDBOX] llm-sandbox library not installed: {e}")
        _llm_sandbox_available = False

    return _llm_sandbox_available


def create_sandbox(
    timeout_seconds: int = 30,
    prefer_docker: bool = True,
    force_backend: Optional[str] = None,
    docker_image: Optional[str] = None
) -> SandboxInterface:
    """Create a sandbox instance with the best available backend.

    The function auto-detects available backends and selects the most suitable one.
    Docker (via llm-sandbox) is preferred when available, with multiprocessing as fallback.

    Args:
        timeout_seconds: Execution timeout in seconds (default: 30).
        prefer_docker: If True, prefer Docker when available (default: True).
        force_backend: Force a specific backend ("docker" or "multiprocessing").
            Raises RuntimeError if forced backend is not available.
        docker_image: Custom Docker image for llm-sandbox backend.

    Returns:
        SandboxInterface instance.

    Raises:
        RuntimeError: If forced backend is not available.

    Environment Variables:
        SANDBOX_PREFER_DOCKER: Override prefer_docker (default: "true")
        SANDBOX_FORCE_BACKEND: Override force_backend ("docker" or "multiprocessing")
        SANDBOX_TIMEOUT: Override timeout_seconds
        SANDBOX_DOCKER_IMAGE: Override docker_image
    """
    # Read environment overrides
    logger.info("[SANDBOX] ============================================")
    logger.info("[SANDBOX] Creating sandbox instance...")
    logger.info("[SANDBOX] Reading environment configuration:")

    env_prefer = os.environ.get("SANDBOX_PREFER_DOCKER")
    if env_prefer is not None:
        prefer_docker = env_prefer.lower() == "true"
        logger.info(f"[SANDBOX]   SANDBOX_PREFER_DOCKER={env_prefer} -> prefer_docker={prefer_docker}")
    else:
        logger.info(f"[SANDBOX]   SANDBOX_PREFER_DOCKER not set (default: {prefer_docker})")

    env_force = os.environ.get("SANDBOX_FORCE_BACKEND")
    if env_force:
        force_backend = env_force
        logger.info(f"[SANDBOX]   SANDBOX_FORCE_BACKEND={env_force} -> forcing '{force_backend}' backend")
    else:
        logger.info("[SANDBOX]   SANDBOX_FORCE_BACKEND not set (auto-detect)")

    env_timeout = os.environ.get("SANDBOX_TIMEOUT")
    if env_timeout:
        try:
            timeout_seconds = int(env_timeout)
            logger.info(f"[SANDBOX]   SANDBOX_TIMEOUT={env_timeout} -> {timeout_seconds}s")
        except ValueError:
            logger.warning(f"[SANDBOX]   SANDBOX_TIMEOUT={env_timeout} (invalid, using default: {timeout_seconds}s)")
    else:
        logger.info(f"[SANDBOX]   SANDBOX_TIMEOUT not set (default: {timeout_seconds}s)")

    env_image = os.environ.get("SANDBOX_DOCKER_IMAGE")
    if env_image:
        docker_image = env_image
        logger.info(f"[SANDBOX]   SANDBOX_DOCKER_IMAGE={env_image}")
    else:
        logger.info("[SANDBOX]   SANDBOX_DOCKER_IMAGE not set (using default)")

    logger.info("[SANDBOX] --------------------------------------------")

    # Handle forced backend
    if force_backend == "docker":
        logger.info("[SANDBOX] Backend FORCED to 'docker' - checking requirements...")
        if not is_docker_available():
            logger.error("[SANDBOX] FAILED: Docker backend requested but Docker is not available!")
            raise RuntimeError("Docker backend requested but Docker is not available")
        if not is_llm_sandbox_available():
            logger.error("[SANDBOX] FAILED: Docker backend requested but llm-sandbox is not installed!")
            raise RuntimeError("Docker backend requested but llm-sandbox is not installed")
        from utils.llm_sandbox_kernel import LLMSandboxKernel
        logger.info("[SANDBOX] SUCCESS: Using Docker backend via llm-sandbox (FORCED)")
        logger.info("[SANDBOX] ============================================")
        return LLMSandboxKernel(timeout_seconds=timeout_seconds, image=docker_image)

    if force_backend == "multiprocessing":
        logger.info("[SANDBOX] Backend FORCED to 'multiprocessing'")
        from utils.sandbox import SandboxKernel
        logger.info("[SANDBOX] SUCCESS: Using multiprocessing backend (FORCED)")
        logger.info("[SANDBOX] ============================================")
        return SandboxKernel(timeout_seconds=timeout_seconds)

    # Auto-detection
    logger.info("[SANDBOX] Auto-detecting best available backend...")
    docker_ok = is_docker_available()
    llm_sandbox_ok = is_llm_sandbox_available()

    logger.info(f"[SANDBOX] Detection results: Docker available={docker_ok}, llm-sandbox installed={llm_sandbox_ok}, prefer_docker={prefer_docker}")

    if prefer_docker and docker_ok and llm_sandbox_ok:
        try:
            from utils.llm_sandbox_kernel import LLMSandboxKernel
            logger.info("[SANDBOX] SUCCESS: Using Docker backend via llm-sandbox (auto-detected)")
            logger.info("[SANDBOX] ============================================")
            return LLMSandboxKernel(timeout_seconds=timeout_seconds, image=docker_image)
        except Exception as e:
            # If Docker backend fails to initialize, fall back to multiprocessing
            logger.warning(f"[SANDBOX] Docker backend initialization failed, falling back: {e}")

    # Fallback to multiprocessing
    from utils.sandbox import SandboxKernel
    if not docker_ok:
        logger.info("[SANDBOX] SUCCESS: Using multiprocessing backend (Docker not available)")
    elif not llm_sandbox_ok:
        logger.info("[SANDBOX] SUCCESS: Using multiprocessing backend (llm-sandbox not installed)")
    else:
        logger.info("[SANDBOX] SUCCESS: Using multiprocessing backend (Docker not preferred)")
    logger.info("[SANDBOX] ============================================")
    return SandboxKernel(timeout_seconds=timeout_seconds)


def get_available_backends() -> list[str]:
    """Get list of available sandbox backends.

    Returns:
        List of backend names that are currently available.
    """
    backends = ["multiprocessing"]  # Always available

    if is_docker_available() and is_llm_sandbox_available():
        backends.append("docker")

    return backends
