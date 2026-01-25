"""
Sandbox Manager using Multiprocessing.
Manages persistent Python kernels for sessions.
"""

import multiprocessing
import time
from typing import Any, Dict, Optional
from utils.kernel import run_worker
from utils.sandbox_interface import SandboxInterface


class SandboxKernel(SandboxInterface):
    """Multiprocessing-based sandbox implementation."""

    @property
    def backend_name(self) -> str:
        return "multiprocessing"

    def __init__(self, timeout_seconds: int = 30):
        self.timeout_seconds = timeout_seconds
        self.process: Optional[multiprocessing.Process] = None
        self.pipe_parent = None
        self.pipe_child = None
        self._start()

    def _start(self):
        """Start the kernel process."""
        self.pipe_parent, self.pipe_child = multiprocessing.Pipe()
        self.process = multiprocessing.Process(target=run_worker, args=(self.pipe_child,))
        self.process.daemon = True
        self.process.start()

    def restart(self):
        """Restart the kernel (clears memory)."""
        self.terminate()
        self._start()

    def terminate(self):
        """Force kill the kernel."""
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1)
            if self.process.is_alive():
                self.process.kill()
        self.process = None

    def execute(self, code: str) -> Dict[str, Any]:
        """Execute code in the persistent kernel."""
        if not self.process or not self.process.is_alive():
            self._start()

        try:
            self.pipe_parent.send({'type': 'execute', 'code': code})
            
            # Poll for response with timeout
            if self.pipe_parent.poll(self.timeout_seconds):
                result = self.pipe_parent.recv()
                
                if result.get('status') == 'fatal_error':
                    self.restart()
                    return {
                        "success": False, 
                        "error": f"Kernel crashed: {result.get('error')}",
                        "stdout": "", "stderr": "", "artifacts": {}
                    }
                
                return {
                    "success": result['status'] == 'ok',
                    "stdout": result.get('stdout', ''),
                    "stderr": result.get('stderr', ''),
                    "artifacts": result.get('artifacts', {}),
                    "error": result.get('error')
                }
            else:
                # Timeout
                # Kill the stuck process to avoid zombie workers.
                self.restart()
                return {
                    "success": False,
                    "error": f"Execution timed out ({self.timeout_seconds}s). Kernel restarted.",
                    "stdout": "", "stderr": "", "artifacts": {}
                }
        except Exception as e:
            self.restart()
            return {
                "success": False,
                "error": f"System error: {e}",
                "stdout": "", "stderr": "", "artifacts": {}
            }

    def load_file(self, filename: str, arrow_data: bytes) -> bool:
        """Load a file into the kernel."""
        if not self.process or not self.process.is_alive():
            self._start()
            
        self.pipe_parent.send({
            'type': 'load_file',
            'filename': filename,
            'arrow_data': arrow_data
        })
        
        if self.pipe_parent.poll(10): # 10s timeout for loading
            res = self.pipe_parent.recv()
            return res['status'] == 'ok'
        return False

# Compatibility function (creates a transient kernel)
def execute_sandboxed_code(code: str, context: dict = None, timeout_seconds: int = 30) -> dict:
    """Run code in a throwaway kernel (used by older flows/tests)."""
    kernel = SandboxKernel(timeout_seconds)
    # If context has files, we need to load them. 
    # But context currently passes DataFrames directly. 
    # To support transient execution with DataFrames, we'd need to serialize them here.
    # For now, let's assume the new architecture uses the Kernel object directly.
    result = kernel.execute(code)
    kernel.terminate()
    return result
