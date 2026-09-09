"""Execute every notebook in place using this interpreter.

Run after `scripts/build_project.py` so notebook 05 can load real artifacts.
"""

import os
import sys
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["IPYTHONDIR"] = str(ROOT / ".ipython")
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        nb = nbformat.read(path, as_version=4)
        manager = KernelManager(kernel_name="python3")
        manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        client = NotebookClient(nb, km=manager, timeout=240,
                                resources={"metadata": {"path": str(ROOT)}})
        try:
            client.execute()
            nbformat.write(nb, path)
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
        print(f"executed {path.name}", flush=True)
