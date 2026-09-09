"""Execute notebooks using this interpreter without registering a global kernel."""
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["IPYTHONDIR"] = str(ROOT / ".ipython")
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        nb = nbformat.read(path, as_version=4)
        manager = KernelManager(kernel_name="python3")
        manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        client = NotebookClient(nb, km=manager, timeout=180, resources={"metadata": {"path": str(ROOT)}})
        try:
            client.execute()
            nbformat.write(nb, path)
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
        print(f"Executed {path.name}", flush=True)
