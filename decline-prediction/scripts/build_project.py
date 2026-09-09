"""Run the full reproducible pipeline and print the headline numbers.

    python scripts/build_project.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transfer_decline.pipeline import run

if __name__ == "__main__":
    summary = run(ROOT)
    print(json.dumps(summary, indent=2))
