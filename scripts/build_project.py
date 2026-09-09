"""Generate, validate, train, evaluate and write honest, reproducible artifacts."""
import json
from pathlib import Path
from transfer_assistant.evaluation import run_pipeline

if __name__ == "__main__":
    results = run_pipeline(Path(__file__).resolve().parents[1])
    print(json.dumps({k: {f: v[f] for f in ["accuracy", "macro_f1", "coverage"]}
                      for k, v in results.items() if "coverage" in v}, indent=2))
