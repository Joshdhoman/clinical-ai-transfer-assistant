"""Run after installing the local project: python scripts/generate_data.py."""
from pathlib import Path
from transfer_assistant.synthetic import write_dataset

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    data = write_dataset(root)
    print((root / "data/synthetic/validation.json").read_text())
