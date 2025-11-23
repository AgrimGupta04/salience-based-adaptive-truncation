"""
scripts/preprocess_pipeline.py
Prepares *_pairs.json files for all datasets (consistent with run_full_pipeline).
"""

from src.data_loader import load_dataset, prepare_data
from scripts.run_full_pipeline import load_dataset_config
import os

def main():
    configs = load_dataset_config()
    os.makedirs("data/processed", exist_ok=True)

    for dataset_name, cfg in configs.items():
        print(f" Preprocessing dataset: {dataset_name}")

        ds = load_dataset(dataset_name)

        out_path = f"data/processed/{dataset_name}_pairs.json"
        pairs = prepare_data(
            ds,
            dataset_name=dataset_name,
            input_field=cfg["input_field"],
            summary_field=cfg["summary_field"],
            chunk=True,
            max_tokens=cfg.get("max_chunk_tokens", 512),
            save_path=out_path
        )

        print(f"[done] {dataset_name}: saved {len(pairs)} pairs -> {out_path}")


if __name__ == "__main__":
    main()
