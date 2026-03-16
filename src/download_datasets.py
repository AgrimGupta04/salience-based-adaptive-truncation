"""
download_datasets.py

Downloads and saves summarization datasets (CNN/DailyMail, GovReport, ArXiv)
for Salience-based Adaptive Truncation experiments.

"""

import os
from datasets import load_dataset, load_from_disk
import tiktoken

## Define datasets and local save paths
DATASETS = {
    # "cnn_dailymail": {
    #     "hf_name": "cnn_dailymail",
    #     "subset": "3.0.0",
    #     "split": "test",
    #     "input_field": "article",
    #     "summary_field": "highlights",
    #     "revision": "main",
    # },
    # "govreport": {
    #     "hf_name": "ccdv/govreport-summarization",
    #     "subset": None,
    #     "split": "test",
    #     "input_field": "document",
    #     "summary_field": "summary",
    #     "revision": "main",
    #     "verification_mode": "no_checks",
    # },
    "arxiv": {
        "hf_name": "ccdv/arxiv-summarization",
        "subset": None,
        "split": "test",
        "input_field": "article",
        "summary_field": "abstract",
        "revision": "main",
        "verification_mode": "no_checks",
    },
}

## Tokenizer for estimating input lengths
enc = tiktoken.get_encoding("cl100k_base")

## Ensure directories exist
os.makedirs("data/raw", exist_ok=True)

def estimate_avg_length(dataset, field, n_samples=200):
    """Estimate average token count for the input field."""
    total = 0
    ## Selecting a small range to estimate length without processing the whole dataset
    for i, sample in enumerate(dataset.select(range(min(len(dataset), n_samples)))):
        total += len(enc.encode(sample[field]))
    return total / n_samples


def download_and_save(name, config):
    local_path = f"data/raw/{name}"
    if os.path.exists(local_path):
        print(f" {name} already downloaded. Skipping.")
        return load_from_disk(local_path)

    ## Retrieving configuration parameters
    hf_name = config["hf_name"]
    subset = config.get("subset", None)
    split = config.get("split", "train")
    revision = config.get("revision", None)
    verification_mode = config.get("verification_mode", "all_checks")
    # trust_code = config.get("trust_remote_code", False)

    print(f"\n  Downloading {name} dataset.")
    try:
        ## We use a unified call, passing None for parameters that aren't set.
        ds = load_dataset(
            hf_name, 
            subset, 
            split=split, 
            revision=revision,
            verification_mode = verification_mode
        )

        ds.save_to_disk(local_path)
        print(f" Saved to {local_path}")

        ## Display some basic info (guarded)
        try:
            sample = ds[0]
            input_field = config.get("input_field")
            summary_field = config.get("summary_field")
            
            if input_field and input_field in sample:
                print(f" Example Input: {sample[input_field][:150]}.")
            if summary_field and summary_field in sample:
                print(f" Reference Summary: {sample[summary_field][:150]}.")
        except Exception:
            pass

        ## Estimate avg length if input_field exists
        try:
            avg_len = estimate_avg_length(ds, config["input_field"])
            print(f" Average input length (first 200 samples): {avg_len:.2f} tokens")
        except Exception:
            pass

        return ds

    except Exception as e:      #3 Incase of failure due to hugginggface version
        print(f" Failed to download {name}: {e}")
        print("-> If you are using datasets 2.11.0, the Invalid pattern: '**' bug may require updating fsspec/huggingface_hub.")
        print("Please try running: pip install -U fsspec huggingface_hub")
        return None


def main():
    print("Starting dataset downloads for summarization experiments.\n")
    for name, cfg in DATASETS.items():
        try:
            out = download_and_save(name, cfg)
            if out is None:
                print(f"Warning: {name} failed to download (see message). Continuing to next dataset.")
        except Exception as e:
            print(f" Failed to download {name}: {e}")

    print("\n All datasets ready under `data/raw/` (or failed with warnings).\n")

if __name__ == "__main__":
    main()