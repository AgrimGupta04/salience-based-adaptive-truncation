"""
download_datasets.py

Downloads and saves summarization datasets (CNN/DailyMail, GovReport, ArXiv)
for Salience-based Adaptive Truncation experiments.

Author: Agrim Gupta
"""

import os
from datasets import load_dataset, load_from_disk
import tiktoken

## Define datasets and local save paths (only datasets used in the paper)
DATASETS = {
    "cnn_dailymail": {
        "hf_name": "cnn_dailymail",
        "subset": "3.0.0",
        "split": "test",
        "input_field": "article",
        "summary_field": "highlights",
        "revision": "main",
        "token_budget": 512,  # From paper: 512 tokens for CNN
    },
    "govreport": {
        "hf_name": "ccdv/govreport-summarization",
        "subset": None,
        "split": "test",
        "input_field": "document",
        "summary_field": "summary",
        "revision": "main",
        "verification_mode": "no_checks",
        "token_budget": 4096,  # From paper: 4096 tokens for GovReport
    },
    "arxiv": {
        "hf_name": "ccdv/arxiv-summarization",
        "subset": None,
        "split": "test",
        "input_field": "article",
        "summary_field": "abstract",
        "revision": "main",
        "verification_mode": "no_checks",
        "token_budget": 4096,  # From paper: 4096 tokens for ArXiv
    },
}

## Ensure directories exist
os.makedirs("data/raw", exist_ok=True)

def estimate_avg_length(dataset, field, n_samples=200):
    """Estimate average token count for the input field using word count approximation."""
    total = 0
    # Select a small range to estimate length without processing the whole dataset
    for i, sample in enumerate(dataset.select(range(min(len(dataset), n_samples)))):
        # Simple word count as proxy (more stable across tokenizers)
        text = sample[field]
        if isinstance(text, str):
            total += len(text.split())  # Word count as approximation
        else:
            total += len(str(text).split())
    return total / n_samples

def collect_dataset_stats(dataset, config):
    """Collect and print dataset statistics."""
    print(f"\n{'='*50}")
    print(f"Dataset: {config.get('name', 'Unknown')}")
    print(f"{'='*50}")
    
    # Basic stats
    print(f"Size: {len(dataset)} samples")
    
    # Sample input/output lengths
    if len(dataset) > 0:
        sample = dataset[0]
        input_field = config.get("input_field")
        summary_field = config.get("summary_field")
        
        if input_field and input_field in sample:
            input_text = sample[input_field]
            input_words = len(str(input_text).split())
            print(f"Sample input length: {input_words} words")
            print(f"Sample input preview: {str(input_text)[:200]}...")
        
        if summary_field and summary_field in sample:
            summary_text = sample[summary_field]
            summary_words = len(str(summary_text).split())
            print(f"Sample summary length: {summary_words} words")
            print(f"Sample summary: {str(summary_text)[:200]}...")
    
    # Token budget info
    token_budget = config.get("token_budget")
    if token_budget:
        print(f"Token budget for experiments: {token_budget}")
    
    return {
        "size": len(dataset),
        "input_field": config.get("input_field"),
        "summary_field": config.get("summary_field"),
        "token_budget": token_budget
    }


def download_and_save(name, config):
    local_path = f"data/raw/{name}"
    if os.path.exists(local_path):
        print(f"✓ {name} already downloaded. Skipping.")
        ds = load_from_disk(local_path)
        stats = collect_dataset_stats(ds, {"name": name, **config})
        return ds, stats

    # Retrieve configuration parameters
    hf_name = config["hf_name"]
    subset = config.get("subset", None)
    split = config.get("split", "test")
    revision = config.get("revision", None)
    verification_mode = config.get("verification_mode", "all_checks")

    print(f"\n Downloading {name} dataset...")
    try:
        # Load dataset
        ds = load_dataset(
            hf_name, 
            subset, 
            split=split, 
            revision=revision,
            verification_mode=verification_mode
        )

        # Save to disk
        ds.save_to_disk(local_path)
        print(f"✓ Saved to {local_path}")
        
        # Collect and display stats
        stats = collect_dataset_stats(ds, {"name": name, **config})
        
        # Estimate average length
        try:
            avg_len = estimate_avg_length(ds, config["input_field"])
            print(f"📊 Average input length (first 200 samples): {avg_len:.2f} words")
            token_budget = config.get("token_budget")
            if token_budget:
                # Approximate conversion: 1 token ≈ 0.75 words for English
                avg_tokens = avg_len * 1.33
                compression = (token_budget / avg_tokens) * 100 if avg_tokens > 0 else 0
                print(f"Expected compression: ~{min(compression, 100):.1f}% of original tokens")
        except Exception as e:
            print(f"Could not estimate average length: {e}")

        return ds, stats

    except Exception as e:
        print(f"Failed to download {name}: {e}")
        print("\nTroubleshooting tips:")
        print("   1. Check internet connection")
        print("   2. Update packages: pip install -U fsspec huggingface_hub datasets")
        print("   3. For large datasets, try smaller split first")
        return None, None


def main():
    print("\n" + "="*60)
    print("Starting dataset downloads for summarization experiments")
    print("Paper: Salience-based Adaptive Truncation for Efficient Summarization")
    print("="*60 + "\n")
    
    all_stats = {}
    
    for name, cfg in DATASETS.items():
        try:
            print(f"\n🔍 Processing: {name}")
            dataset, stats = download_and_save(name, cfg)
            if dataset is None:
                print(f"  ⚠ Warning: {name} failed to download. Continuing to next dataset.")
            else:
                all_stats[name] = stats
        except Exception as e:
            print(f"Failed to process {name}: {e}")

    # Summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    
    for name, stats in all_stats.items():
        if stats:
            print(f"\n{name}:")
            print(f"  Samples: {stats.get('size', 'N/A')}")
            print(f"  Token budget: {stats.get('token_budget', 'N/A')}")
            print(f"  Input field: {stats.get('input_field', 'N/A')}")
            print(f"  Summary field: {stats.get('summary_field', 'N/A')}")

    print("\nAll datasets ready under `data/raw/`")
    print("\nNext steps:")
    print("   1. Run data_loader.py to preprocess datasets")
    print("   2. Check the statistics match your paper's description")
    print("   3. Proceed with salience scoring and truncation experiments")

if __name__ == "__main__":
    main()