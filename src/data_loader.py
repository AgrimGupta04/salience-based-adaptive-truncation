"""
Handles loading and preprocessing of datasets stored under data/raw/*.
Supports:
    - Multiple datasets with different input / summary field shapes
    - Missing / nested fields (BookSum, Mixed Science)
    - List-based input (Reddit-TIFU)
    - Optional token-based chunking
"""

from typing import List, Dict, Any
from datasets import load_from_disk
from tqdm import tqdm
import os
import tiktoken 
import nltk
nltk.download('punkt', quiet=True)

enc = tiktoken.get_encoding("cl100k_base")

DATA_PATH = "data/raw/"
PROCESSED_PATH = "data/processed/"

def extract_text(sample: Dict[str, Any], input_field: str) -> str:
    """
    Extracts the document text for a dataset sample.

    Handles:
        - Missing fields (throws clear error)
        - list[str] → join (Reddit-TIFU)
        - nested dict → join values (BookSum)
        - raw strings → cleaned
    """
    if input_field not in sample:
        raise KeyError(
            f"[extract_text] Field '{input_field}' not found in sample.\n"
            f"Available fields: {list(sample.keys())}"
        )

    text = sample[input_field]

    # Example: Reddit-TIFU → "documents": [sent1, sent2, ...]
    if isinstance(text, list):
        text = " ".join([str(t) for t in text])

    # Example: BookSum → nested dict like {"chapter_text": "..."}
    if isinstance(text, dict):
        text = " ".join([str(v) for v in text.values()])

    text = str(text)
    return text.strip().replace("\n", " ")


def extract_summary(sample: Dict[str, Any], summary_field: str) -> str:
    """
    Extracts the summary text.

    Handles:
        - Simple strings
        - Nested structures: {"summary_text": ...} or {"abstract": ...}
        - Missing fields → clear error
    """
    if summary_field not in sample:
        raise KeyError(
            f"[extract_summary] Summary field '{summary_field}' not found.\n"
            f"Available fields: {list(sample.keys())}"
        )

    summary = sample[summary_field]

    # BookSum has nested dict summaries
    if isinstance(summary, dict):
        # Try common keys
        for key in ["summary", "summary_text", "abstract"]:
            if key in summary:
                summary = summary[key]
                break

    return str(summary).strip()

def load_dataset(dataset_name: str):
    """Loads the dataset from disk by a given name."""

    path = os.path.join(DATA_PATH, dataset_name)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset {dataset_name} not found at {path}. Please run download_datasets.py first.")
    
    ds = load_from_disk(path)
    print(f"Loaded dataset {dataset_name} from {path}.")
    return ds

def chunk_text(text: str, max_tokens: int = 512) -> List[str]:
    "Splits a long text into roughly equal token-sized chunks."

    sentences = nltk.sent_tokenize(text)
    chunks, current_chunk, token_count = [], [], 0

    for sent in sentences:
        tokens = enc.encode(sent)
        if token_count + len(tokens) > max_tokens:
            chunks.append(" ".join(current_chunk))  ## To make a chunk when max tokens exceeded
            current_chunk, token_count = [sent], len(tokens)    ## Storing the current sent and token count to be used for next chunk check.
        else:
            current_chunk.append(sent)
            token_count += len(tokens)

    if current_chunk:
        chunks.append(" ".join(current_chunk))  ## To make a final chunk if any sentences remains

    return chunks

def prepare_data(dataset, dataset_name: str, input_field: str, summary_field: str, chunk: bool = False, max_tokens: int = 512, save_path: str = None) -> List[dict]:
        """
        Prepares text summary pairs from a Hugging face dataset.
        
        Args:
            dataset: HF Dataset object (from load_from_disk())
            dataset_name: name of dataset for ID prefix
            input_field: name of document field ('article', 'report', etc.)
            summary_field: name of summary field ('highlights', 'summary')
            chunk: whether to chunk long documents by token length
            max_tokens: max tokens per chunk (used if chunk=True)
            save_path: optional JSON path to save preprocessed data

        Returns:
            List of dicts:
            {
                "id": "dataset_123_0",
                "text": "...",
                "summary": "...",
            }
        """
        pairs = []

        for idx, sample in tqdm(enumerate(dataset), total=len(dataset), desc=f"[prepare] {dataset_name}"):
            text = extract_text(sample, input_field)
            summary = extract_summary(sample, summary_field)

            base_id = (
                sample["id"]
                if ("id" in sample and isinstance(sample["id"], str))
                else f"{dataset_name}_{idx}"
            )

            if chunk:       ## Chunking if document too big.
                chunks = chunk_text(text, max_tokens = max_tokens)
                for i, chunked_text in enumerate(chunks):
                    pairs.append({
                        'id': f"{base_id}_{i}",
                        'text': chunked_text,
                        'summary': summary
                    })

            else:       ## Else making/working in the single chunk.
                pairs.append({
                    "id": base_id,
                    "text": text,
                    "summary": summary
                })

        print(f"Prepared {len(pairs)} text-summary pairs.")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            import json
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(pairs, f, indent=2, ensure_ascii=False)
            print(f"Saved preprocessed data to {save_path}")

        return pairs

if __name__ == "__main__":
    ds = load_dataset("cnn_dailymail")
    sample_pairs = prepare_data(ds, dataset_name = "cnn_dailymail", input_field="article", summary_field="highlights", chunk = True)
    print(sample_pairs[0])