"""
Handles loading and preprocessing of datasets stored under data/raw/*.
Supports:
    - Multiple datasets with different input / summary field shapes
    - Optional token-based chunking
"""

from typing import List, Dict, Any
from datasets import load_from_disk
from tqdm import tqdm
import os
from transformers import AutoTokenizer
import nltk
nltk.download('punkt_tab', quiet=True)

_TOKENIZERS = {}

DATA_PATH = "data/raw/"
PROCESSED_PATH = "data/processed/"

def get_tokenizer(model_name="facebook/bart-large-cnn"):
    if model_name not in _TOKENIZERS:
        # print(f"[DataLoader] Loading tokenizer: {model_name}")
        _TOKENIZERS[model_name] = AutoTokenizer.from_pretrained(model_name)
    return _TOKENIZERS[model_name]

def extract_text(sample: Dict[str, Any], input_field: str) -> str:
    """
    Extracts the document text for a dataset sample.
    """
    if input_field not in sample:

        aliases = {
            "article_text": "article",
            "text": "article",
            "document": "text"
        }

        if input_field in aliases and aliases[input_field] in sample:
            input_field = aliases[input_field]
        else:
            raise KeyError(
                f"[extract_text] Field '{input_field}' not found in sample.\n"
                f"Available fields: {list(sample.keys())}"
            )

    text = sample[input_field]

    text = str(text)
    text = text.replace("\r", " ").replace("\t", " ").strip()

    return " ".join(text.split())


def extract_summary(sample: Dict[str, Any], summary_field: str) -> str:
    """
    Extracts the summary text.
    """
    if summary_field not in sample:
        aliases = {
            "abstract_text": "abstract",
            "summary_text": "summary",
            "highlights": "summary"
        }
        
        if summary_field in aliases and aliases[summary_field] in sample:
            summary_field = aliases[summary_field]
        ## Generic fallback: ArXiv always has 'abstract'
        elif "abstract" in sample:
            summary_field = "abstract"
        else:
            raise KeyError(
                f"[extract_summary] Summary field '{summary_field}' not found.\n"
                f"Available fields: {list(sample.keys())}"
            )
        
    summary = sample[summary_field]

    if isinstance(summary, dict):
        # Try common keys
        for key in ["summary", "summary_text", "abstract", "highlights"]:
            if key in summary:
                summary = summary[key]
                break

    summary = str(summary).replace("\r", " ").replace("\t", " ").strip()
    return " ".join(summary.split())

def load_dataset(dataset_name: str):
    """Loads the dataset from disk by a given name."""

    path = os.path.join(DATA_PATH, dataset_name)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset {dataset_name} not found at {path}. Please run download_datasets.py first.")
    
    ds = load_from_disk(path)
    print(f"Loaded dataset {dataset_name} from {path}.")
    return ds

def chunk_text(text: str, max_tokens: int = 512, model_name: str = "facebook/bart-large-cnn") -> List[str]:
    "Splits a long text into roughly equal token-sized chunks."

    tokenizer = get_tokenizer(model_name)

    sentences = nltk.sent_tokenize(text)
    chunks, current_chunk, token_count = [], [], 0

    space_tokens = len(tokenizer.encode(" ", add_special_tokens=False))

    for sent in sentences:
        ## Counts tokens correctly
        tokens = len(tokenizer.encode(sent, add_special_tokens=False))
        
        ## Add space overhead if not the first sentence
        added_cost = tokens + (space_tokens if current_chunk else 0)

        if token_count + added_cost > max_tokens:
            chunks.append(" ".join(current_chunk))  ## To make a chunk when max tokens exceeded
            current_chunk, token_count = [sent], tokens
        else:
            current_chunk.append(sent)
            token_count += added_cost

    if current_chunk:
        chunks.append(" ".join(current_chunk))  ## To make a final chunk if any sentences remains i.e last chunk was full with less than a chunk size sent remaining

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
      """
    pairs = []

    for idx, sample in tqdm(enumerate(dataset), total=len(dataset), desc=f"[prepare] {dataset_name}"):
        text = extract_text(sample, input_field)
        summary = extract_summary(sample, summary_field)

        raw_id = f"{dataset_name}_{idx}"

        base_id = raw_id.replace("/", "_").replace(" ", "_")

        if chunk:       ## Chunking if document too big.
            if "gov" in dataset_name or "arxiv" in dataset_name:
                model_name = "allenai/led-base-16384"
            else:
                model_name = "facebook/bart-large-cnn"
            chunks = chunk_text(text, max_tokens = max_tokens, model_name=model_name)
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