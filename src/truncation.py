"""
truncation.py
----------------

This module turns salience scores inro actual trancated inputs for the summarizer. 
It's the experimental lever we vary to produce the token-quality tradeoff curves.
Selects adn keeps only top K chunks based on precomputed salience scores.
"""

from typing import List, Dict
import numpy as np
import os
import json
from tqdm import tqdm
## from chromadb import PersistentClient
import tiktoken

DATA_PATH = "data/processed/"
SALIENCE_PATH = os.path.join(DATA_PATH, "salience_scores/")
TRUNCATED_PATH = os.path.join(DATA_PATH, "truncated_texts/")

enc = tiktoken.get_encoding("cl100k_base")

def load_salience_scores(dataset_name: str) -> Dict[str, float]:
    """Read salience scores from JSON and return maping from chunk ID to score.
    
    1. Load salience scores from {dataset_name}_salience_scores.json.
    2. Validate format -> expect list of objects {"id": id, "salience_score": score}.
    3. Returns dictionary {id: float(score)}.
    """

    path = os.path.join(SALIENCE_PATH, f"{dataset_name}_salience_scores.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Salience scores for {dataset_name} not found at {path}. Please run salience_scoring.py first.")
    
    with open(path, "r", encoding="utf-8") as f:
        salience_data = json.load(f)

    return {d["id"]: float(d.get("salience_score", 0.0)) for d in salience_data}

def group_chunks_by_document(pairs: List[dict]) -> Dict[str, List[dict]]:
    """
    Groups chunked text samples (from prepare_data) by their original dicument.
    Expects chunk IDS like '<docid>_<chunkindex>'.
    """

    groups = {}
    for sample in pairs:
        sample_id = sample["id"]
        doc_id = "_".join(sample_id.split("_")[:-1])  if "_" in sample_id else sample_id
        token_count = len(enc.encode(sample["text"]))
        entry = {
            "id": sample_id,
            "doc_id": doc_id,
            "text": sample["text"],
            "summary": sample["summary"],
            "token_count": token_count
        }
        groups.setdefault(doc_id, []).append(entry)

    for doc in groups:
        groups[doc] = sorted(
            groups[doc],
            key=lambda ch: int(ch["id"].split("_")[-1]) if "_" in ch["id"] else 0
        )

    return groups

# def select_top_chunks_by_score(chunks: List[dict], scores: Dict[str, float], keep_ratio: float = 0.3) -> List[dict]:
#     """Selects top k% chunks for a single document based on salience scores.

#     Args:
#         chunks: List of chunk dicts for a single document.
#         scores: Mapping from chunk ID to salience score.
#         keep_ratio: Fraction of chunks to keep (0 < keep_ratio <= 1).
#     """

#     scored = [(ch, scores.get(ch["id"], 0.0)) for ch in chunks]
#     scored = sorted(scored, key=lambda x: x[1], reverse=True)
#     keep_n = max(1, int(len(chunks) * keep_ratio))
#     if len(chunks) < 3:
#         keep_n = len(chunks)    ## Keep all chunks if less than 3 chunks.
#     selected = [ch for ch, _ in scored[:keep_n]]         ## Select top k chunks based on score
#     selected = sorted(selected, key=lambda x: int(x["id"].split("_")[-1]) if "_" in x["id"] else 0)  ## Restore original order
#     return selected

def select_top_tokens_by_score(chunks: List[dict], scores: Dict[str, float], token_budget: int = 2048) -> List[dict]:
    """Select top chunks whose total token count is within the token budget., maximizing total salience score.
    
    1. Compute pre-chunk token counts score_per_token = score / token_count.
    2. Sort chunks b y score_per_token descending.
    3. Iteratively add chunks while sum(tokens) + chunk.token_chunk <= token_budget.
    4. Return selected chunks in original order.

    Args:
        chunks: List of chunk dicts for a single document.
        scores: Mapping from chunk ID to salience score.
        token_budget: Maximum total token count to keep.
    """

    scored = [] 
    for ch in chunks:
        s = scores.get(ch["id"], 0.0)
        tc = ch["token_count"]
        if tc > 0:
            scored.append((ch, s/tc))

    scored = sorted(scored, key=lambda x: x[1], reverse=True)

    selected, used_tokens = [], 0
    for ch, _ in scored:
        if used_tokens + ch["token_count"] > token_budget:
            continue
        selected.append(ch)
        used_tokens += ch["token_count"]

    if not selected:
        selected = [chunks[0]]

    selected = sorted(selected, key = lambda x: int(x["id"].split("_")[-1])  if "_" in x["id"] else 0)  ## Restore original order
    return selected

def assemble_truncated_text(selected_chunks: List[dict]) -> str:
    """
    Concatenates selected chunks into a single truncated text string.
    """
    return "\n\n".join([ch["text"] for ch in selected_chunks])

def truncate_dataset(dataset_name: str, token_budget: int):
    """
    Truncates dataset adaptively using precomputed salience scores.

    Args:
        dataset_name: name of dataset ("cnn_dailymail", etc.)
        token_budget: "token_budget"
        use_chroma: if True, uses ChromaDB for top-K retrieval
        chroma_path: path to persistent ChromaDB (optional)

    Returns:
        dict of summary statistics
    """

    pairs_path = os.path.join(DATA_PATH, f"{dataset_name}_pairs.json")
    if not os.path.exists(pairs_path):
        raise FileNotFoundError(f"Pairs not found at {pairs_path}. Run prepare_data first.")

    with open(pairs_path, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    print(f"Loaded {len(pairs)} samples from {pairs_path}")
    groups = group_chunks_by_document(pairs)
    scores = load_salience_scores(dataset_name)

    truncated_records = []
    tokens_before, tokens_after = [], []

    checkpoint_every = 500
    processed = 0

    for doc_id, chunks in tqdm(groups.items(), desc=f"Truncating {dataset_name}"):

        selected = select_top_tokens_by_score(
            chunks, scores, token_budget
        )
        truncated_text = assemble_truncated_text(selected)

        rec = {
            "id": doc_id,
            "truncated_text": truncated_text,
            "summary": chunks[0]["summary"],
            "tokens_before": sum(c["token_count"] for c in chunks),
            "tokens_after": sum(c["token_count"] for c in selected),
            "kept_chunk_count": len(selected),
        }

        truncated_records.append(rec)
        tokens_before.append(rec["tokens_before"])
        tokens_after.append(rec["tokens_after"])

        processed += 1
        if processed % checkpoint_every == 0:
            print(f"[checkpoint] Saving partial — {processed} docs...")
            save_truncated(dataset_name, truncated_records, token_budget)

    save_truncated(dataset_name, truncated_records, token_budget)
    stats = {
        "dataset": dataset_name,
        "token_budget": token_budget,
        "avg_tokens_before": float(np.mean(tokens_before)),
        "avg_tokens_after": float(np.mean(tokens_after)),
        "pct_reduction": 100* (1 - float(np.mean(tokens_after)) / float(np.mean(tokens_before))),
    }

    print(f"[{dataset_name}] Avg tokens reduced by {stats['pct_reduction']:.2f}%")
    return stats

def save_truncated(dataset_name: str, truncated_records: List[dict], token_budget: int):
    """
    Saves truncated dataset to JSON file.
    """
    os.makedirs(TRUNCATED_PATH, exist_ok=True)
    name = f"{dataset_name}_token_budget_{token_budget}_truncated_summaries.json"
    path = os.path.join(TRUNCATED_PATH, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(truncated_records, f, indent=2)
    print(f"Saved truncated dataset to {path}")

def main():
    configs = [
        ("cnn_dailymail", 512),
        ("govreport", 2048),
        ("arxiv", 4096),
    ]

    results = []
    for ds, budget in configs:
        stats = truncate_dataset(ds, token_budget = budget)
        results.append(stats)

    ## Save global stats
    os.makedirs("results", exist_ok=True)
    with open("results/truncation_stats.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("All truncations complete. Stats saved to results/truncation_stats.json")


if __name__ == "__main__":
    main()