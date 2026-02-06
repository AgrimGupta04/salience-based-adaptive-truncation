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
# import tiktoken

from transformers import AutoTokenizer

DATA_PATH = "data/processed/"
SALIENCE_PATH = os.path.join(DATA_PATH, "salience_scores/")
TRUNCATED_PATH = os.path.join(DATA_PATH, "truncated_texts/")

# enc = tiktoken.get_encoding("cl100k_base")

def get_project_tokenizer(dataset_name: str):
    """
    Returns the tokenizer matching the model used for summarization.
    CNN -> BART
    GovReport/ArXiv -> LED
    """
    if "cnn" in dataset_name.lower():
        # print("[Truncator] Loading BART tokenizer for CNN...")
        return AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    else:
        # print("[Truncator] Loading LED tokenizer for GovReport/ArXiv...")
        return AutoTokenizer.from_pretrained("allenai/led-base-16384")


def load_salience_scores(dataset_name: str, salience_type: str) -> Dict[str, float]:
    """Read salience scores from JSON and return maping from chunk ID to score.
    
    1. Load salience scores from {dataset_name}_salience_scores.json.
    2. Validate format -> expect list of objects {"id": id, "salience_score": score}.
    3. Returns dictionary {id: float(score)}.
    """

    path = os.path.join(SALIENCE_PATH, f"{dataset_name}_{salience_type}_salience.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Salience scores for {dataset_name} not found at {path}. Please run salience_scoring.py first.")
    
    with open(path, "r", encoding="utf-8") as f:
        salience_data = json.load(f)
        assert isinstance(salience_data, list), "Salience file must be a list of dicts"

    return {d["id"]: float(d.get("salience_score", 0.0)) for d in salience_data}

def group_chunks_by_document(pairs: List[dict], tokenizer) -> Dict[str, List[dict]]:
    """
    Groups chunked text samples (from prepare_data) by their original dicument.
    Expects chunk IDS like '<docid>_<chunkindex>'.
    """

    groups = {}
    for sample in pairs:
        sample_id = sample["id"]
        doc_id = "_".join(sample_id.split("_")[:-1])  if "_" in sample_id else sample_id
        token_count = len(tokenizer.encode(sample["text"], add_special_tokens=False))
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
            scored.append((ch, s/max(tc, 1)))

    scored = sorted(scored, key=lambda x: x[1], reverse=True)

    selected, used_tokens = [], 0
    for ch, _ in scored:
        if used_tokens + ch["token_count"] > token_budget:
            continue
        selected.append(ch)
        used_tokens += ch["token_count"]

    if not selected:
        selected = [min(chunks, key=lambda x: x["token_count"])]  ## Ensure at least one chunk is selected.

    selected = sorted(selected, key = lambda x: int(x["id"].split("_")[-1])  if "_" in x["id"] else 0)  ## Restore original order
    return selected

def assemble_truncated_text(selected_chunks: List[dict]) -> str:
    """
    Concatenates selected chunks into a single truncated text string.
    """
    return "\n\n".join([ch["text"] for ch in selected_chunks])

def truncate_dataset(dataset_name: str, token_budget: int, truncation_method: str, salience_type:str | None = None, seed: int = 42):
    """
    This module converts salience scores into adaptively truncated inputs
    under a fixed token budget. Chunks are selected greedily to maximize
    salience per token, producing token quality tradeoff curves.

    Args:
    dataset_name: Name of dataset (e.g., "cnn_dailymail")
    token_budget: Maximum number of tokens allowed after truncation
    salience_type: Salience scoring method used ("tfidf", "cosine", "hybrid")
    """

    pairs_path = os.path.join(DATA_PATH, f"{dataset_name}_pairs.json")
    if not os.path.exists(pairs_path):
        raise FileNotFoundError(f"Pairs not found at {pairs_path}. Run prepare_data first.")

    with open(pairs_path, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    print(f"Loaded {len(pairs)} samples from {pairs_path}")
    print(f"[truncate] {dataset_name} | salience={salience_type} | budget={token_budget}")

    tokenizer = get_project_tokenizer(dataset_name)
    groups = group_chunks_by_document(pairs, tokenizer)
    
    scores = None
    if truncation_method == "salience":
        scores = load_salience_scores(dataset_name, salience_type)

    truncated_records = []
    tokens_before, tokens_after = [], []

    checkpoint_every = 500
    processed = 0

    for doc_id, chunks in tqdm(groups.items(), desc=f"Truncating {dataset_name}"):

        if truncation_method == "salience":
            selected = select_top_tokens_by_score(chunks, scores, token_budget)

        elif truncation_method == "first_k":
            selected = select_first_k_tokens(chunks, token_budget)

        elif truncation_method == "random_k":
            selected = select_random_k_tokens(chunks, token_budget, seed)

        elif truncation_method == "lead_n":
            selected = select_lead_n_chunks(chunks, token_budget)

        else:
            raise ValueError(f"Unknown truncation_method: {truncation_method}")

        truncated_text = assemble_truncated_text(selected)

        rec = {
            "id": doc_id,
            "truncated_text": truncated_text,
            "summary": chunks[0]["summary"],
            "tokens_before": sum(c["token_count"] for c in chunks),
            "tokens_after": sum(c["token_count"] for c in selected),
            "kept_chunk_count": len(selected),
            "salience_type": salience_type if truncation_method == "salience" else truncation_method,
            "token_budget": token_budget,
        }

        truncated_records.append(rec)
        tokens_before.append(rec["tokens_before"])
        tokens_after.append(rec["tokens_after"])

        processed += 1
        if processed % checkpoint_every == 0:
            print(f"[checkpoint] Saving partial — {processed} docs...")
            save_truncated(dataset_name, truncated_records, token_budget, truncation_method, salience_type)

    save_truncated(dataset_name, truncated_records, token_budget, truncation_method, salience_type)
    stats = {
        "dataset": dataset_name,
        "salience_type": salience_type if truncation_method == "salience" else truncation_method,
        "token_budget": token_budget,
        "avg_tokens_before": float(np.mean(tokens_before)),
        "avg_tokens_after": float(np.mean(tokens_after)),
        "percentage_reduction": 100* (1 - float(np.mean(tokens_after)) / float(np.mean(tokens_before))),
    }

    print(f"[{dataset_name}] Avg tokens reduced by {stats['percentage_reduction']:.2f}%")
    return stats

def save_truncated(dataset_name: str, truncated_records: List[dict], token_budget: int, truncation_method: str, salience_type:str | None = None):
    """
    Saves truncated dataset to JSON file.
    """
    os.makedirs(TRUNCATED_PATH, exist_ok=True)
    if truncation_method == "salience":
        name = f"{dataset_name}_salience_{salience_type}_budget_{token_budget}.json"
    else:
        name = f"{dataset_name}_{truncation_method}_budget_{token_budget}.json"

    path = os.path.join(TRUNCATED_PATH, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(truncated_records, f, indent=2)
    print(f"Saved truncated dataset to {path}")


def select_first_k_tokens(chunks: List[dict], token_budget: int) -> List[dict]:
    selected, used = [], 0
    for ch in chunks:  # already in original order
        if used + ch["token_count"] > token_budget:
            break
        selected.append(ch)
        used += ch["token_count"]
    return selected if selected else [chunks[0]]


def select_random_k_tokens(chunks: List[dict], token_budget: int, seed: int = 42) -> List[dict]:
    rng = np.random.default_rng(seed)
    shuffled = list(chunks)
    rng.shuffle(shuffled)

    selected, used = [], 0
    for ch in shuffled:
        if used + ch["token_count"] > token_budget:
            continue
        selected.append(ch)
        used += ch["token_count"]

    if not selected:
        selected = [min(chunks, key=lambda x: x["token_count"])]

    # restore original order
    return sorted(selected, key=lambda x: int(x["id"].split("_")[-1]))


def select_lead_n_chunks(chunks: List[dict], token_budget: int) -> List[dict]:
    return select_first_k_tokens(chunks, token_budget)



# def main():
#     configs = [
#         ("cnn_dailymail", 512),
#         ("govreport", 2048),
#         ("arxiv", 4096),
#     ]
#     salience_types = ["tfidf", "cosine", "hybrid"]

#     results = []
    
#     for salience_type in salience_types:
#         for ds, budget in configs:
#             stats = truncate_dataset(ds, token_budget = budget, salience_type = salience_type)
#             results.append(stats)

#     ## Save global stats
#     os.makedirs("results", exist_ok=True)
#     with open("results/truncation_stats.json", "w", encoding="utf-8") as f:
#         json.dump(results, f, indent=2)
#     print("All truncations complete. Stats saved to results/truncation_stats.json")


# if __name__ == "__main__":
#     main()