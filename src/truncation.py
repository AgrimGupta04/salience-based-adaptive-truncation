"""
truncation.py 

Implements adaptive truncation for salience-based summarization.
Implements the exact algorithm described in the paper.

Author: Agrim Gupta
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import os
import json
from tqdm import tqdm
import random

DATA_PATH = "data/processed/"
SALIENCE_PATH = os.path.join(DATA_PATH, "salience_scores/")
TRUNCATED_PATH = os.path.join(DATA_PATH, "truncated_texts/")
os.makedirs(TRUNCATED_PATH, exist_ok=True)

# Token budgets from paper
TOKEN_BUDGETS = {
    "cnn_dailymail": 512,
    "govreport": 4096,
    "arxiv": 4096
}

def load_salience_scores(dataset_name: str, salience_type: str) -> Dict[str, float]:
    """Load salience scores from JSON file."""
    path = os.path.join(SALIENCE_PATH, f"{dataset_name}_{salience_type}_salience.json")
    
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Salience scores for {dataset_name} ({salience_type}) not found at {path}.\n"
            f"Please run salience_scoring.py first."
        )
    
    with open(path, "r", encoding="utf-8") as f:
        salience_data = json.load(f)
    
    if not isinstance(salience_data, list):
        raise ValueError(f"Salience file must be a list, got {type(salience_data)}")
    
    # Create mapping from chunk ID to score
    scores = {}
    for item in salience_data:
        chunk_id = item.get("id")
        if chunk_id:
            scores[chunk_id] = float(item.get("salience_score", 0.0))
    
    print(f"✓ Loaded {len(scores)} salience scores for {dataset_name} ({salience_type})")
    return scores

def load_pairs_with_token_counts(dataset_name: str) -> List[dict]:
    """Load preprocessed pairs with proper token counts."""
    pairs_path = os.path.join(DATA_PATH, f"{dataset_name}_pairs.json")
    
    if not os.path.exists(pairs_path):
        raise FileNotFoundError(
            f"Pairs not found at {pairs_path}. Run prepare_data in data_loader.py first."
        )
    
    with open(pairs_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Handle both old format (list of pairs) and new format (dict with pairs)
    if isinstance(data, dict) and "pairs" in data:
        pairs = data["pairs"]
    else:
        pairs = data
    
    # Ensure each pair has token count
    for pair in pairs:
        if "tokens" not in pair:
            # Estimate token count (words * 1.33)
            pair["tokens"] = int(len(pair["text"].split()) * 1.33)
    
    print(f"✓ Loaded {len(pairs)} pairs for {dataset_name}")
    return pairs

def group_chunks_by_document(pairs: List[dict]) -> Dict[str, List[dict]]:
    """
    Groups chunked text samples by their original document.
    Expects chunk IDs like 'dataset_docid_chunkidx'.
    """
    groups = {}
    
    for sample in pairs:
        sample_id = sample["id"]
        
        # Extract document ID (remove chunk index)
        parts = sample_id.split("_")
        if len(parts) >= 3:
            # Format: dataset_docid_chunkidx
            doc_id = "_".join(parts[:-1])
            chunk_idx = int(parts[-1])
        else:
            # No chunking was done
            doc_id = sample_id
            chunk_idx = 0
        
        # Add chunk index for sorting
        sample_with_idx = sample.copy()
        sample_with_idx["chunk_idx"] = chunk_idx
        
        # Group by document
        if doc_id not in groups:
            groups[doc_id] = []
        groups[doc_id].append(sample_with_idx)
    
    # Sort chunks within each document by chunk index
    for doc_id in groups:
        groups[doc_id].sort(key=lambda x: x["chunk_idx"])
    
    print(f"  Grouped into {len(groups)} documents")
    
    # Print statistics
    chunk_counts = [len(chunks) for chunks in groups.values()]
    if chunk_counts:
        print(f"  Avg chunks per doc: {np.mean(chunk_counts):.1f}")
        print(f"  Max chunks per doc: {max(chunk_counts)}")
    
    return groups

def select_chunks_by_salience(
    chunks: List[dict], 
    scores: Dict[str, float], 
    token_budget: int
) -> Tuple[List[dict], int, float]:
    """
    Select chunks in descending order of salience score until token budget is reached.
    Returns: (selected_chunks, total_tokens, total_salience)
    """
    if not chunks:
        return [], 0, 0.0
    
    # Score each chunk
    scored_chunks = []
    for chunk in chunks:
        chunk_id = chunk["id"]
        chunk_score = scores.get(chunk_id, 0.0)
        chunk_tokens = chunk.get("tokens", 0)
        
        if chunk_tokens > 0:
            scored_chunks.append({
                "chunk": chunk,
                "score": chunk_score,
                "tokens": chunk_tokens,
                "id": chunk_id
            })
    
    # Sort by score descending
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    
    # Greedy selection: pick highest-scoring chunks until budget is reached
    selected = []
    total_tokens = 0
    total_score = 0.0
    
    for item in scored_chunks:
        if total_tokens + item["tokens"] <= token_budget:
            selected.append(item["chunk"])
            total_tokens += item["tokens"]
            total_score += item["score"]
    
    # If no chunks selected (all too large), pick the smallest chunk
    if not selected and scored_chunks:
        smallest = min(scored_chunks, key=lambda x: x["tokens"])
        selected = [smallest["chunk"]]
        total_tokens = smallest["tokens"]
        total_score = smallest["score"]
    
    # Sort selected chunks by original order
    selected.sort(key=lambda x: x["chunk_idx"])
    
    return selected, total_tokens, total_score

def select_chunks_first_k(
    chunks: List[dict], 
    token_budget: int
) -> Tuple[List[dict], int]:
    """
    Select chunks from the beginning until token budget is reached.
    """
    selected = []
    total_tokens = 0
    
    for chunk in chunks:  # Already in original order
        chunk_tokens = chunk.get("tokens", 0)
        
        if total_tokens + chunk_tokens <= token_budget:
            selected.append(chunk)
            total_tokens += chunk_tokens
        else:
            break
    
    # Ensure at least one chunk
    if not selected and chunks:
        selected = [chunks[0]]
        total_tokens = chunks[0].get("tokens", 0)
    
    return selected, total_tokens

def select_chunks_random_k(
    chunks: List[dict], 
    token_budget: int,
    seed: int = 42
) -> Tuple[List[dict], int]:
    """
    Randomly select chunks until token budget is reached.
    Uses multiple random seeds for stability (averaged in evaluation).
    """
    rng = random.Random(seed)
    
    # Create a copy to shuffle
    chunks_copy = chunks.copy()
    rng.shuffle(chunks_copy)
    
    selected = []
    total_tokens = 0
    
    for chunk in chunks_copy:
        chunk_tokens = chunk.get("tokens", 0)
        
        if total_tokens + chunk_tokens <= token_budget:
            selected.append(chunk)
            total_tokens += chunk_tokens
    
    # Ensure at least one chunk
    if not selected and chunks:
        selected = [chunks[0]]
        total_tokens = chunks[0].get("tokens", 0)
    
    # Restore original order
    selected.sort(key=lambda x: x["chunk_idx"])
    
    return selected, total_tokens

def assemble_truncated_text(selected_chunks: List[dict]) -> str:
    """
    Concatenates selected chunks into a single truncated text string.
    Adds paragraph breaks between chunks for readability.
    """
    if not selected_chunks:
        return ""
    
    # Join with double newlines to preserve paragraph structure
    texts = [chunk["text"].strip() for chunk in selected_chunks]
    return "\n\n".join(texts)

def truncate_dataset(
    dataset_name: str, 
    token_budget: Optional[int] = None,
    truncation_method: str = "salience",
    salience_type: Optional[str] = None,
    seed: int = 42
) -> dict:
    """
    Main truncation function that implements adaptive truncation.
    
    Args:
        dataset_name: Name of dataset
        token_budget: Maximum tokens (default from paper)
        truncation_method: "salience", "first_k", "random_k", "lead_n"
        salience_type: "tfidf", "cosine", "hybrid" (required for salience method)
        seed: Random seed for random_k
    
    Returns:
        Statistics dictionary
    """
    # Set default token budget from paper
    if token_budget is None:
        token_budget = TOKEN_BUDGETS.get(dataset_name, 512)
    
    print(f"\n{'='*60}")
    print(f"TRUNCATING: {dataset_name}")
    print(f"Method: {truncation_method} | Budget: {token_budget} tokens")
    if salience_type:
        print(f"Salience type: {salience_type}")
    print(f"{'='*60}")
    
    # Load pairs and group by document
    pairs = load_pairs_with_token_counts(dataset_name)
    groups = group_chunks_by_document(pairs)
    
    # Load salience scores if needed
    scores = None
    if truncation_method == "salience":
        if not salience_type:
            raise ValueError("salience_type required for salience truncation")
        scores = load_salience_scores(dataset_name, salience_type)
    
    # Process each document
    truncated_records = []
    tokens_before_list = []
    tokens_after_list = []
    salience_scores_list = []
    
    for doc_id, chunks in tqdm(groups.items(), desc=f"Truncating {dataset_name}"):
        # Calculate original document tokens
        doc_tokens_before = sum(chunk.get("tokens", 0) for chunk in chunks)
        tokens_before_list.append(doc_tokens_before)
        
        # Select chunks based on method
        if truncation_method == "salience":
            selected, doc_tokens_after, total_salience = select_chunks_by_salience(
                chunks, scores, token_budget
            )
            salience_scores_list.append(total_salience)
            
        elif truncation_method == "first_k":
            selected, doc_tokens_after = select_chunks_first_k(chunks, token_budget)
            
        elif truncation_method == "random_k":
            selected, doc_tokens_after = select_chunks_random_k(chunks, token_budget, seed)
            
        elif truncation_method == "lead_n":
            # Same as first_k for sentence-aligned chunks
            selected, doc_tokens_after = select_chunks_first_k(chunks, token_budget)
            
        else:
            raise ValueError(f"Unknown truncation_method: {truncation_method}")
        
        # Assemble truncated text
        truncated_text = assemble_truncated_text(selected)
        
        # Get reference summary (from first chunk)
        reference_summary = chunks[0]["summary"] if chunks else ""
        
        # Create record
        record = {
            "id": doc_id,
            "truncated_text": truncated_text,
            "reference_summary": reference_summary,
            "tokens_before": doc_tokens_before,
            "tokens_after": doc_tokens_after,
            "kept_chunks": len(selected),
            "total_chunks": len(chunks),
            "truncation_method": truncation_method,
            "salience_type": salience_type if truncation_method == "salience" else None,
            "token_budget": token_budget,
            "compression_ratio": doc_tokens_after / doc_tokens_before if doc_tokens_before > 0 else 0
        }
        
        truncated_records.append(record)
        tokens_after_list.append(doc_tokens_after)
    
    # Calculate statistics
    avg_tokens_before = float(np.mean(tokens_before_list))
    avg_tokens_after = float(np.mean(tokens_after_list))
    percentage_reduction = 100 * (1 - avg_tokens_after / avg_tokens_before) if avg_tokens_before > 0 else 0
    
    stats = {
        "dataset": dataset_name,
        "truncation_method": truncation_method,
        "salience_type": salience_type,
        "token_budget": token_budget,
        "documents_processed": len(truncated_records),
        "avg_tokens_before": avg_tokens_before,
        "avg_tokens_after": avg_tokens_after,
        "percentage_reduction": percentage_reduction,
        "avg_compression_ratio": avg_tokens_after / avg_tokens_before if avg_tokens_before > 0 else 0,
        "avg_chunks_kept": np.mean([r["kept_chunks"] for r in truncated_records]),
        "avg_total_chunks": np.mean([r["total_chunks"] for r in truncated_records])
    }
    
    if salience_scores_list:
        stats["avg_salience_score"] = float(np.mean(salience_scores_list))
    
    # Save truncated data
    save_truncated_data(dataset_name, truncated_records, token_budget, truncation_method, salience_type)
    
    # Print summary
    print(f"\n✅ Truncation Complete for {dataset_name}")
    print(f"   Documents: {len(truncated_records)}")
    print(f"   Avg tokens before: {avg_tokens_before:.1f}")
    print(f"   Avg tokens after: {avg_tokens_after:.1f}")
    print(f"   Token reduction: {percentage_reduction:.1f}%")
    print(f"   Compression ratio: {stats['avg_compression_ratio']:.3f}")
    
    return stats

def save_truncated_data(
    dataset_name: str,
    truncated_records: List[dict],
    token_budget: int,
    truncation_method: str,
    salience_type: Optional[str] = None
):
    """Save truncated dataset to JSON file."""
    # Create filename
    if truncation_method == "salience":
        filename = f"{dataset_name}_{salience_type}_salience_token_budget_{token_budget}_truncated.json"
    else:
        filename = f"{dataset_name}_{truncation_method}_budget_{token_budget}.json"
    
    path = os.path.join(TRUNCATED_PATH, filename)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(truncated_records, f, indent=2)
    
    print(f"💾 Saved truncated data to: {path}")
    print(f"   File size: {os.path.getsize(path) / 1024:.1f} KB")

def test_truncation():
    """Test the truncation functions."""
    print("\n🧪 Testing truncation functions...")
    
    # Create test chunks
    test_chunks = [
        {"id": "test_0_0", "text": "First chunk of text.", "tokens": 50, "chunk_idx": 0},
        {"id": "test_0_1", "text": "Second chunk with more content.", "tokens": 70, "chunk_idx": 1},
        {"id": "test_0_2", "text": "Third chunk that is less important.", "tokens": 60, "chunk_idx": 2},
        {"id": "test_0_3", "text": "Fourth and final chunk.", "tokens": 40, "chunk_idx": 3},
    ]
    
    # Test scores
    test_scores = {
        "test_0_0": 0.8,
        "test_0_1": 0.9,
        "test_0_2": 0.3,
        "test_0_3": 0.6
    }
    
    print(f"\nTest chunks: {len(test_chunks)} chunks, total {sum(c['tokens'] for c in test_chunks)} tokens")
    
    # Test salience-based selection
    selected, tokens, score = select_chunks_by_salience(test_chunks, test_scores, 120)
    print(f"\nSalience selection (budget=120):")
    print(f"  Selected {len(selected)} chunks, {tokens} tokens, score={score:.2f}")
    for chunk in selected:
        print(f"    - {chunk['id']}: {chunk['tokens']} tokens")
    
    # Test first_k selection
    selected, tokens = select_chunks_first_k(test_chunks, 120)
    print(f"\nFirst-k selection (budget=120):")
    print(f"  Selected {len(selected)} chunks, {tokens} tokens")
    
    # Test random_k selection
    selected, tokens = select_chunks_random_k(test_chunks, 120, seed=42)
    print(f"\nRandom-k selection (budget=120, seed=42):")
    print(f"  Selected {len(selected)} chunks, {tokens} tokens")
    
    return True

if __name__ == "__main__":
    # Run tests
    test_truncation()
    
    # Example: Run truncation for CNN with TF-IDF
    print("\n" + "="*60)
    print("EXAMPLE: Truncating CNN/DailyMail with TF-IDF salience")
    print("="*60)
    
    try:
        stats = truncate_dataset(
            dataset_name="cnn_dailymail",
            token_budget=512,
            truncation_method="salience",
            salience_type="tfidf"
        )
        print(f"\nStats: {stats}")
    except Exception as e:
        print(f"⚠ Example failed (need data first): {e}")
        print("\n💡 To run full truncation:")
        print("   1. Run download_datasets.py")
        print("   2. Run data_loader.py")
        print("   3. Run salience_scoring.py")
        print("   4. Then run truncation.py with your parameters")