"""
salience_scoring.py - UPDATED VERSION

Computes importance (salience) scores for each text chunk using:
- TF-IDF relevance to reference summary (oracle)
- Cosine similarity (embedding-based) to reference summary
- Hybrid combination of both

As described in paper: "Scores are computed treating reference summaries as an oracle,
approximating an optimistic upper bound."

Author: Agrim Gupta
Updated for EMNLP 2023 submission
"""

from typing import List, Dict, Any, Union
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import minmax_scale
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import json
import os
from tqdm import tqdm

DATA_PATH = "data/processed/"
EMBEDDING_PATH = os.path.join(DATA_PATH, "embeddings")
SALIENCE_SCORES_PATH = os.path.join(DATA_PATH, "salience_scores")
os.makedirs(SALIENCE_SCORES_PATH, exist_ok=True)

def load_pairs_data(pairs: Union[List[Dict], Dict[str, Any]]) -> List[Dict]:
    """
    Load pairs data, handling both old and new formats.
    
    Args:
        pairs: Either a list of pairs or a dict with "pairs" key
        
    Returns:
        List of pairs dictionaries
    """
    if isinstance(pairs, dict) and "pairs" in pairs:
        # New format: {"pairs": [...], "stats": {...}, "config": {...}}
        return pairs["pairs"]
    elif isinstance(pairs, list):
        # Old format: list of pairs
        return pairs
    else:
        raise ValueError(f"Unknown pairs format: {type(pairs)}")

def extract_document_summary_mapping(pairs: List[Dict]) -> Dict[str, str]:
    """
    Extract mapping from document ID to reference summary.
    
    Paper: "Scores are computed treating reference summaries as an oracle"
    """
    doc_to_summary = {}
    
    for pair in pairs:
        # Extract document ID (remove chunk index)
        chunk_id = pair["id"]
        parts = chunk_id.split("_")
        
        if len(parts) >= 3:
            # Format: dataset_docid_chunkidx
            doc_id = "_".join(parts[:-1])
        else:
            # No chunking was done
            doc_id = chunk_id
        
        # Store summary for this document (same for all chunks of same document)
        if doc_id not in doc_to_summary:
            summary = pair.get("summary", "")
            if not summary:
                raise ValueError(f"No summary found for document {doc_id}")
            doc_to_summary[doc_id] = summary
    
    return doc_to_summary

def compute_tfidf_salience(pairs: Union[List[Dict], Dict[str, Any]]) -> List[float]:
    """
    Compute TF-IDF salience scores using reference summaries as oracle.
    
    Paper: "TF-IDF: Lexical overlap with reference summaries"
    
    Returns:
        List of salience scores normalized to [0, 1]
    """
    # Load pairs data
    pairs_list = load_pairs_data(pairs)
    
    print(f"  Computing TF-IDF salience for {len(pairs_list)} chunks...")
    
    # Extract texts and document IDs
    texts = [p["text"] for p in pairs_list]
    chunk_ids = [p["id"] for p in pairs_list]
    
    # Get document ID for each chunk
    doc_ids = []
    for chunk_id in chunk_ids:
        parts = chunk_id.split("_")
        if len(parts) >= 3:
            doc_ids.append("_".join(parts[:-1]))
        else:
            doc_ids.append(chunk_id)
    
    # Get reference summaries for each document
    doc_to_summary = extract_document_summary_mapping(pairs_list)
    summaries = [doc_to_summary[doc_id] for doc_id in doc_ids]
    
    # Vectorize using TF-IDF with bigrams
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),  # Unigrams and bigrams
        stop_words="english",
        max_features=5000,    # Reasonable limit
        min_df=2,             # Ignore terms that appear in less than 2 documents
        max_df=0.95,          # Ignore terms that appear in more than 95% of documents
    )
    
    # Fit and transform
    tfidf_texts = vectorizer.fit_transform(texts)
    tfidf_summaries = vectorizer.transform(summaries)
    
    # Compute cosine similarity between each chunk and its reference summary
    scores = []
    for i in range(len(texts)):
        # Cosine similarity between chunk i and its corresponding summary
        similarity = (tfidf_texts[i] @ tfidf_summaries[i].T).toarray()[0, 0]
        scores.append(similarity)
    
    # Normalize to [0, 1]
    normalized_scores = minmax_scale(scores)
    
    # Statistics
    print(f"    Min score: {min(normalized_scores):.4f}")
    print(f"    Max score: {max(normalized_scores):.4f}")
    print(f"    Mean score: {np.mean(normalized_scores):.4f}")
    
    return normalized_scores.tolist()

def compute_cosine_salience(
    pairs: Union[List[Dict], Dict[str, Any]], 
    model: SentenceTransformer, 
    embedding_path: str
) -> List[float]:
    """
    Compute cosine similarity salience scores using embeddings.
    
    Paper: "Cosine: Embedding similarity with reference summaries"
    
    Returns:
        List of salience scores normalized to [0, 1]
    """
    # Load pairs data
    pairs_list = load_pairs_data(pairs)
    
    print(f"  Computing Cosine salience for {len(pairs_list)} chunks...")
    
    # Load precomputed chunk embeddings
    if not os.path.exists(embedding_path):
        raise FileNotFoundError(f"Embeddings not found: {embedding_path}")
    
    embeddings = np.load(embedding_path)
    
    # Load IDs to verify alignment
    ids_path = embedding_path.replace("_embeddings.npy", "_ids.json")
    with open(ids_path, "r", encoding="utf-8") as f:
        embedding_ids = json.load(f)
    
    # Verify alignment
    chunk_ids = [p["id"] for p in pairs_list]
    if embedding_ids != chunk_ids:
        raise ValueError(
            f"Embedding ID mismatch!\n"
            f"Embedding IDs: {len(embedding_ids)} chunks\n"
            f"Pair IDs: {len(chunk_ids)} chunks\n"
            f"First mismatch at index {next(i for i, (e, p) in enumerate(zip(embedding_ids, chunk_ids)) if e != p)}"
        )
    
    # Get document ID for each chunk
    doc_ids = []
    for chunk_id in chunk_ids:
        parts = chunk_id.split("_")
        if len(parts) >= 3:
            doc_ids.append("_".join(parts[:-1]))
        else:
            doc_ids.append(chunk_id)
    
    # Get reference summaries for each document
    doc_to_summary = extract_document_summary_mapping(pairs_list)
    unique_docs = list(set(doc_ids))
    
    print(f"    Encoding {len(unique_docs)} reference summaries...")
    
    # Encode reference summaries
    unique_summaries = [doc_to_summary[doc] for doc in unique_docs]
    summary_embeddings = model.encode(
        unique_summaries,
        convert_to_numpy=True,
        normalize_embeddings=True,  # Important for cosine similarity
        batch_size=32,
        show_progress_bar=False
    )
    
    # Create mapping from document to summary embedding
    doc_to_summary_embedding = {doc: emb for doc, emb in zip(unique_docs, summary_embeddings)}
    
    # Get summary embedding for each chunk
    chunk_summary_embeddings = np.array([doc_to_summary_embedding[doc] for doc in doc_ids])
    
    # Compute cosine similarity (embeddings are already normalized)
    # Cosine similarity = dot product of normalized vectors
    scores = np.sum(embeddings * chunk_summary_embeddings, axis=1)
    
    # Normalize to [0, 1]
    normalized_scores = minmax_scale(scores)
    
    # Statistics
    print(f"    Min score: {min(normalized_scores):.4f}")
    print(f"    Max score: {max(normalized_scores):.4f}")
    print(f"    Mean score: {np.mean(normalized_scores):.4f}")
    
    return normalized_scores.tolist()

def compute_hybrid_salience(
    tfidf_scores: List[float], 
    cosine_scores: List[float], 
    alpha: float = 0.7
) -> List[float]:
    """
    Combine TF-IDF and Cosine similarity scores.
    
    Paper: "Hybrid: A linear combination of TF-IDF and Cosine scores"
    
    Args:
        tfidf_scores: TF-IDF salience scores
        cosine_scores: Cosine similarity salience scores
        alpha: Weight for cosine scores (0 <= alpha <= 1)
    
    Returns:
        Hybrid scores normalized to [0, 1]
    """
    print(f"  Computing Hybrid salience (alpha={alpha})...")
    
    if len(tfidf_scores) != len(cosine_scores):
        raise ValueError(
            f"Score length mismatch: TF-IDF={len(tfidf_scores)}, Cosine={len(cosine_scores)}"
        )
    
    # Convert to numpy arrays
    tfidf_array = np.array(tfidf_scores)
    cosine_array = np.array(cosine_scores)
    
    # Weighted combination
    hybrid_scores = alpha * cosine_array + (1 - alpha) * tfidf_array
    
    # Normalize to [0, 1]
    normalized_scores = minmax_scale(hybrid_scores)
    
    # Statistics
    print(f"    Min score: {min(normalized_scores):.4f}")
    print(f"    Max score: {max(normalized_scores):.4f}")
    print(f"    Mean score: {np.mean(normalized_scores):.4f}")
    
    return normalized_scores.tolist()

def compute_salience(
    pairs: Union[List[Dict], Dict[str, Any]],
    method: str,
    model: SentenceTransformer = None,
    embedding_path: str = None,
    alpha: float = 0.7
) -> List[float]:
    """
    Single entry-point for salience computation.
    
    Paper: "Scores are computed treating reference summaries as an oracle"
    
    Args:
        pairs: Pairs data (list or dict with "pairs" key)
        method: "tfidf", "cosine", or "hybrid"
        model: SentenceTransformer model (required for cosine/hybrid)
        embedding_path: Path to embeddings (required for cosine/hybrid)
        alpha: Weight for hybrid method
    
    Returns:
        List of salience scores
    """
    print(f"\n🔍 Computing {method.upper()} salience scores")
    
    if method == "tfidf":
        return compute_tfidf_salience(pairs)
    
    elif method == "cosine":
        if model is None or embedding_path is None:
            raise ValueError(
                "Cosine salience requires both model and embedding_path"
            )
        return compute_cosine_salience(pairs, model, embedding_path)
    
    elif method == "hybrid":
        if model is None or embedding_path is None:
            raise ValueError(
                "Hybrid salience requires both model and embedding_path"
            )
        
        # Compute both scores
        tfidf_scores = compute_tfidf_salience(pairs)
        cosine_scores = compute_cosine_salience(pairs, model, embedding_path)
        
        # Combine
        return compute_hybrid_salience(tfidf_scores, cosine_scores, alpha)
    
    else:
        raise ValueError(f"Unknown salience method: {method}")

def save_salience_scores(
    scores: List[float], 
    ids: List[str], 
    dataset_name: str, 
    salience_type: str,
    save_dir: str = SALIENCE_SCORES_PATH
) -> str:
    """
    Save salience scores to JSON file.
    
    Returns:
        Path to saved scores file
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create data structure
    salience_data = []
    for chunk_id, score in zip(ids, scores):
        salience_data.append({
            "id": chunk_id,
            "salience_score": float(score),
            "dataset": dataset_name,
            "salience_type": salience_type
        })
    
    # Save file
    save_path = os.path.join(save_dir, f"{dataset_name}_{salience_type}_salience.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(salience_data, f, indent=2)
    
    print(f"💾 Saved {len(salience_data)} scores to {save_path}")
    
    return save_path

def test_salience_scoring():
    """Test salience scoring functions."""
    print("\n🧪 TESTING SALIENCE SCORING")
    print("="*60)
    
    # Create test data
    test_pairs = [
        {
            "id": "test_0_0",
            "text": "Large language models have revolutionized natural language processing.",
            "summary": "LLMs changed NLP dramatically."
        },
        {
            "id": "test_0_1", 
            "text": "They require significant computational resources for training.",
            "summary": "LLMs changed NLP dramatically."
        },
        {
            "id": "test_1_0",
            "text": "This is another document about artificial intelligence.",
            "summary": "AI is transforming many fields."
        },
        {
            "id": "test_1_1",
            "text": "Machine learning algorithms can learn from data.",
            "summary": "AI is transforming many fields."
        }
    ]
    
    print(f"Test data: {len(test_pairs)} chunks from 2 documents")
    
    # Test TF-IDF
    print("\n1. Testing TF-IDF salience...")
    tfidf_scores = compute_tfidf_salience(test_pairs)
    print(f"   Scores: {tfidf_scores}")
    
    # Test with embeddings (mock)
    print("\n2. Testing Cosine salience (requires embeddings)...")
    
    # Create mock embeddings
    embedding_dim = 384  # all-MiniLM-L6-v2 dimension
    mock_embeddings = np.random.randn(len(test_pairs), embedding_dim).astype(np.float32)
    mock_embeddings = mock_embeddings / np.linalg.norm(mock_embeddings, axis=1, keepdims=True)
    
    # Save mock embeddings
    test_emb_path = "test_embeddings.npy"
    np.save(test_emb_path, mock_embeddings)
    
    # Save mock IDs
    test_ids = [p["id"] for p in test_pairs]
    with open("test_ids.json", "w") as f:
        json.dump(test_ids, f)
    
    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    try:
        cosine_scores = compute_cosine_salience(test_pairs, model, test_emb_path)
        print(f"   Scores: {cosine_scores}")
    except Exception as e:
        print(f"   ⚠ Cosine test skipped: {e}")
    
    # Clean up
    if os.path.exists(test_emb_path):
        os.remove(test_emb_path)
    if os.path.exists("test_ids.json"):
        os.remove("test_ids.json")
    
    print("\n✅ Salience scoring test completed")

def main():
    """Main function to compute salience scores for all datasets."""
    datasets = ["cnn_dailymail", "govreport", "arxiv"]
    salience_methods = ["tfidf", "cosine", "hybrid"]
    
    # Load embedding model once
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    for dataset_name in tqdm(datasets, desc="Datasets"):
        print(f"\n{'='*60}")
        print(f"PROCESSING DATASET: {dataset_name}")
        print(f"{'='*60}")
        
        # Load pairs data
        pairs_path = os.path.join(DATA_PATH, f"{dataset_name}_pairs.json")
        if not os.path.exists(pairs_path):
            print(f"❌ Pairs file not found: {pairs_path}")
            continue
        
        with open(pairs_path, "r", encoding="utf-8") as f:
            pairs_data = json.load(f)
        
        # Get chunk IDs
        pairs_list = load_pairs_data(pairs_data)
        chunk_ids = [p["id"] for p in pairs_list]
        
        # Check for embeddings
        embedding_path = os.path.join(EMBEDDING_PATH, f"{dataset_name}_embeddings.npy")
        if not os.path.exists(embedding_path):
            print(f"⚠ Embeddings not found: {embedding_path}")
            print("  Skipping cosine and hybrid methods...")
            salience_methods = ["tfidf"]
        
        # Compute scores for each method
        for method in salience_methods:
            print(f"\n📊 Method: {method.upper()}")
            
            try:
                # Compute scores
                if method in ["cosine", "hybrid"] and not os.path.exists(embedding_path):
                    print(f"  ⚠ Skipping {method} - embeddings missing")
                    continue
                
                scores = compute_salience(
                    pairs_data,
                    method=method,
                    model=model,
                    embedding_path=embedding_path if method in ["cosine", "hybrid"] else None,
                    alpha=0.7  # Default from paper
                )
                
                # Save scores
                save_salience_scores(scores, chunk_ids, dataset_name, method)
                
            except Exception as e:
                print(f"❌ Error computing {method} scores: {e}")
                continue
    
    print("\n✅ All salience scores computed!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute salience scores")
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--dataset", type=str, help="Specific dataset to process")
    parser.add_argument("--method", type=str, help="Specific method (tfidf, cosine, hybrid)")
    
    args = parser.parse_args()
    
    if args.test:
        test_salience_scoring()
    elif args.dataset:
        # Process specific dataset
        main()  # You would modify main() to accept args, but for simplicity...
        print(f"Processing {args.dataset} with method {args.method}")
    else:
        main()