"""
embedding.py - UPDATED VERSION

Builds embeddings for document chunks using SentenceTransformer.
Compatible with the updated data format from data_loader.py.

Author: Agrim Gupta
"""

from sentence_transformers import SentenceTransformer
import numpy as np 
import os
import json
import torch
from tqdm import tqdm

def load_embedding_model(
    model_name: str = "all-MiniLM-L6-v2", 
    device: str = None
) -> SentenceTransformer:
    """
    Loads a SentenceTransformer model with explicit device handling.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"📦 Loading embedding model: {model_name} on {device}")
    model = SentenceTransformer(model_name, device=device)
    
    # Test the model with a small sample
    test_embedding = model.encode(["test"], convert_to_numpy=True)
    print(f"  Test embedding shape: {test_embedding.shape}")
    print(f"  Model dimension: {model.get_sentence_embedding_dimension()}")
    
    return model

def load_pairs_data(pairs_file: str):
    """
    Load pairs data, handling both old and new formats.
    
    Returns:
        texts: List of chunk texts
        ids: List of chunk IDs
        dataset_name: Name of dataset
    """
    with open(pairs_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Extract dataset name from filename
    base = os.path.basename(pairs_file)
    dataset_name = base.replace("_pairs.json", "")
    
    # Handle both old and new formats
    if isinstance(data, dict) and "pairs" in data:
        # New format: {"pairs": [...], "stats": {...}, "config": {...}}
        pairs = data["pairs"]
        print(f"  Detected new format: {len(pairs)} chunks")
    elif isinstance(data, list):
        # Old format: list of pairs
        pairs = data
        print(f"  Detected old format: {len(pairs)} chunks")
    else:
        raise ValueError(f"Unknown data format in {pairs_file}")
    
    # Extract texts and IDs
    texts = []
    ids = []
    
    for pair in pairs:
        if "text" not in pair:
            raise KeyError(f"Pair missing 'text' field: {pair.get('id', 'unknown')}")
        
        texts.append(pair["text"])
        ids.append(pair["id"])
    
    # Validate
    if len(texts) != len(ids):
        raise ValueError(f"Mismatch: {len(texts)} texts vs {len(ids)} IDs")
    
    return texts, ids, dataset_name

def build_embedding_index(
    pairs_file: str, 
    model: SentenceTransformer, 
    save_dir: str = "data/processed/embeddings", 
    batch_size: int = 32,
    normalize: bool = True
) -> tuple:
    """
    Encodes each text sample and stores embeddings locally.
    
    Args:
        pairs_file: Path to pairs JSON file
        model: Loaded SentenceTransformer model
        save_dir: Directory to save embeddings
        batch_size: Batch size for encoding
        normalize: Whether to normalize embeddings (recommended for cosine similarity)
    
    Returns:
        Tuple of (embeddings_path, ids_path)
    """
    print(f"\n{'='*60}")
    print(f"BUILDING EMBEDDING INDEX")
    print(f"Input: {pairs_file}")
    print(f"{'='*60}")
    
    # Load data
    texts, ids, dataset_name = load_pairs_data(pairs_file)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Output paths
    emb_path = os.path.join(save_dir, f"{dataset_name}_embeddings.npy")
    ids_path = os.path.join(save_dir, f"{dataset_name}_ids.json")
    meta_path = os.path.join(save_dir, f"{dataset_name}_metadata.json")
    
    print(f"\n📊 Dataset: {dataset_name}")
    print(f"  Chunks: {len(texts)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Model: {model.get_sentence_embedding_dimension()}D")
    
    # Check if embeddings already exist
    if os.path.exists(emb_path) and os.path.exists(ids_path):
        print(f"\n⚠ Embeddings already exist at {emb_path}")
        print("  Loading existing embeddings...")
        
        try:
            embeddings = np.load(emb_path)
            with open(ids_path, "r", encoding="utf-8") as f:
                existing_ids = json.load(f)
            
            if len(existing_ids) == len(ids) and existing_ids == ids:
                print("✅ Existing embeddings are valid and match current data")
                return emb_path, ids_path
            else:
                print("⚠ Existing embeddings don't match current data")
                print(f"  Existing: {len(existing_ids)} IDs")
                print(f"  Current: {len(ids)} IDs")
                print("  Rebuilding embeddings...")
        except Exception as e:
            print(f"⚠ Error loading existing embeddings: {e}")
            print("  Rebuilding embeddings...")
    
    # Encode in batches with progress bar
    print(f"\n🔧 Encoding {len(texts)} chunks...")
    
    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(texts), batch_size), total=total_batches, desc="Encoding"):
        batch_texts = texts[i:i + batch_size]
        
        # Encode batch
        batch_emb = model.encode(
            batch_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        ).astype(np.float32)
        
        all_embeddings.append(batch_emb)
    
    # Combine all embeddings
    embeddings = np.vstack(all_embeddings)
    
    # Validate dimensions
    expected_dim = model.get_sentence_embedding_dimension()
    if embeddings.shape[1] != expected_dim:
        raise ValueError(
            f"Embedding dimension mismatch: expected {expected_dim}, "
            f"got {embeddings.shape[1]}"
        )
    
    # Save embeddings
    np.save(emb_path, embeddings)
    
    # Save IDs
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(ids, f, indent=2)
    
    # Save metadata
    metadata = {
        "dataset_name": dataset_name,
        "num_chunks": len(texts),
        "embedding_dim": embeddings.shape[1],
        "model_name": model._first_module().auto_model.config._name_or_path,
        "normalized": normalize,
        "created_at": str(os.path.getctime(emb_path)),
        "file_sizes": {
            "embeddings_npy": os.path.getsize(emb_path) / (1024 * 1024),  # MB
            "ids_json": os.path.getsize(ids_path) / 1024,  # KB
        }
    }
    
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Embeddings saved!")
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Embeddings file: {emb_path} ({embeddings.nbytes / (1024*1024):.1f} MB)")
    print(f"  IDs file: {ids_path}")
    print(f"  Metadata: {meta_path}")
    
    return emb_path, ids_path

def load_embeddings(embeddings_path: str, ids_path: str):
    """
    Load precomputed embeddings and IDs.
    
    Returns:
        Tuple of (embeddings, ids)
    """
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")
    
    if not os.path.exists(ids_path):
        raise FileNotFoundError(f"IDs not found: {ids_path}")
    
    embeddings = np.load(embeddings_path)
    
    with open(ids_path, "r", encoding="utf-8") as f:
        ids = json.load(f)
    
    # Validate
    if len(embeddings) != len(ids):
        raise ValueError(
            f"Embedding/ID mismatch: {len(embeddings)} embeddings, "
            f"{len(ids)} IDs"
        )
    
    print(f"📥 Loaded {len(embeddings)} embeddings, shape: {embeddings.shape}")
    
    return embeddings, ids

def test_embedding():
    """Test embedding functions."""
    print("\n🧪 TESTING EMBEDDING MODULE")
    print("="*60)
    
    # Load model
    model = load_embedding_model("all-MiniLM-L6-v2")
    
    # Test encoding
    test_texts = [
        "This is a test sentence about natural language processing.",
        "Embeddings are useful for semantic similarity.",
        "Sentence transformers create good embeddings."
    ]
    
    print("\nTesting encoding...")
    embeddings = model.encode(test_texts, convert_to_numpy=True)
    print(f"  Input texts: {len(test_texts)}")
    print(f"  Embedding shape: {embeddings.shape}")
    print(f"  Sample embedding (first 5 dims): {embeddings[0][:5]}")
    
    # Test similarity
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(embeddings)
    print(f"\nCosine similarities:")
    for i in range(len(test_texts)):
        for j in range(i+1, len(test_texts)):
            sim = sim_matrix[i, j]
            print(f"  Text {i} vs Text {j}: {sim:.3f}")
    
    print("\n✅ Embedding test completed")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build embeddings for document chunks")
    parser.add_argument("--pairs", type=str, help="Path to pairs JSON file")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", 
                       help="SentenceTransformer model name")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--test", action="store_true", help="Run tests")
    
    args = parser.parse_args()
    
    if args.test:
        test_embedding()
    
    elif args.pairs:
        if not os.path.exists(args.pairs):
            print(f"❌ Pairs file not found: {args.pairs}")
            exit(1)
        
        # Load model
        model = load_embedding_model(args.model)
        
        # Build embeddings
        emb_path, ids_path = build_embedding_index(
            args.pairs, 
            model, 
            batch_size=args.batch
        )
        
        print(f"\n✅ Embeddings ready for use in salience scoring")
        print(f"   To use: embeddings, ids = load_embeddings('{emb_path}', '{ids_path}')")
    
    else:
        print("Usage:")
        print("  python embedding.py --test")
        print("  python embedding.py --pairs data/processed/cnn_dailymail_pairs.json")
        print("\nAvailable models:")
        print("  all-MiniLM-L6-v2 (default, 384D)")
        print("  all-mpnet-base-v2 (768D, better quality)")
        print("  paraphrase-MiniLM-L6-v2 (384D, tuned for similarity)")