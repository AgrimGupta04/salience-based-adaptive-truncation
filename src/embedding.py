from sentence_transformers import SentenceTransformer
import numpy as np 
import os
import json
import torch

def load_embedding_model(model_name: str = "all-MiniLM-L6-v2", device: str = None) -> SentenceTransformer:
    """
    Loads a SentenceTransformer model with explicit device handling.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f" Loading embedding model: {model_name} on {device}")
    model = SentenceTransformer(model_name, device=device)
    return model

def build_embedding_index(pairs_file, model, save_dir = "data/processed/embeddings", batch_size: int = 32):
    """Encodes each text sample and stores embeddings in locally."""

    with open(pairs_file, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    texts = [p["text"] for p in pairs]
    ids = [p["id"] for p in pairs]
    base = os.path.basename(pairs_file)
    dataset_name = base.replace("_pairs.json", "")
    
    os.makedirs(save_dir, exist_ok = True)

    ## Output paths
    emb_path = os.path.join(save_dir, f"{dataset_name}_embeddings.npy")
    ids_path = os.path.join(save_dir, f"{dataset_name}_ids.json")

    print(f"Encoding {len(texts)} chunks for dataset: {dataset_name}")
    print(f"Batch size: {batch_size}")

    ## Memory-safe incremental embedding
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_emb = model.encode(
            batch_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        all_embeddings.append(batch_emb)

    embeddings = np.vstack(all_embeddings)

    ## Save files
    np.save(emb_path, embeddings)
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(ids, f, indent=2)

    print(f"\nSaved embeddings!")
    print(f" - Embedding shape: {embeddings.shape}")
    print(f" - Embeddings file: {emb_path}")
    print(f" - IDs file: {ids_path}")

    return emb_path, ids_path
