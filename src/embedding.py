from sentence_transformers import SentenceTransformer
import numpy as np 
import os
import json

def load_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Loads a sentence transformer model for embeddings."""
    print(f" Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    return model

def build_embedding_index(pairs_file, model, save_dir = "data/processed/embeddings"):
    """Encodes each text sample and stores embeddings in locally."""

    with open(pairs_file, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    texts = [p["text"] for p in pairs]
    ids = [p["id"] for p in pairs]
    base = os.path.basename(pairs_file)
    dataset_name = base.replace("_pairs.json", "")

    
    os.makedirs(save_dir, exist_ok = True)

    embeddings = model.encode(      ## Encode datasets in batches 
        texts, batch_size = 32, show_progress_bar = True, convert_to_numpy = True
    )       ## This is a 2D Numpy array shaped like (N, D) where N -> the number of input texts(chunks) encoded and D -> the dimensionality of each embedding vector.
            ## For all-MiniLM-L6-v2 the D = 384, so if 10000 chunks (10000, 384)

    emb_path = os.path.join(save_dir, f"{dataset_name}_embeddings.npy")
    ids_path = os.path.join(save_dir, f"{dataset_name}_ids.json")

    np.save(emb_path, embeddings)
    with open(ids_path, "w", encoding = "utf-8") as f:
        json.dump(ids, f, indent=2)

    print(f"Saved embeddings for {dataset_name} with shape {embeddings.shape}.")
    print(f"Embedding files - {emb_path}, {ids_path}")
    return emb_path, ids_path