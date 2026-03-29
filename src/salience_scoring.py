"""
salience_scoring.py

Computes importance (salience) scores for each text chunk using:
- TF-IDF relevance to summary
- Cosine similarity (embedding-based)
- Hybrid combination of both
Outputs scored results for use in truncation.
"""

from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import minmax_scale
from sentence_transformers import SentenceTransformer
import json
import os
from tqdm import tqdm
import time

DATA_PATH = "data/processed/"
EMBEDDING_PATH = os.path.join(DATA_PATH, "embeddings")


def compute_salience_with_timing(pairs, method, model=None, embedding_path=None, alpha=0.7):
    """Wrapper that times salience computation and returns scores + timing dict."""
    start = time.perf_counter()
    scores = compute_salience(pairs, method, model, embedding_path, alpha)
    elapsed = time.perf_counter() - start
    
    n_docs = len(set(p["id"].rsplit("_", 1)[0] for p in pairs))
    timing = {
        "method": method,
        "total_sec": round(elapsed, 4),
        "per_doc_sec": round(elapsed / max(n_docs, 1), 6),
        "n_docs": n_docs,
        "n_chunks": len(pairs)
    }
    print(f"[timing] {method}: {elapsed:.2f}s total | {timing['per_doc_sec']:.4f}s/doc | {n_docs} docs")
    return scores, timing


def compute_tfidf_salience(pairs, summary_field = "summary") -> List[float]:
    """Sentences that share more key terms (bigrams) with the reference summary are likely to be more salient.

    1. Vectorize all chunks and the summary using TF-IDF (bigrams).
    2. Compute cosine similarity between each chunk and the summary.
    3. Normalize scores to [0,1].
    
    Returns:
        List of salience scores (float) for each text chunk.
    """

    texts = [p["text"] for p in pairs]

    ## extract document ID (everything except last _chunkindex)
    doc_ids = [p["id"].rsplit("_", 1)[0] for p in pairs]

    ## map doc → summary
    doc_to_summary = {}
    for p, doc in zip(pairs, doc_ids):
        if doc not in doc_to_summary:     ## same summary repeated across chunks
            doc_to_summary[doc] = p["summary"]

    summaries = [doc_to_summary[doc] for doc in doc_ids]

    vectorizer = TfidfVectorizer(
        ngram_range = (1, 2),
        stop_words = "english",
        max_features = 6000
    )

    tfidf_text = vectorizer.fit_transform(texts)
    tfidf_summary = vectorizer.transform(summaries)

    scores = []
    for i in range(len(texts)):
        sim = (tfidf_text[i] @ tfidf_summary[i].T).toarray().ravel()[0]
        scores.append(sim)

    return minmax_scale(scores)  ## Normalize to [0,1]

def compute_cosine_salience(pairs, model, embedding_path) -> List[float]:
    """Chunks semantically cloesest to the summary embedding are more salient.
    We want to measure how semantically similar each document chunk is to its summary.
    
    1. Load precomputed embeddings from {dataset_name}_embeddings.npy from the data folder.
    2. Compute cosine similarity with the mean summary embedding(encode the summanry with the same model).
    3. Normalize scores to [0,1].
    """

    embeddings = np.load(embedding_path)

    ids_path = embedding_path.replace("_embeddings.npy", "_ids.json")
    with open(ids_path, "r", encoding = "utf-8") as f:
        embedding_ids = json.load(f)

    pair_ids = [p["id"] for p in pairs]
    assert embedding_ids == pair_ids, (
        "Embedding pair ID mismatch! "
        "Embeddings do not align with pairs.json ordering."
    )

    ## extract document IDs
    doc_ids = [p["id"].rsplit("_", 1)[0] for p in pairs]

    ## unique doc -> summary
    doc_to_summary = {}
    for p, doc in zip(pairs, doc_ids):
        if doc not in doc_to_summary:
            doc_to_summary[doc] = p["summary"]

    unique_docs = list(doc_to_summary.keys())
    unique_summaries = [doc_to_summary[d] for d in unique_docs]

    ## Encode summary embeddings in batches
    summary_embs = model.encode(
        unique_summaries,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=True
    )

    ## Map: doc → summary embedding
    doc_to_emb = {doc: summary_embs[i] for i, doc in enumerate(unique_docs)}

    ## Build aligned array
    aligned_summary_embs = np.vstack([doc_to_emb[doc] for doc in doc_ids])

    ## embeddings from embeddings.npy are normalized
    cosine_scores = np.sum(embeddings * aligned_summary_embs, axis=1)

    return minmax_scale(cosine_scores)  ## Normalize to [0,1]

def compute_hybrid_salience(tfidf_scores, cosine_scores, alpha = 0.7) -> List[float]:
    """Combines TF-IDF and Cosine similarity scores using a weighted sum.
    
    Args:
        tfidf_scores: List of TF-IDF salience scores.
        cosine_scores: List of Cosine similarity salience scores.
        alpha: Weight for cosine scores (0 <= alpha <= 1).

    hybrid_score = alpha * np.array(cosine_scores) + (1 - alpha) * np.array(tfidf_scores)    
    """

    return minmax_scale(alpha * np.array(cosine_scores) + (1 - alpha) * np.array(tfidf_scores))

def compute_salience(
    pairs,
    method: str,
    model: SentenceTransformer = None,
    embedding_path: str = None,
    alpha: float = 0.7
) -> List[float]:
    """
    Entry-point for salience computation.
    """

    if method == "tfidf":
        return compute_tfidf_salience(pairs)

    if method == "cosine":
        if model is None or embedding_path is None:
            raise ValueError("Cosine salience requires model and embedding_path")
        return compute_cosine_salience(pairs, model, embedding_path)

    if method == "hybrid":
        if model is None or embedding_path is None:
            raise ValueError("Hybrid salience requires model and embedding_path")

        tfidf = compute_tfidf_salience(pairs)
        cosine = compute_cosine_salience(pairs, model, embedding_path)
        return compute_hybrid_salience(tfidf, cosine, alpha)

    raise ValueError(f"Unknown salience method: {method}")


def save_salience_scores(scores, ids, dataset_name, salience_type, save_dir = "data/processed/salience_scores/"):
    """Saves salience scores to a JSON file."""

    os.makedirs(save_dir, exist_ok = True)
    
    salience_data = [{"id": id_, "salience_score": float(score)} for id_, score in zip(ids, scores)]
    save_path = os.path.join(save_dir, f"{dataset_name}_{salience_type}_salience.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(salience_data, f, indent=2)

    print(f"Saved salience scores to {save_path}")

# def main():
#     model = SentenceTransformer("all-MiniLM-L6-v2")
#     salience_methods = ["tfidf", "cosine", "hybrid"]

#     for ds in tqdm(["cnn_dailymail", "govreport", "arxiv"], desc = "Datasets"):
#         with open(f"data/processed/{ds}_pairs.json", "r", encoding="utf-8") as f:
#             pairs = json.load(f)
            
#         embed_file = os.path.join(EMBEDDING_PATH, f"{ds}_embeddings.npy")
#         if not os.path.exists(embed_file):
#             raise FileNotFoundError(f"Missing embedding file: {embed_file}")
        
#         ids = [p["id"] for p in pairs]

#         for method in salience_methods:
#             scores = compute_salience(
#                 pairs,
#                 method=method,
#                 model=model,
#                 embedding_path=embed_file
#             )
#             save_salience_scores(scores, ids, ds, method)

# if __name__ == "__main__":
#     main()        