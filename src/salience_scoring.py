"""
salience_scoring.py
-------------------
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

DATA_PATH = "data/processed/"
EMBEDDING_PATH = os.path.join(DATA_PATH, "embeddings")

def compute_tfidf_salience(pairs, summary_field = "summary") -> List[float]:
    """Sentences that share more key terms (bigrams) with the reference summary are likely to be more salient.

    1. Vectorize all chunks and the summary using TF-IDF (bigrams).
    2. Compute cosine similarity between each chunk and the summary.
    3. Normalize scores to [0,1].
    
    Returns:
        List of salience scores (float) for each text chunk.
    """

    doc_ids = [p["id"].rsplit("_", 1)[0] for p in pairs] 

    unique_docs = list(dict.fromkeys(doc_ids))
    doc_to_summary = {doc: pairs[doc_ids.index(doc)]["summary"] for doc in unique_docs}

    summaries = [doc_to_summary[doc_ids[i]] for i in range(len(pairs))]
    texts = [p["text"] for p in pairs]

    vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english', max_features = 6000)
    tfidf_text = vectorizer.fit_transform(texts)
    tfidf_summary = vectorizer.transform(summaries)
    n = len(texts)      ## To use as an index to separate text and summary vectors
    salience_scores = []

    for i in range(n):
        text_vec = tfidf_text[i]
        summary_vec = tfidf_summary[i]
        similarity_score = (text_vec @ summary_vec.T).toarray().ravel()[0]
        salience_scores.append(similarity_score)
    return minmax_scale(salience_scores)  ## Normalize to [0,1]

def compute_cosine_salience(pairs, model, embedding_path) -> List[float]:
    """Chunks semantically cloesest to the summary embedding are more salient.
    We want to measure how semantically similar each document chunk is to its summary.
    
    1. Load precomputed embeddings from {dataset_name}_embeddings.npy from the data folder.
    2. Compute cosine similarity with the mean summary embedding(encode the summanry with the same model).
    3. Normalize scores to [0,1].
    """

    doc_ids = [p["id"].rsplit("_", 1)[0] for p in pairs]
    unique_docs = list(dict.fromkeys(doc_ids))
    doc_to_summary = {doc: pairs[doc_ids.index(doc)]["summary"] for doc in unique_docs}
    summaries = [doc_to_summary[doc_ids[i]] for i in range(len(pairs))]
    embeddings = np.load(embedding_path)
    summary_embeddings = model.encode(summaries, convert_to_numpy = True, show_progress_bar = True)
    denom = (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(summary_embeddings, axis=1)) + 1e-8
    cosine_scores = np.sum(embeddings * summary_embeddings, axis=1) / denom


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

def save_salience_scores(scores, ids, dataset_name, save_dir = "data/processed/salience_scores/"):
    """Saves salience scores to a JSON file."""

    os.makedirs(save_dir, exist_ok = True)
    salience_data = [{"id": id_, "salience_score": score} for id_, score in zip(ids, scores)]
    save_path = os.path.join(save_dir, f"{dataset_name}_salience_scores.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(salience_data, f, indent=2)
    print(f"Saved salience scores to {save_path}")

def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    for ds in tqdm(["cnn_dailymail", "govreport", "arxiv"], desc = "Datasets"):
        with open(f"data/processed/{ds}_pairs.json", "r", encoding="utf-8") as f:
            pairs = json.load(f)
            
        tfidf_scores = compute_tfidf_salience(pairs)
        embed_file = os.path.join(EMBEDDING_PATH, f"{ds}_embeddings.npy")
        if not os.path.exists(embed_file):
            raise FileNotFoundError(f"Missing embedding file: {embed_file}")
        
        cosine_scores = compute_cosine_salience(
            pairs, model, os.path.join(EMBEDDING_PATH, f"{ds}_embeddings.npy")
        )
        hybrid = compute_hybrid_salience(tfidf_scores, cosine_scores)
        ids = [p["id"] for p in pairs]
        save_salience_scores(hybrid, ids, ds)

if __name__ == "__main__":
    main()        