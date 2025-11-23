"""
scripts/tune_alpha.py
Grid-search tuning of alpha (α) for hybrid salience scoring (token-budget version).
"""

import os
import json
import argparse
import numpy as np
from tqdm import tqdm

from src.salience_scoring import compute_hybrid_salience
from src.truncation import truncate_dataset
from src.summarizer import load_summarization_model, summarize_truncated_files
from src.evaluation import compute_rouge
from scripts.run_full_pipeline import choose_summarization_model_from_truncated


def load_base_scores(dataset):
    """Loads tfidf + cosine stored earlier when building salience."""
    base_dir = "data/processed/salience_scores"
    tfidf_p = os.path.join(base_dir, f"{dataset}_tfidf_scores.json")
    cosine_p = os.path.join(base_dir, f"{dataset}_cosine_scores.json")

    if not (os.path.exists(tfidf_p) and os.path.exists(cosine_p)):
        raise FileNotFoundError("TF-IDF and cosine scores missing. Run salience scoring first.")

    tfidf = json.load(open(tfidf_p))
    cosine = json.load(open(cosine_p))

    ids = [d["id"] for d in tfidf]
    tfidf_vals = np.array([d["salience_score"] for d in tfidf])
    cosine_vals = np.array([d["salience_score"] for d in cosine])

    return ids, tfidf_vals, cosine_vals


def evaluate_truncated(dataset, budget, model_name, max_eval=20):
    path = f"data/processed/truncated_texts/{dataset}_token_budget_{budget}_truncated.json"
    data = json.load(open(path))

    data = data[:max_eval]
    scorer_input_ref = []
    scorer_input_pred = []

    pipe = load_summarization_model(model_name)

    # texts = [d["truncated_text"] for d in data]
    preds = summarize_truncated_files(path, model_pipe=pipe, batch_size=4, return_outputs=True)

    for rec, pred in zip(data, preds):
        scorer_input_ref.append(rec["reference"])
        scorer_input_pred.append(pred)

    scores = compute_rouge(scorer_input_ref, scorer_input_pred)
    return scores["rougeL"]


def tune_alpha(dataset, budget, alphas):
    ids, tfidf_vals, cosine_vals = load_base_scores(dataset)
    results = []

    for alpha in tqdm(alphas, desc=f"Tuning α for {dataset}"):
        hybrid = compute_hybrid_salience(tfidf_vals, cosine_vals, alpha)

        tmp_file = f"data/processed/salience_scores/{dataset}_hybrid_temp.json"
        json.dump([{"id": i, "salience_score": float(s)} for i, s in zip(ids, hybrid)], open(tmp_file, "w"))

        truncate_dataset(dataset_name=dataset, mode="token_budget", param=budget, salience_override=tmp_file)
        trunc_file = f"data/processed/truncated_texts/{dataset}_token_budget_{budget}_truncated.json"
        model = choose_summarization_model_from_truncated(trunc_file)

        rougeL = evaluate_truncated(dataset, budget, model)
        results.append({"alpha": alpha, "rougeL": rougeL})

    best = max(results, key=lambda x: x["rougeL"])
    return best, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[round(x * 0.1, 2) for x in range(0, 11)])
    args = parser.parse_args()

    best, rows = tune_alpha(args.dataset, args.budget, args.alphas)

    out_dir = "results/alpha_tuning"
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"{args.dataset}_alpha_results.csv")

    with open(out_csv, "w") as f:
        f.write("alpha,rougeL\n")
        for r in rows:
            f.write(f"{r['alpha']},{r['rougeL']}\n")

    print(f"\nBest α = {best['alpha']:.2f} (ROUGE-L={best['rougeL']:.4f})")
