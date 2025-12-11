"""
scripts/run_full_pipeline.py

End-to-end orchestration for the salience-based token-budget truncation experiments.

Usage: run this script from the repository root:
    python scripts/run_full_pipeline.py

It expects the repo layout you've been building (data/raw, data/processed, src/...).
"""

import os
import sys
import json 
import numpy as np
from typing import Dict

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import prepare_data, load_dataset
from src.embedding import load_embedding_model, build_embedding_index
from src.salience_scoring import (
    compute_tfidf_salience,
    compute_cosine_salience,
    compute_hybrid_salience,
    save_salience_scores,
)
from src.truncation import truncate_dataset
from src.summarizer import (
    load_summarization_model,
    summarize_full_pairs,
    summarize_truncated_files,
)
from src.evaluation import evaluate_summary_file, save_evaluation_results
from src.visualization import (
    load_metrics_csv,
    plot_quality_vs_compression,
    plot_rouge_bars,
    plot_token_distribution,
    plot_rouge_drop,
    plot_salience_heatmap,
    plot_model_selection_histogram,
    plot_bootstrap_ci
)

# ------------------------------
# 1. Dataset configuration
# ------------------------------
def load_dataset_config() -> Dict[str, dict]:
    """
    Edit this dict to add/remove datasets and set token budgets + default baseline model.
    Keep field names consistent with your download_datasets.py / data_loader mappings.
    """
    return {
        "cnn_dailymail": {
            "budget": 512,
            "model": "facebook/bart-large-cnn",
            "input_field": "article",
            "summary_field": "highlights",
            "max_chunk_tokens": 512
        },
        "govreport": {
            "budget": 2048,
            "model": "allenai/led-base-16384",
            "input_field": "report",
            "summary_field": "summary",
            "max_chunk_tokens": 1024
        },
        "arxiv": {
            "budget": 4096,
            "model": "allenai/led-base-16384",
            "input_field": "article",
            "summary_field": "abstract",
            "max_chunk_tokens": 1024,
            "chunk": True,
        }
    }

# ------------------------------
# 2. Prepare dataset
# ------------------------------
def run_prepare_data(dataset_name: str, cfg: dict) -> str:
    print(f"\n[prepare] {dataset_name} — loading raw dataset and preparing pairs")
    ds = load_dataset(dataset_name)
    pairs = prepare_data(
        ds,
        dataset_name=dataset_name,
        input_field=cfg["input_field"],
        summary_field=cfg["summary_field"],
        chunk=True,
        max_tokens=cfg.get("max_chunk_tokens", 512),
        save_path=f"data/processed/{dataset_name}_pairs.json"
    )
    out_path = f"data/processed/{dataset_name}_pairs.json"
    print(f"[prepare] saved {len(pairs)} pairs -> {out_path}")
    return out_path

# ------------------------------
# 3. Build embeddings
# ------------------------------
def run_build_embeddings(pairs_file: str, dataset_name: str):
    print(f"\n[embeddings] building embeddings for {dataset_name}")
    emb_model = load_embedding_model()
    emb, ids = build_embedding_index(pairs_file, emb_model, save_dir="data/processed/embeddings")
    print(f"[embeddings] saved embeddings shape {getattr(emb, 'shape', 'n/a')}")
    return emb, ids

# ------------------------------
# 4. Salience scoring
# ------------------------------
def run_salience_scoring(dataset_name: str):
    print(f"\n[salience] computing salience scores for {dataset_name}")
    pairs_path = f"data/processed/{dataset_name}_pairs.json"
    if not os.path.exists(pairs_path):
        raise FileNotFoundError(pairs_path)
    pairs = json.load(open(pairs_path, "r", encoding="utf-8"))

    emb_path = f"data/processed/embeddings/{dataset_name}_embeddings.npy"
    emb_model = load_embedding_model()

    tfidf_scores = compute_tfidf_salience(pairs)
    cosine_scores = compute_cosine_salience(pairs, emb_model, emb_path)
    hybrid_scores = compute_hybrid_salience(tfidf_scores, cosine_scores)

    ids = [p["id"] for p in pairs]
    save_salience_scores(hybrid_scores, ids, dataset_name)
    print(f"[salience] saved salience scores for {dataset_name}")

# ------------------------------
# Helper: choose summarizer based on truncated tokens
# ------------------------------
def choose_summarization_model_from_truncated(truncated_json_path: str) -> str:
    """
    Choose an HF model name depending on average tokens_after in truncated file.
    """
    if not os.path.exists(truncated_json_path):
        raise FileNotFoundError(truncated_json_path)
    with open(truncated_json_path, "r", encoding="utf-8") as f:
        recs = json.load(f)

    tokens = []
    for r in recs:
        ta = r.get("tokens_after")
        if ta is None:
            txt = r.get("truncated_text", "")
            ta = max(1, int(len(txt) / 4))   ## heuristic fallback
        tokens.append(ta)

    avg_tokens = float(np.mean(tokens)) if tokens else 1.0
    print(f"[choose_model] avg tokens_after={avg_tokens:.1f}")

    # thresholds (tunable)
    if avg_tokens <= 1024:
        return "facebook/bart-large-cnn"
    if avg_tokens <= 4096:
        return "google/long-t5-local-base"      ## supports longer input than BART
    if avg_tokens <= 16384:
        return "allenai/led-base-16384"
    return "allenai/led-base-16384"

# ------------------------------
# 5. Truncate dataset
# ------------------------------
def run_truncation(dataset_name: str, token_budget: int):
    print(f"\n[truncate] {dataset_name} -> token_budget={token_budget}")
    stats = truncate_dataset(dataset_name=dataset_name, token_budget=token_budget)
    print(f"[truncate] done: avg_tokens_before={stats.get('avg_tokens_before'):.1f} avg_tokens_after={stats.get('avg_tokens_after'):.1f}")
    return stats

# ------------------------------
# 6. Summarization
# ------------------------------
def run_summarization(dataset_name: str, cfg: dict):
    """
    Summarize full pairs (baseline) using cfg['model'], then choose model
    for truncated inputs and summarize them.
    """
    print(f"\n[summarize] running summarization for {dataset_name}")

    baseline_model = cfg.get("model", "facebook/bart-large-cnn")
    print(f"[summarize] baseline model: {baseline_model}")
    baseline_pipe = load_summarization_model(baseline_model)

    pairs_file = f"data/processed/{dataset_name}_pairs.json"
    if os.path.exists(pairs_file):
        summarize_full_pairs(pairs_file, model_pipe=baseline_pipe, batch_size=8)
    else:
        print(f"[summarize] WARNING: pairs file not found: {pairs_file}")

    # Choose summarizer for truncated inputs
    truncated_file = f"data/processed/truncated_texts/{dataset_name}_{cfg['budget']}_truncated_summaries.json"
    if not os.path.exists(truncated_file):
        raise FileNotFoundError(f"[summarize] truncated file missing: {truncated_file}")

    chosen_model = choose_summarization_model_from_truncated(truncated_file)
    print(f"[summarize] chosen model for truncated: {chosen_model}")
    trunc_pipe = load_summarization_model(chosen_model)

    summarize_truncated_files(truncated_file, model_pipe=trunc_pipe, batch_size=8)
    print(f"[summarize] summaries produced for {dataset_name}")

# ------------------------------
# 7. Evaluation
# ------------------------------
def run_evaluation(dataset_name: str, cfg: dict):
    print(f"\n[evaluate] computing metrics for {dataset_name}")

    # expected summary filenames created by summarizer functions
    full_summary_file = f"data/processed/summaries/{os.path.basename(f'{dataset_name}_pairs')}_full_summaries.json"
    trunc_summary_file = f"data/processed/summaries/{dataset_name}_{cfg['budget']}_truncated_summaries.json"

    # Prior code used slightly different filenames; handle possible variants
    # Try standard names first, fall back to alternative pattern
    candidates = [
        f"data/processed/summaries/{dataset_name}_pairs_full_summaries.json",
        f"data/processed/summaries/{dataset_name}_full_summaries.json",
        full_summary_file
    ]
    full_file = next((p for p in candidates if os.path.exists(p)), None)
    if full_file is None:
        print(f"[evaluate] WARNING: full summary file not found for {dataset_name}; skipping baseline evaluation")
    else:
        res_full = evaluate_summary_file(full_file)
        save_evaluation_results(res_full)

    if not os.path.exists(trunc_summary_file):
        print(f"[evaluate] WARNING: truncated summary file not found: {trunc_summary_file}")
    else:
        res_trunc = evaluate_summary_file(trunc_summary_file)
        save_evaluation_results(res_trunc)

# ------------------------------
# 8. Visualization
# ------------------------------
def run_visualization(metrics_csv: str = "results/metrics.csv"):
    if not os.path.exists(metrics_csv):
        print(f"[visualize] metrics CSV not found: {metrics_csv} — skipping visualization")
        return

    df = load_metrics_csv(metrics_csv)
    os.makedirs("results/plots", exist_ok=True)

    plot_quality_vs_compression(df, out_path="results/plots/quality_vs_compression.png")
    plot_rouge_bars(df, out_path="results/plots/rouge_bars.png")
    plot_token_distribution(df, out_path="results/plots/token_distribution.png")
    plot_model_selection_histogram()
    plot_rouge_drop(df, out_path="results/plots/rouge_drop.png")

    ## Per-dataset bootstrap CI plots
    for dataset_name in df["dataset"].unique():
        try:
            plot_bootstrap_ci(dataset_name)
            plot_salience_heatmap(dataset_name)
        except Exception as e:
            print(f"[visualize] Failed to generate CI plot for {dataset_name}: {e}")

    print("[visualize] plots saved to results/plots/")

# ------------------------------
# 9. Main orchestrator
# ------------------------------
def main():
    configs = load_dataset_config()
    os.makedirs("results", exist_ok=True)

    for dataset_name, cfg in configs.items():
        print("\n" + "=" * 80)
        print(f"RUNNING PIPELINE FOR: {dataset_name}")
        print("=" * 80)

        try:
            # 1. Prepare
            run_prepare_data(dataset_name, cfg)

            # 2. Embeddings
            run_build_embeddings(f"data/processed/{dataset_name}_pairs.json", dataset_name)

            # 3. Salience scoring
            run_salience_scoring(dataset_name)

            # 4. Truncation (token_budget)
            run_truncation(dataset_name, cfg["budget"])

            # 5. Summarization (baseline + truncated with auto model)
            run_summarization(dataset_name, cfg)

            # 6. Evaluation (appends to results/metrics.csv)
            run_evaluation(dataset_name, cfg)

            print(f"[main] completed dataset: {dataset_name}")

        except Exception as e:
            print(f"[main] ERROR processing {dataset_name}: {e}")
            # continue with next dataset

    # Final: make global plots from metrics.csv
    run_visualization(metrics_csv="results/metrics.csv")
    print("\nALL DONE")

if __name__ == "__main__":
    main()
