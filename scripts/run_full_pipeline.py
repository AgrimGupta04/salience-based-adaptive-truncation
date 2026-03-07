"""
scripts/run_full_pipeline.py

End-to-end orchestration for the salience-based token-budget truncation experiments.
"""

import os
import sys
import json 
import numpy as np
from typing import Dict

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

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
    plot_tradeoff_curves_aggregated,
    plot_rouge_bars_aggregated,
    plot_cost_vs_budget_aggregated,
    plot_cost_vs_quality_aggregated,
    plot_token_distribution,
    plot_rouge_drop,
    plot_model_selection_histogram,
    plot_bootstrap_ci,
    plot_cost_at_quality_threshold
)
from src.utils import get_truncated_filename

SHARED_GEN_KWARGS = {
    "max_new_tokens": 512,
    "min_length": 150,
    "num_beams": 4, 
    "early_stopping": True, 
    "do_sample": False, 
    "no_repeat_ngram_size": 3, 
    "length_penalty": 2.0
}

TRUNCATION_METHODS = [
    ("salience", "tfidf"),
    ("salience", "cosine"),
    ("salience", "hybrid"),
    ("first_k", None),
    ("random_k", None),
    ("lead_n", None),
]

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
            "max_chunk_tokens": 512,
            "skip_full": False
        },
        "govreport": {
            "budget": 4096,
            "model": "allenai/led-base-16384",
            "input_field": "report",
            "summary_field": "summary",
            "max_chunk_tokens": 1024,
            "skip_full": False
        },
        "arxiv": {
            "budget": 4096,
            "model": "allenai/led-large-16384-arxiv",
            "input_field": "article",
            "summary_field": "abstract",
            "max_chunk_tokens": 1024,
            "chunk": True,
            "skip_full": False,
        }
    }

# ------------------------------
# 2. Prepare dataset
# ------------------------------
def run_prepare_data(dataset_name: str, cfg: dict) -> str:
    print(f"\n[prepare] {dataset_name} —> loading raw dataset and preparing pairs")
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

    if not cfg.get("skip_full", False):
        print(f"[prepare] {dataset_name} —> preparing FULL pairs (no chunking) for baseline")
        prepare_data(
            ds,
            dataset_name=dataset_name,
            input_field=cfg["input_field"],
            summary_field=cfg["summary_field"],
            chunk=False,  ## No chunking for baseline !!
            save_path=f"data/processed/{dataset_name}_full_pairs.json"
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
def run_salience_scoring(dataset_name: str, salience_type: str):
    print(f"\n[salience] computing salience scores for {dataset_name}")
    pairs_path = f"data/processed/{dataset_name}_pairs.json"
    if not os.path.exists(pairs_path):
        raise FileNotFoundError(pairs_path)
    pairs = json.load(open(pairs_path, "r", encoding="utf-8"))

    emb_path = f"data/processed/embeddings/{dataset_name}_embeddings.npy"
    emb_model = load_embedding_model()

    if salience_type == "tfidf":
        scores = compute_tfidf_salience(pairs)
    elif salience_type == "cosine":
        scores = compute_cosine_salience(pairs, emb_model, emb_path)
    elif salience_type == "hybrid":
        tfidf = compute_tfidf_salience(pairs)
        cosine = compute_cosine_salience(pairs, emb_model, emb_path)
        scores = compute_hybrid_salience(tfidf, cosine)
    else:
        raise ValueError(f"Unknown salience_type: {salience_type}")

    ids = [p["id"] for p in pairs]
    save_salience_scores(scores, ids, dataset_name, salience_type)
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

    if avg_tokens <= 1024:
        return "facebook/bart-large-cnn"
    return "allenai/led-base-16384"

# ------------------------------
# 5. Truncate dataset
# ------------------------------
def run_truncation(dataset_name: str, token_budget: int, truncation_method: str, salience_type: str):
    print(f"[truncate] {dataset_name} | method={truncation_method} | budget={token_budget}")
    stats = truncate_dataset(dataset_name=dataset_name, token_budget=token_budget, truncation_method=truncation_method, salience_type=salience_type)
    print(f"[truncate] done: avg_tokens_before={stats.get('avg_tokens_before'):.1f} avg_tokens_after={stats.get('avg_tokens_after'):.1f}")
    return stats

# ------------------------------
# 6. Summarization
# ------------------------------
def run_summarization(dataset_name: str, cfg: dict, truncation_method: str, salience_type: str):
    """
    Summarize full pairs (baseline) using cfg['model'], then choose model
    for truncated inputs and summarize them.
    """
    print(f"\n[summarize] running summarization for {dataset_name}")

    if not cfg.get("skip_full", False):
        baseline_model = cfg.get("model", "facebook/bart-large-cnn")
        print(f"[summarize] baseline model: {baseline_model}")
        baseline_pipe = load_summarization_model(baseline_model)

        pairs_file = f"data/processed/{dataset_name}_full_pairs.json"
        if os.path.exists(pairs_file):
            summarize_full_pairs(pairs_file, model_pipe=baseline_pipe, gen_kwargs=SHARED_GEN_KWARGS, batch_size=32, force=False)
        else:
            print(f"[summarize] WARNING: pairs file not found: {pairs_file}")
    else:
        print(f"[summarize] skipping full-input baseline for {dataset_name}")


    ## Choose summarizer for truncated inputs
    truncated_base = get_truncated_filename(
        dataset_name,
        truncation_method,
        salience_type,
        cfg["budget"]
    )
    truncated_file = f"data/processed/truncated_texts/{truncated_base}"

    if not os.path.exists(truncated_file):
        raise FileNotFoundError(f"[summarize] truncated file missing: {truncated_file}")

    if dataset_name == "arxiv":
        chosen_model = "allenai/led-large-16384-arxiv"
    else:
        chosen_model = choose_summarization_model_from_truncated(truncated_file)

    print(f"[summarize] chosen model for truncated: {chosen_model}")
    trunc_pipe = load_summarization_model(chosen_model)

    out_name = truncated_base.replace(".json", "_summaries")

    summarize_truncated_files(truncated_file, model_pipe=trunc_pipe, gen_kwargs=SHARED_GEN_KWARGS, batch_size=32, force=False, out_name=out_name)
    print(f"[summarize] summaries produced for {dataset_name}")

# ------------------------------
# 7. Evaluation
# ------------------------------
def run_evaluation(dataset_name: str, cfg: dict, truncation_method: str, salience_type: str):
    print(f"\n[evaluate] computing metrics for {dataset_name}")

    skip_full = cfg.get("skip_full", False)

    ## expected summary filenames created by summarizer functions
    full_summary_file = f"data/processed/summaries/{os.path.basename(f'{dataset_name}_pairs')}_full_summaries.json"
    truncated_base = get_truncated_filename(
        dataset_name,
        truncation_method,
        salience_type,
        cfg["budget"]
    )
    trunc_summary_file = f"data/processed/summaries/{truncated_base.replace('.json', '_summaries.json')}"


    candidates = [
        f"data/processed/summaries/{dataset_name}_full_pairs_full_summaries.json",
        f"data/processed/summaries/{dataset_name}_full_summaries.json",
        full_summary_file
    ]
    full_file = next((p for p in candidates if os.path.exists(p)), None)
    if skip_full:
        print(f"[evaluate] skipping baseline evaluation for {dataset_name}")
    elif full_file is None:
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
def run_visualization(metrics_csv: str):
    if not os.path.exists(metrics_csv): 
        print(f"[visualize] CSV {metrics_csv} not found.")
        return

    print(f"\n[visualize] Generating plots from {metrics_csv}...")
    df = load_metrics_csv(metrics_csv)
    os.makedirs("results/plots", exist_ok=True)

    plot_tradeoff_curves_aggregated(
        df, out_path="results/plots/tradeoff_curve.png"
    )
    
    plot_cost_vs_quality_aggregated(
        df, out_path="results/plots/cost_vs_quality.png"
    )
    
    plot_cost_vs_budget_aggregated(
        df, out_path="results/plots/cost_vs_budget.png"
    )
    
    plot_rouge_bars_aggregated(
        df, out_path="results/plots/rouge_bars.png"
    )
    
    plot_token_distribution(
        df, out_path="results/plots/token_distribution.png"
    )
    
    plot_rouge_drop(
        df, out_path="results/plots/rouge_drop.png"
    )
    
    plot_model_selection_histogram(
        out_path="results/plots/model_selection.png"
    )

    plot_cost_at_quality_threshold(
        df, out_path="results/plots/cost_threshold.png"
    )

    for ds in df['dataset_clean'].unique():
        plot_bootstrap_ci(ds)

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
            ## 1. Prepare
            run_prepare_data(dataset_name, cfg)

            ## 2. Embeddings
            run_build_embeddings(f"data/processed/{dataset_name}_pairs.json", dataset_name)
    
            for truncation_method, salience_type in TRUNCATION_METHODS:

                if truncation_method == "salience":
                    run_salience_scoring(dataset_name, salience_type)

                run_truncation(
                    dataset_name=dataset_name,
                    token_budget=cfg["budget"],
                    truncation_method=truncation_method,
                    salience_type=salience_type
                )

                run_summarization(dataset_name, cfg, truncation_method, salience_type)
                run_evaluation(dataset_name, cfg, truncation_method, salience_type)

            print(f"[main] completed dataset: {dataset_name}")

        except Exception as e:
            print(f"[main] ERROR processing {dataset_name}: {e}")

    run_visualization(metrics_csv="results/metrics.csv")
    print("\nALL DONE")

if __name__ == "__main__":
    main()
