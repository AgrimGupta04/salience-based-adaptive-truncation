import os 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
import json
from rouge_score import rouge_scorer
from src.evaluation import significance_test_bootstrap
import glob


def load_metrics_csv(csv_path: str)-> pd.DataFrame:
    """Loads evaluation metrics CSV into a DataFrame (produced by evaluation.py).
    
    Ensures poroper numeric types and automatically extracts dataset and mode info.
    """

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metrics CSV not found at {csv_path}. Please run evaluation.py first.")

    df = pd.read_csv(csv_path)

    required_cols = [
        "dataset",
        "token_budget",
        "rouge1",
        "rouge2",
        "rougeL",
        "bert_score_f1",
        "avg_tokens_before",
        "avg_tokens_after",
        "percentage_reduction"
    ]

    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in metrics CSV: {missing}")

    numeric_cols = [
        "rouge1", "rouge2", "rougeL",
        "bert_score_f1",
        "avg_tokens_before", "avg_tokens_after",
        "percentage_reduction"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["is_full"] = df["token_budget"].isna()

    df["compression_ratio"] = df["avg_tokens_after"] / df["avg_tokens_before"]
    df["token_budget"] = pd.to_numeric(df["token_budget"], errors="coerce")

    return df

def plot_quality_vs_compression(df: pd.DataFrame, out_path: str):
    """ROUGE VS Compression Ratio Plot.
    
    1. Shows how summary quality drops as token count shrinks.
    2. Each truncation mode (eg. keep_ratio = 0.3, 0.2, 0.1) creates a point on the plot.
    3. The curve shows the trade-off between quality and compression.
    """

    plt.figure(figsize=(10,6))
    sns.scatterplot(
        data = df,
        x = "compression_ratio",
        y = "rougeL",
        hue = "dataset",
        style=df["token_budget"].fillna("full"),
        marker = "o"
    )

    plt.title("ROUGE-L vs Compression Ratio")
    plt.xlabel("Compression Ratio (tokens_after / tokens_before)")
    plt.ylabel("ROUGE-L")
    plt.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Visualization saved plot: {out_path}")

def plot_rouge_bars(df: pd.DataFrame, out_path: str):
    """Creates bar plots comparing full context ROUGE-L to truncated ROUGE-L."""

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data = df,
        x = "dataset",
        y = "rougeL",
        hue = "token_budget",      ## full = 1.0, truncated < 1.0
        palette = "viridis"
    )

    plt.title("ROUGE-L Comparison: Full vs Truncated")
    plt.xlabel("Dataset")
    plt.ylabel("ROUGE-L")
    plt.xticks(rotation=45)
    plt.grid(True, axis="y", alpha=0.3)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plot: {out_path}")

def plot_token_distribution(df: pd.DataFrame, out_path: str):
    """
    Plots histograms of tokens_before and tokens_after across datasets.

    Helps visualize how truncation reduces input length.
    """
    plt.figure(figsize=(12, 6))
    sns.histplot(
        df, x="avg_tokens_before",
        bins=40, kde=True, label="Before", color="blue", alpha=0.5
    )
    sns.histplot(
        df, x="avg_tokens_after",
        bins=40, kde=True, label="After", color="orange", alpha=0.5
    )

    plt.title("Token Count Distribution (Before vs After Truncation)")
    plt.xlabel("Token Count")
    plt.ylabel("Frequency")
    plt.legend()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Visualization saved plot: {out_path}")

def plot_bootstrap_ci(dataset_name: str):
    """
    Computes bootstrap confidence intervals for ROUGE-1 F1
    comparing full-context vs truncated summaries,
    and plots the CI with mean differences.
    """

    # Locate summary json files
    base = f"data/processed/summaries/{dataset_name}"
    full_file = f"{base}_pairs_full_summaries.json"

    # Find actual truncated filename (because budget may vary)
    files = glob.glob(f"{base}_token_budget_*_truncated_summaries.json")
    if len(files) == 0:
        raise FileNotFoundError("No truncated summaries found for CI bootstrap.")
    trunc_file = files[0]

    # Load summary records
    full = json.load(open(full_file, "r"))
    trunc = json.load(open(trunc_file, "r"))

    # Compute ROUGE per document
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    full_scores = []
    trunc_scores = []

    full = sorted(full, key=lambda x: x["id"])
    trunc = sorted(trunc, key=lambda x: x["id"])


    for f, t in zip(full, trunc):
        if f["id"] != t["id"]:
            raise ValueError("ID mismatch between full and truncated summaries")
        
        r_full = scorer.score(f["references"], f["generated_summary"])["rouge1"].fmeasure
        r_trunc = scorer.score(t["references"], t["generated_summary"])["rouge1"].fmeasure

        full_scores.append(r_full)
        trunc_scores.append(r_trunc)

    # Compute bootstrap significance
    stats = significance_test_bootstrap(full_scores, trunc_scores)

    # Plot CI
    plt.figure(figsize=(6, 5))
    ci_low = stats["ci_low"]
    ci_high = stats["ci_high"]
    mean_diff = stats["mean_diff"]

    plt.errorbar(
        x=[0],
        y=[mean_diff],
        yerr=[[mean_diff - ci_low], [ci_high - mean_diff]],
        fmt="o",
        capsize=5
    )

    plt.axhline(0, linestyle="--", color="gray")
    plt.title(f"ROUGE-1 Bootstrap CI: Full − Truncated ({dataset_name})")
    plt.ylabel("Mean Difference in ROUGE-1 F1")
    plt.xticks([])

    out = f"results/{dataset_name}_bootstrap_ci.png"
    os.makedirs("results", exist_ok=True)
    plt.savefig(out, dpi=300)
    plt.close()

    print(f"Visualization Saved bootstrap CI plot -> {out}")

def plot_rouge_drop(df: pd.DataFrame, out_path: str):
    """Plots the drop in ROUGE scores from full to truncated summaries."""

    pivot = df.pivot_table(
        index="dataset",
        columns="token_budget",
        values="rougeL",
        aggfunc="mean"
    )

    num_cols = [c for c in pivot.columns if isinstance(c, (int, float))]
    if len(num_cols) < 2:
        raise ValueError(
            "Need results for at least two token budgets (including full baseline)."
        )

    ## Determine full baseline automatically as the highest token budget
    full_budget = max(num_cols)
    trunc_budgets = [c for c in num_cols if c != full_budget]

    pivot = pivot.reset_index()

    drop_df = pd.concat([
        pd.DataFrame({
            "dataset": pivot["dataset"],
            "token_budget": tb,
            "rouge_drop": pivot[full_budget] - pivot[tb]
        })
        for tb in trunc_budgets
    ])

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=drop_df,
        x="dataset",
        y="rouge_drop",
        hue="token_budget",
        palette="coolwarm"
    )

    plt.title("ROUGE-L Drop After Truncation (Full − Truncated)")
    plt.xlabel("Dataset")
    plt.ylabel("ROUGE-L Drop")
    plt.xticks(rotation=45)
    plt.grid(True, axis="y", alpha=0.3)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Visualization Saved ROUGE drop plot -> {out_path}")

def plot_salience_heatmap(dataset_name: str):
    """
    Visualizes average salience score as a function of chunk index.
    Shows whether salience picks early/middle/late chunks.
    """

    pairs_file = f"data/processed/{dataset_name}_pairs.json"
    sal_file = f"data/processed/salience_scores/{dataset_name}_salience_scores.json"

    if not os.path.exists(pairs_file) or not os.path.exists(sal_file):
        print(f"[heatmap] Missing files for {dataset_name}. Skipping.")
        return

    pairs = json.load(open(pairs_file))
    sal = json.load(open(sal_file))

    sal_map = {item["id"]: item["salience_score"] for item in sal}

    ## build mapping: chunk_index → list of salience
    chunk_positions = {}

    for item in pairs:
        cid = item["id"]
        if "_" not in cid:
            continue  
        idx = int(cid.split("_")[-1])
        score = sal_map.get(cid, 0.0)
        chunk_positions.setdefault(idx, []).append(score)

    ## convert to dataframe for heatmap
    heat = pd.DataFrame([
        [pos, np.mean(scores)] for pos, scores in chunk_positions.items()
    ], columns=["chunk_position", "avg_salience"])

    heat = heat.pivot_table(values="avg_salience", index="chunk_position")

    plt.figure(figsize=(10, 6))
    sns.heatmap(heat, cmap="viridis")

    plt.title(f"Salience vs Chunk Position — {dataset_name}")
    plt.xlabel("Chunk Position")
    plt.ylabel("Average Salience")

    os.makedirs("results/plots", exist_ok=True)
    plt.savefig(f"results/plots/{dataset_name}_salience_heatmap.png", dpi=300)
    plt.close()

def plot_model_selection_histogram():
    """
    Reads logs from summarizer outputs and checks which model was selected 
    for truncated summarization.
    """

    summary_dir = "data/processed/summaries"
    if not os.path.exists(summary_dir):
        print("[model_hist] No summaries folder found.")
        return

    model_counts = {}

    for file in os.listdir(summary_dir):
        if "truncated" in file and file.endswith(".json"):
            path = os.path.join(summary_dir, file)
            data = json.load(open(path))

            ## summarizer.py saves model name at file-level metadata
            ## If not, fallback to checking first record
            model_name = data[0].get("model_name", None)

            if model_name is None:
                continue

            model_counts[model_name] = model_counts.get(model_name, 0) + 1

    if not model_counts:
        print("[model_hist] No model_name found in summaries.")
        return

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=list(model_counts.keys()),
        y=list(model_counts.values()),
        palette="magma"
    )

    plt.title("Model Auto-Selection Frequency")
    plt.xticks(rotation=45)
    plt.ylabel("Number of Datasets/Files Using This Model")

    os.makedirs("results/plots", exist_ok=True)
    plt.savefig("results/plots/model_selection_histogram.png", dpi=300)
    plt.close()
