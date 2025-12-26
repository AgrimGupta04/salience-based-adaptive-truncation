import os 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
import json
from rouge_score import rouge_scorer
from src.evaluation import significance_test_bootstrap
import glob

INPUT_COST_PER_1K = 0.01    # e.g., $0.01 / 1K input tokens
OUTPUT_COST_PER_1K = 0.03   # unused for now (output length ~ fixed)
ASSUMED_OUTPUT_TOKENS = 128

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
    df["budget_label"] = df["token_budget"].fillna("full").astype(str)

    df["compression_ratio"] = df["avg_tokens_after"] / df["avg_tokens_before"]
    df["token_budget"] = pd.to_numeric(df["token_budget"], errors="coerce")

    return df


def plot_tradeoff_curves(df: pd.DataFrame, out_path: str):
    """
    Plots ROUGE-L vs compression ratio.
    This is the PRIMARY trade-off curve for the paper.
    """

    plt.figure(figsize=(10, 6))

    for dataset in df["dataset"].unique():
        sub = df[df["dataset"] == dataset].sort_values("compression_ratio")
        plt.plot(
            sub["compression_ratio"],
            sub["rougeL"],
            marker="o",
            label=dataset
        )

    plt.xlabel("Compression Ratio (tokens_after / tokens_before)")
    plt.ylabel("ROUGE-L")
    plt.title("Quality–Compression Trade-off Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved trade-off curve → {out_path}")


def plot_quality_vs_budget(df: pd.DataFrame, out_path: str):
    """
    Plots ROUGE-L vs absolute token budget.
    Useful secondary trade-off view.
    """

    plt.figure(figsize=(10, 6))

    trunc = df[~df["is_full"]]

    for dataset in trunc["dataset"].unique():
        sub = trunc[trunc["dataset"] == dataset].sort_values("token_budget")
        plt.plot(
            sub["token_budget"],
            sub["rougeL"],
            marker="o",
            label=dataset
        )

    plt.xlabel("Token Budget")
    plt.ylabel("ROUGE-L")
    plt.title("Quality vs Token Budget")
    plt.grid(True, alpha=0.3)
    plt.legend()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved quality-vs-budget curve → {out_path}")


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
        style="budget_label",
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
        hue = "budget_label",     
    )

    plt.title("ROUGE-L Comparison: Full Context vs Truncated")
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

def estimate_input_cost(tokens: float, price_per_1k: float = INPUT_COST_PER_1K) -> float:
    """
    Estimates input-side API cost given token count.
    """
    if tokens is None:
        return None
    return (tokens / 1000.0) * price_per_1k

def add_cost_columns(df):
    """
    Adds estimated cost columns based on token statistics.
    """
    df = df.copy()

    df["estimated_cost"] = df["avg_tokens_after"].apply(
        lambda x: estimate_input_cost(x)
    )

    df["estimated_cost_full"] = df["avg_tokens_before"].apply(
        lambda x: estimate_input_cost(x)
    )

    df["cost_reduction_percent"] = 100 * (
        1 - df["estimated_cost"] / df["estimated_cost_full"]
    )

    return df


def plot_cost_vs_budget(df, out_path="results/plots/cost_vs_budget.png"):
    """
    Plots estimated input cost vs token budget.
    """
    df = add_cost_columns(df)

    plt.figure(figsize=(7, 5))
    for dataset in df["dataset"].unique():
        sub = df[df["dataset"] == dataset]
        plt.plot(
            sub["token_budget"],
            sub["estimated_cost"],
            marker="o",
            label=f"{dataset} (truncated)"
        )

    plt.xlabel("Token Budget")
    plt.ylabel("Estimated Input Cost ($)")
    plt.title("Inference Cost vs Token Budget")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    print(f"[plot] saved {out_path}")

def plot_cost_vs_quality(df, metric="rouge1", out_path="results/plots/cost_vs_quality.png"):
    """
    Scatter plot of estimated cost vs summarization quality.
    """
    df = add_cost_columns(df)

    plt.figure(figsize=(7, 5))
    for dataset in df["dataset"].unique():
        sub = df[df["dataset"] == dataset]
        plt.scatter(
            sub["estimated_cost"],
            sub[metric],
            label=dataset,
            alpha=0.8
        )

    plt.xlabel("Estimated Input Cost ($)")
    plt.ylabel(metric.upper())
    plt.title(f"Cost vs Quality ({metric.upper()})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    print(f"[plot] saved {out_path}")


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
    trunc_files = glob.glob(f"{base}_*_token_budget_*_truncated_summaries.json")
    if len(trunc_files) == 0:
        raise FileNotFoundError("No truncated summaries found for CI bootstrap.")

    # Load summary records
    full = sorted(json.load(open(full_file)), key=lambda x: x["id"])
    trunc = sorted(json.load(open(trunc_files[0])), key=lambda x: x["id"])

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

    if df["is_full"].sum() == 0:
        raise ValueError("Full-context baseline missing")

    # Ensure the full index is clean (e.g., 'cnn_dailymail')
    full = df[df["is_full"]].copy()
    full["dataset"] = full["dataset"].str.replace(r'_(tfidf|cosine|hybrid|salience)$', '', regex=True)
    full = full.set_index("dataset")["rougeL"]

    trunc = df[~df["is_full"]].copy()
    drop_records = []

    for _, row in trunc.iterrows():
        # Clean the dataset name for lookup (remove the suffix)
        clean_name = pd.Series([row["dataset"]]).str.replace(r'_(tfidf|cosine|hybrid|salience)$', '', regex=True).iloc[0]
        
        if clean_name in full.index:
            drop_records.append({
                "dataset": row["dataset"], # Keep original for the plot label
                "token_budget": row["token_budget"],
                "rouge_drop": full[clean_name] - row["rougeL"]
            })

    if not drop_records:
        print("[Warning] No matching datasets found between full and truncated for ROUGE drop plot.")
        return

    drop_df = pd.DataFrame(drop_records)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=drop_df,
        x="dataset",
        y="rouge_drop",
        hue="token_budget"
    )

    plt.title("ROUGE-L Drop After Truncation")
    plt.ylabel("ROUGE-L Drop (Full − Truncated)")
    plt.xticks(rotation=45)
    plt.grid(axis="y", alpha=0.3)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_salience_heatmap(dataset_name: str, salience_type: str):
    """
    Visualizes average salience score as a function of chunk index.
    Shows whether salience picks early/middle/late chunks.
    """

    pairs_file = f"data/processed/{dataset_name}_pairs.json"
    sal_file = f"data/processed/salience_scores/{dataset_name}_{salience_type}_salience.json"

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
        {"chunk_position": k, "avg_salience": np.mean(v)}
        for k, v in chunk_positions.items()
    ]).set_index("chunk_position")


    plt.figure(figsize=(10, 6))
    sns.heatmap(heat, cmap="viridis")

    plt.title(f"Salience vs Chunk Position — {dataset_name}")
    plt.xlabel("Chunk Position")
    plt.ylabel("Average Salience")

    os.makedirs("results/plots", exist_ok=True)
    plt.savefig(f"results/plots/{dataset_name}_{salience_type}_salience_heatmap.png", dpi=300)
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
