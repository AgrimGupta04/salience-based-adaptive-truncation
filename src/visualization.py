import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import re
import glob
from rouge_score import rouge_scorer

INPUT_COST_PER_1K = 0.01    # e.g., $0.01 / 1K input tokens

def load_metrics_csv(csv_path: str) -> pd.DataFrame:
    """
    Loads evaluation metrics CSV and cleans dataset names/methods.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metrics CSV not found at {csv_path}. Please run evaluation.py first.")

    df = pd.read_csv(csv_path)

    numeric_cols = [
        "rouge1", "rouge2", "rougeL",
        "bert_score_f1", "avg_tokens_before", 
        "avg_tokens_after", "percentage_reduction", "token_budget"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    ## Parse Dataset and Method from 'dataset' column or filename
    def parse_dataset_info(row):
        name = str(row.get('dataset', ''))
        
        if 'pairs_full_summaries' in name:
            clean_name = name.replace('_pairs_full_summaries.json', '')
            return clean_name, 'Full Context'

        match = re.search(r'^(.*)_(tfidf|cosine|hybrid|salience)(?:_|$)', name)
        if match:
            return match.group(1), match.group(2)
        
        ## Fallback
        if 'arxiv' in name: return 'arxiv', 'unknown'
        return name, 'unknown'

    df[['dataset_clean', 'method']] = df.apply(
        lambda x: pd.Series(parse_dataset_info(x)), axis=1
    )

    df["is_full"] = df["method"] == "Full Context"
    df["budget_label"] = df["token_budget"].fillna("Full").astype(str)
    df["compression_ratio"] = df["avg_tokens_after"] / df["avg_tokens_before"]

    df = add_cost_columns(df)
    
    return df

def add_cost_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Helper to add estimated cost columns."""
    df = df.copy()
    df["estimated_cost"] = (df["avg_tokens_after"] / 1000.0) * INPUT_COST_PER_1K
    df["estimated_cost_full"] = (df["avg_tokens_before"] / 1000.0) * INPUT_COST_PER_1K
    return df

# ========================================================
# PLOTTING FUNCTIONS
# ========================================================

def plot_tradeoff_curves_aggregated(df: pd.DataFrame, out_path: str):
    """
    ROUGE-L vs Compression Ratio (aggregated across salience methods).
    """
    agg = aggregate_by_budget(df)
    if agg.empty:
        return

    plt.figure(figsize=(8, 6))
    sns.lineplot(
        data=agg,
        x="compression_ratio",
        y="rougeL",
        hue="dataset_clean",
        marker="o"
    )

    plt.xlabel("Compression Ratio (Tokens After / Before)")
    plt.ylabel("ROUGE-L")
    plt.title("Quality–Compression Trade-off")
    plt.grid(alpha=0.3)
    plt.legend(title="Dataset")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_cost_vs_quality_aggregated(df: pd.DataFrame, out_path: str, metric="rougeL"):
    """
    Cost vs Quality frontier (aggregated).
    """
    agg = aggregate_by_budget(df)

    plt.figure(figsize=(8, 6))

    ## Full context points
    full = df[df["is_full"]]
    if not full.empty:
        plt.scatter(
            full["estimated_cost_full"],
            full[metric],
            marker="*",
            s=200,
            color="black",
            label="Full Context"
        )

    sns.scatterplot(
        data=agg,
        x="estimated_cost",
        y=metric,
        hue="dataset_clean",
        s=100
    )

    plt.xlabel("Estimated Input Cost ($)")
    plt.ylabel(metric)
    plt.title("Cost–Quality Trade-off")
    plt.grid(alpha=0.3)
    plt.legend()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_cost_vs_budget_aggregated(df: pd.DataFrame, out_path: str):
    agg = aggregate_by_budget(df)

    plt.figure(figsize=(8, 6))
    sns.lineplot(
        data=agg,
        x="token_budget",
        y="estimated_cost",
        hue="dataset_clean",
        marker="o"
    )

    plt.xlabel("Token Budget")
    plt.ylabel("Estimated Input Cost ($)")
    plt.title("Inference Cost vs Token Budget")
    plt.grid(alpha=0.3)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_rouge_drop(df: pd.DataFrame, out_path: str):
    """Plots ROUGE Drop (Full - Truncated)."""
    full_avgs = df[df["is_full"]].groupby("dataset_clean")["rougeL"].mean()
    
    drop_records = []
    for _, row in df[~df["is_full"]].iterrows():
        ds = row["dataset_clean"]
        if ds in full_avgs.index:
            drop = full_avgs[ds] - row["rougeL"]
            drop_records.append({
                "dataset": ds, "method": row["method"],
                "token_budget": row["token_budget"], "rouge_drop": drop
            })
            
    if not drop_records: return

    plt.figure(figsize=(10, 6))
    sns.barplot(data=pd.DataFrame(drop_records), x="dataset", y="rouge_drop", hue="method")
    plt.title("Performance Drop (Full Context - Truncated)")
    plt.ylabel("Delta ROUGE-L (Lower is Better)")
    plt.grid(axis="y", alpha=0.3)
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved {out_path}")

def plot_rouge_bars_aggregated(df: pd.DataFrame, out_path: str):
    agg = aggregate_by_budget(df)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=agg,
        x="dataset_clean",
        y="rougeL",
        hue="token_budget"
    )

    plt.title("ROUGE-L Across Token Budgets")
    plt.ylabel("ROUGE-L")
    plt.xlabel("Dataset")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_token_distribution(df: pd.DataFrame, out_path: str):
    """Visualizes distribution of tokens Before vs After truncation."""
    plt.figure(figsize=(12, 6))
    trunc = df[~df["is_full"]].copy()
    
    if trunc.empty: return

    sns.histplot(trunc["avg_tokens_before"], color="blue", alpha=0.4, label="Original Length", kde=True)
    sns.histplot(trunc["avg_tokens_after"], color="orange", alpha=0.6, label="Truncated Length", kde=True)

    plt.title("Token Count Distribution (Before vs After)")
    plt.xlabel("Number of Tokens")
    plt.ylabel("Frequency")
    plt.legend()
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved {out_path}")

def plot_model_selection_histogram(out_path="results/plots/model_selection.png"):
    """Scans processed summaries to see which models were auto-selected."""
    summary_dir = "data/processed/summaries"
    if not os.path.exists(summary_dir): return

    model_counts = {}
    files = glob.glob(os.path.join(summary_dir, "*truncated_summaries.json"))
    
    for path in files:
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    model = data[0].get("model_name", "Unknown")
                    model_counts[model] = model_counts.get(model, 0) + 1
        except Exception:
            continue

    if not model_counts: return

    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(model_counts.keys()), y=list(model_counts.values()), palette="viridis")
    plt.title("Model Auto-Selection Frequency")
    plt.ylabel("Count")
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved {out_path}")

def aggregate_by_budget(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates truncated runs by dataset and token budget.
    Salience methods are averaged out.
    """
    trunc = df[~df["is_full"]].copy()

    grouped = trunc.groupby(
        ["dataset_clean", "token_budget"],
        as_index=False
    ).agg({
        "rougeL": "mean",
        "rouge1": "mean",
        "bert_score_f1": "mean",
        "compression_ratio": "mean",
        "estimated_cost": "mean",
        "avg_tokens_after": "mean"
    })

    return grouped


def plot_bootstrap_ci(dataset_name: str, n_rounds: int = 10000):
    """
    Computes and plots 95% Bootstrap Confidence Intervals.
    HANDLES ID MISMATCH (e.g. cnn_dailymail_0 vs cnn_dailymail_0_0).
    """
    if 'arxiv' in dataset_name.lower():
        print(f"[CI Plot] Skipping {dataset_name} (No Full Context baseline).")
        return

    base_dir = "data/processed/summaries"
    full_pattern = os.path.join(base_dir, f"*{dataset_name}*full_summaries.json")
    trunc_pattern = os.path.join(base_dir, f"*{dataset_name}*truncated_summaries.json")
    
    full_files = glob.glob(full_pattern)
    trunc_files = glob.glob(trunc_pattern)
    
    if not full_files or not trunc_files:
        print(f"[CI Plot] Missing summary files for {dataset_name}")
        return

    full_path = full_files[0]
    trunc_path = trunc_files[0]
    print(f"[CI Plot] Processing {dataset_name}...")

    try:
        with open(trunc_path, 'r') as f:
            trunc_data = {item['id']: item for item in json.load(f)}
    except Exception as e:
        print(f"[CI Plot] Error loading trunc JSON: {e}")
        return

    full_data = {}
    try:
        with open(full_path, 'r') as f:
            raw_full = json.load(f)
            
        for item in raw_full:
            fid = item['id']
            if fid in trunc_data:
                full_data[fid] = item
                continue
            if "_" in fid:
                norm_id = fid.rsplit('_', 1)[0]
                if norm_id in trunc_data:
                    full_data[norm_id] = item
    except Exception as e:
        print(f"[CI Plot] Error loading full JSON: {e}")
        return

    common_ids = set(full_data.keys()) & set(trunc_data.keys())
    if not common_ids:
        print(f"[CI Plot] No matching IDs found for {dataset_name} even after normalization.")
        return
    else:
        print(f"[CI Plot] Found {len(common_ids)} matching documents for Bootstrap CI.")

    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    differences = []

    for doc_id in common_ids:
        f_rec = full_data[doc_id]
        t_rec = trunc_data[doc_id]
        
        ref = f_rec['references'][0] if isinstance(f_rec['references'], list) else f_rec['references']
        
        s_full = scorer.score(ref, f_rec['generated_summary'])['rouge1'].fmeasure
        s_trunc = scorer.score(ref, t_rec['generated_summary'])['rouge1'].fmeasure
        
        differences.append(s_full - s_trunc)

    differences = np.array(differences)
    if len(differences) < 2: return

    means = []
    for _ in range(n_rounds):
        sample = np.random.choice(differences, size=len(differences), replace=True)
        means.append(np.mean(sample))
    
    ci_low = np.percentile(means, 2.5)
    ci_high = np.percentile(means, 97.5)
    mean_diff = np.mean(differences)


    plt.figure(figsize=(6, 5))
    plt.errorbar(x=[0], y=[mean_diff], yerr=[[mean_diff - ci_low], [ci_high - mean_diff]], 
                 fmt='o', color='black', capsize=10, label=f'Mean Drop: {mean_diff:.4f}')
    
    plt.axhline(0, linestyle='--', color='grey', alpha=0.5)
    plt.ylabel("ROUGE-1 Drop (Full - Truncated)")
    plt.title(f"95% CI of Quality Drop\n({dataset_name})")
    plt.xticks([])
    plt.legend()
    plt.grid(axis='y', alpha=0.2)

    out_path = f"results/plots/{dataset_name}_bootstrap_ci.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[CI Plot] Saved {out_path}")