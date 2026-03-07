import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import re
import glob
from rouge_score import rouge_scorer

INPUT_COST_PER_1K = 0.01    ## e.g., $0.01 / 1K input tokens

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
    def get_method(filename):
        name = str(filename).lower()
        if 'full_pairs_full_summaries' in name: return 'Full Context'
        if 'tfidf' in name: return 'tfidf'
        if 'cosine' in name: return 'cosine'
        if 'hybrid' in name: return 'hybrid'
        if 'first_k' in name: return 'first_k'
        if 'random_k' in name: return 'random_k'
        if 'lead_n' in name: return 'lead_n'
        return 'unknown'

    df['dataset_clean'] = df['dataset'] 
    df['method'] = df['file'].apply(get_method)

    df["is_baseline"] = df["method"].isin(["first_k", "random_k", "lead_n"])
    df["is_salience"] = df["method"].isin(["tfidf", "cosine", "hybrid", "salience"])
    df["is_full"] = df["method"] == "Full Context"
    df["budget_label"] = df["token_budget"].fillna("Full").astype(str)

    ## Since the full context runs don't have a token budget, 
    ## we set their "after" tokens to be the same as "before" to reflect no truncation.
    mask_full = df["is_full"]       
    df.loc[mask_full, "avg_tokens_after"] = df.loc[mask_full, "avg_tokens_before"]
    df.loc[mask_full, "percentage_reduction"] = 0.0

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

    plt.figure(figsize = (8, 6))
    sns.lineplot(
        data = agg,
        x = "compression_ratio",
        y = "rougeL",
        hue = "dataset_clean",
        marker = "o"
    )

    plt.xlabel("Compression Ratio (Tokens After / Before)")
    plt.ylabel("ROUGE-L")
    plt.title("Quality-Compression Trade-off")
    plt.grid(alpha=0.3)
    plt.legend(title="Dataset")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print("")
    plt.close()


def plot_cost_vs_quality_aggregated(df: pd.DataFrame, out_path: str, metric="rougeL"):
    """
    Cost vs Quality frontier (aggregated).
    """
    agg = aggregate_by_budget(df)

    plt.figure(figsize = (8, 6))

    ## Full context points
    full = df[df["is_full"]].groupby("dataset_clean", as_index = False).mean(numeric_only = True)
    if not full.empty:
        plt.scatter(
            full["estimated_cost_full"],
            full[metric],
            marker = "*",
            s= 200,
            color = "black",
            label = "Full Context",
            zorder = 5
        )

    if not agg.empty:
        sns.scatterplot(
            data = agg,
            x = "estimated_cost",
            y = metric,
            hue = "method",
            style = "dataset_clean",
            s = 100,
            zorder = 4
        )

    plt.xlabel("Estimated Input Cost ($)")
    plt.ylabel(metric)
    plt.title("Cost Vs Quality Trade-off by Truncation Method")
    plt.grid(alpha=0.3)
    plt.legend(bbox_to_anchor = (1.05, 1), loc = "upper left")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi = 300, bbox_inches = "tight")
    plt.close()


def plot_cost_vs_budget_aggregated(df: pd.DataFrame, out_path: str):
    agg = aggregate_by_budget(df)
    if agg.empty: return

    agg["budget_cost"] = (agg["token_budget"] / 1000.0) * INPUT_COST_PER_1K

    melted = agg.melt(
        id_vars = ["dataset_clean", "method"],
        value_vars = ["budget_cost", "estimated_cost"],
        var_name = "Cost_Type",
        value_name = "Cost_USD"
    )

    melted["Cost_Type"] = melted["Cost_Type"].map({
        "budget_cost": "Max Budget Cost Limit",
        "estimated_cost": "Actual Inference Cost (Truncated)"
    })

    plt.figure(figsize = (10, 6))
    sns.barplot(
        data = melted,
        x = "dataset_clean",
        y = "Cost_USD",
        hue = "Cost_Type",
        errorbar = None, 
        palette = "muted"
    )

    plt.xlabel("Dataset")
    plt.ylabel("Inference Cost ($)")
    plt.title("Inference Cost vs Token Budget")
    plt.grid(axis = 'y', alpha = 0.3)

    os.makedirs(os.path.dirname(out_path), exist_ok = True)
    plt.savefig(out_path, dpi = 300, bbox_inches = "tight")
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
    bar_df = df.groupby(["dataset_clean", "method"], as_index=False)["rougeL"].mean()

    plt.figure(figsize = (10, 6))
    sns.barplot(
        data = bar_df,
        x = "dataset_clean",
        y = "rougeL",
        hue = "method"
    )

    plt.title("ROUGE-L Performance: Full Context Vs Truncation Methods")
    plt.ylabel("ROUGE-L")
    plt.xlabel("Dataset")
    plt.grid(axis = 'y', alpha = 0.3)

    os.makedirs(os.path.dirname(out_path), exist_ok = True)
    plt.savefig(out_path, dpi = 300, bbox_inches = "tight")
    plt.close()

def plot_token_distribution(df: pd.DataFrame, out_path: str):
    """Visualizes distribution of tokens Before vs After truncation."""
    plt.figure(figsize=(12, 6))
    trunc = df[~df["is_full"]].copy()
    
    if trunc.empty: return

    sns.histplot(trunc["avg_tokens_before"], color = "blue", alpha = 0.4, label="Original Length", kde = True)
    sns.histplot(trunc["avg_tokens_after"], color = "orange", alpha = 0.6, label="Truncated Length", kde = True)

    plt.title("Token Count Distribution (Before vs After)")
    plt.xlabel("Number of Tokens")
    plt.ylabel("Frequency")
    plt.legend()
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi = 300, bbox_inches = "tight")
    plt.close()
    print(f"[plot] Saved {out_path}")

def plot_model_selection_histogram(out_path="results/plots/model_selection.png"):
    """Scans processed summaries to see which models were auto-selected."""
    summary_dir = "data/processed/summaries"
    if not os.path.exists(summary_dir): return

    model_counts = {}
    files = [f for f in glob.glob(os.path.join(summary_dir, "*_summaries.json")) if 'full_pairs' not in f]
    
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
    sns.barplot(x = list(model_counts.keys()), y = list(model_counts.values()), palette = "viridis", legend = False)
    plt.title("Model Auto-Selection Frequency")
    plt.ylabel("Count")
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi = 300, bbox_inches = "tight")
    plt.close()
    print(f"[plot] Saved {out_path}")

def aggregate_by_budget(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates truncated runs by dataset and token budget.
    Salience methods are averaged out.
    """
    trunc = df[(~df["is_full"]) & (df["is_salience"])].copy()

    grouped = trunc.groupby(
        ["dataset_clean", "method", "token_budget"],
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

def plot_bootstrap_ci(dataset_name: str, n_rounds: int = 1000):
    """
    Computes and plots 95% Bootstrap Confidence Intervals for all methods side-by-side.
    """
    base_dir = "data/processed/summaries"
    
    full_pattern = os.path.join(base_dir, f"*{dataset_name}*full_pairs_full_summaries.json")
    full_files = glob.glob(full_pattern)
    
    if not full_files:
        print(f"[CI Plot] Missing full baseline for {dataset_name}. Skipping.")
        return
    full_path = full_files[0]

    ## Target all truncated methods
    trunc_pattern = os.path.join(base_dir, f"*{dataset_name}*_budget_*_summaries.json")
    trunc_files = glob.glob(trunc_pattern)
    
    if not trunc_files:
        print(f"[CI Plot] No truncated files found for {dataset_name}.")
        return

    print(f"[CI Plot] Processing multiple methods for {dataset_name}...")

    ## Load Baseline Data
    full_data = {}
    try:
        with open(full_path, 'r') as f:
            raw_full = json.load(f)
        for item in raw_full:
            full_data[item['id']] = item
    except Exception as e:
        print(f"[CI Plot] Error loading full JSON: {e}")
        return

    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    results = []

    ## Iterate over all truncation methods found in the directory
    for trunc_path in trunc_files:
        filename = os.path.basename(trunc_path)
        
        # Determine Method Name for the chart label
        if 'tfidf' in filename: method_name = 'TF-IDF'
        elif 'cosine' in filename: method_name = 'Cosine'
        elif 'hybrid' in filename: method_name = 'Hybrid'
        elif 'first_k' in filename: method_name = 'First-k'
        elif 'random_k' in filename: method_name = 'Random-k'
        elif 'lead_n' in filename: method_name = 'Lead-n'
        else: continue

        try:
            with open(trunc_path, 'r') as f:
                trunc_data = {item['id']: item for item in json.load(f)}
        except Exception:
            continue

        ## Handle ID Mismatches
        common_ids = set(full_data.keys()) & set(trunc_data.keys())
        if not common_ids:
            normalized_full = {}
            for k, v in full_data.items():
                if "_" in k: normalized_full[k.rsplit('_', 1)[0]] = v
            full_data_to_use = normalized_full
            common_ids = set(full_data_to_use.keys()) & set(trunc_data.keys())
        else:
            full_data_to_use = full_data

        if not common_ids:
            continue

        ## Compute ROUGE differences
        differences = []
        for doc_id in common_ids:
            f_rec = full_data_to_use[doc_id]
            t_rec = trunc_data[doc_id]
            
            ref = f_rec['references'][0] if isinstance(f_rec['references'], list) else f_rec['references']
            
            s_full = scorer.score(ref, f_rec['generated_summary'])['rouge1'].fmeasure
            s_trunc = scorer.score(ref, t_rec['generated_summary'])['rouge1'].fmeasure
            
            differences.append(s_full - s_trunc)

        differences = np.array(differences)
        if len(differences) < 2: continue

        ## Bootstrap
        means = []
        for _ in range(n_rounds):
            sample = np.random.choice(differences, size = len(differences), replace = True)
            means.append(np.mean(sample))
        
        ci_low = np.percentile(means, 2.5)
        ci_high = np.percentile(means, 97.5)
        mean_diff = np.mean(differences)

        results.append({
            'method': method_name,
            'mean_diff': mean_diff,
            'ci_low': ci_low,
            'ci_high': ci_high
        })

    if not results:
        return

    ## Plot the aggregated CI comparison
    plt.figure(figsize=(8, 6))
    methods = [r['method'] for r in results]
    means = [r['mean_diff'] for r in results]
    
    ## Error bar format: [ [distances below mean], [distances above mean] ]
    yerr = [
        [r['mean_diff'] - r['ci_low'] for r in results], 
        [r['ci_high'] - r['mean_diff'] for r in results]
    ]

    plt.errorbar(x = methods, y = means, yerr = yerr, fmt = 'o', color = 'black', capsize = 8, markersize = 8)
    
    ## Adding a red dashed line at 0
    plt.axhline(0, linestyle='--', color='red', alpha = 0.5, label = 'Baseline Quality (No Drop)')
    
    plt.ylabel("ROUGE-1 Drop (Full - Truncated) -> Lower is Better")
    plt.title(f"95% CI of Quality Drop by Compression Method\n({dataset_name})")
    plt.grid(axis = 'y', alpha = 0.2)
    plt.legend()

    out_path = f"results/plots/{dataset_name}_bootstrap_ci_comparison.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi = 300, bbox_inches = "tight")
    plt.close()
    print(f"[CI Plot] Saved {out_path}")

    ## Plot the aggregated CI comparison
    plt.figure(figsize=(8, 6))
    methods = [r['method'] for r in results]
    means = [r['mean_diff'] for r in results]
    
    ## Error bar format: [ [distances below mean], [distances above mean] ]
    yerr = [
        [r['mean_diff'] - r['ci_low'] for r in results], 
        [r['ci_high'] - r['mean_diff'] for r in results]
    ]

    plt.errorbar(x = methods, y = means, yerr = yerr, fmt = 'o', color = 'black', capsize = 8, markersize = 8)
    
    ## Adding a red dashed line at 0 
    plt.axhline(0, linestyle='--', color='red', alpha=0.5, label='Baseline Quality (No Drop)')
    
    plt.ylabel("ROUGE-1 Drop (Full - Truncated) -> Lower is Better")
    plt.title(f"95% CI of Quality Drop by Compression Method\n({dataset_name})")
    plt.grid(axis = 'y', alpha = 0.2)
    plt.legend()

    out_path = f"results/plots/{dataset_name}_bootstrap_ci_comparison.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi = 300, bbox_inches = "tight")
    plt.close()
    print(f"[CI Plot] Saved {out_path}")

def plot_quality_vs_cost(df, out_path, metric="rougeL"):
    trunc = df[~df["is_full"]]

    plt.figure(figsize = (8, 6))
    sns.scatterplot(
        data = trunc,
        x = "estimated_cost",
        y = metric,
        hue = "method",
        style = "dataset_clean",
        s = 100
    )
    plt.xlabel("Average Cost (USD)")
    plt.ylabel(metric)
    plt.title("Quality vs Real Inference Cost")
    plt.grid(alpha=0.3)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_cost_at_quality_threshold(df, out_path="results/plots/cost_threshold.png", threshold=0.95):
    records = []

    for ds in df["dataset_clean"].unique():
        full_score = df[(df["dataset_clean"] == ds) & (df["is_full"])]["rougeL"].mean()
        cutoff = threshold * full_score

        for method in df["method"].unique():
            subset = df[
                (df["dataset_clean"] == ds) &
                (df["method"] == method) &
                (df["rougeL"] >= cutoff)
            ]
            if not subset.empty:
                records.append({
                    "dataset": ds,
                    "method": method,
                    "min_cost_usd": subset["estimated_cost"].min()
                })

    result_df = pd.DataFrame(records)
    if result_df.empty: return result_df
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data = result_df, x = "dataset", y = "min_cost_usd", hue = "method")
    plt.title(f"Minimum Inference Cost to Achieve {threshold*100}% of Baseline Quality")
    plt.ylabel("Minimum Cost (USD)")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi = 300, bbox_inches = "tight")
    plt.close()
    return result_df