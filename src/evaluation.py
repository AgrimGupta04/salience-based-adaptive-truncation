import os 
import json 
import traceback
from typing import List, Dict
import torch
import numpy as np
import csv
from rouge_score import rouge_scorer
from bert_score import score as bert_score 
from src.cost_model import estimate_cost

SUMMARIES_DIR = "data/processed/summaries/"
PAIRS_DIR = "data/processed/"
TRUNCATED_DIR = os.path.join(PAIRS_DIR, "truncated_texts/")

def load_summary_file(path: str) -> Dict[str, any]:
    """Loads summary JSON file produced by summarizer.py."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    references = [d.get("references", "") for d in data]
    predictions = [d.get("generated_summary", "") for d in data]
    ids = [d.get("id") for d in data]
    tokens_before = [d.get("tokens_before") for d in data]
    tokens_after = [d.get("tokens_after") for d in data]

    return {
        "ids": ids,
        "records": data,
        "references": references,
        "predictions": predictions,
        "tokens_before": tokens_before,
        "tokens_after": tokens_after
    }

def compute_rouge(references: List[str], predictions: List[str]):
    """Computes ROUGE-1, ROUGE-2, ROUGE-L F1 scores."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1_scores, r2_scores, rl_scores = [], [], []

    for ref, pred in zip(references, predictions):
        score = scorer.score(ref, pred)
        r1_scores.append(score["rouge1"].fmeasure)
        r2_scores.append(score["rouge2"].fmeasure)
        rl_scores.append(score["rougeL"].fmeasure)

    return {
        "rouge1": float(np.mean(r1_scores)),
        "rouge2": float(np.mean(r2_scores)),
        "rougeL": float(np.mean(rl_scores)),
        "rouge1_list": r1_scores,
        "rouge2_list": r2_scores,
        "rougeL_list": rl_scores
    }

def load_full_context_scores(full_summary_path: str) -> Dict[str, List[float]]:
    """
    Loads full-context summaries and computes per-document ROUGE lists
    for significance testing.
    """
    loaded = load_summary_file(full_summary_path)
    return compute_rouge(loaded["references"], loaded["predictions"])

def compute_bertscore(references: List[str], predictions: List[str], device = None):
    """Rouge is lexical -> counts word overlaps.
    BERTScore is semantic -> looks for meaning similarity using pre-trained language models.
    
    1. Load BertScore's scorer.
    2. Compute F1 per pair.
    3. Average the scores.
    
    Returns: Average BERTScore F1.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    P, R, F1 = bert_score(predictions, references, lang = "en", device="cuda" if torch.cuda.is_available() else "cpu")

    return {
        "bert_score_f1": float(F1.mean()),
    }

def compute_token_stats(records: List[dict]) -> Dict[str, float]:
    """Our objective is mainly about token reduction VS summarization quality.
    So we must compuute

    1. Avg tokens before
    2. Avg tokens after
    3. Percent reduction

    Returns: Dict with token statistics.
    """

    before = [r.get("tokens_before", None) for r in records]
    after = [r.get("tokens_after", None) for r in records]

    before = [b for b in before if b is not None]
    after = [a for a in after if a is not None]

    if len(before) == 0 or len(after) == 0:
        return {"avg_tokens_before": None, "avg_tokens_after": None, "percentage_reduction": None}
    
    return {
        "avg_tokens_before": float(np.mean(before)),
        "avg_tokens_after": float(np.mean(after)),
        "percentage_reduction": 100 * (1 - np.mean(after) / np.mean(before))
    }

def compute_cost_stats(records: List[dict], is_baseline: bool = False) -> Dict[str, float]:
    costs = []
    for r in records:
        ## Determine correct token count
        if is_baseline:
            tokens_in = r.get("tokens_before")  ## full input for baseline
        else:
            tokens_in = r.get("tokens_after")   ## truncated input for truncated
        
        if tokens_in is None:
            continue

        model_name = r.get("model_name", "proxy")
        cost = estimate_cost(tokens_in=tokens_in, tokens_out=0, model_name=model_name)
        if cost is not None:
            costs.append(cost)

    if not costs:
        return {"avg_cost_usd": None, "total_cost_usd": None}
    return {
        "avg_cost_usd": float(np.mean(costs)),
        "total_cost_usd": float(np.sum(costs))
    }

def evaluate_summary_file(summary_json_path: str, use_bertscore: bool = True, full_context_scores: Dict[str, List[float]] = None) -> Dict[str, float]:
    try:
        dataset, salience_type, token_budget = parse_metadata_from_filename(summary_json_path)
        loaded = load_summary_file(summary_json_path)

        filtered_refs, filtered_preds = [], []
        for r, p in zip(loaded["references"], loaded["predictions"]):
            if isinstance(r, str) and isinstance(p, str) and len(r.strip()) > 0 and len(p.strip()) > 0:
                filtered_refs.append(r)
                filtered_preds.append(p)

        if len(filtered_refs) == 0:
            raise ValueError(f"No valid pairs found in {summary_json_path}")

        rouge_scores = compute_rouge(filtered_refs, filtered_preds)

        bert_scores = compute_bertscore(
            filtered_refs, filtered_preds
        ) if use_bertscore else {}
            
        token_stats = compute_token_stats(loaded["records"])

        is_baseline = (salience_type == "None" or "full_pairs" in summary_json_path)
        cost_stats = compute_cost_stats(loaded["records"], is_baseline=is_baseline)

        ## significance testing with length alignment
        sig_results = {}
        if full_context_scores is not None and dataset in ["cnn_dailymail", "govreport", "arxiv"]: 
            min_len = min(len(full_context_scores["rouge1_list"]), len(rouge_scores["rouge1_list"]))
            
            if min_len > 0:
                for metric in ["rouge1", "rougeL"]:
                    aligned_full = full_context_scores[f"{metric}_list"][:min_len]
                    aligned_trunc = rouge_scores[f"{metric}_list"][:min_len]

                    sig = significance_test_bootstrap(aligned_full, aligned_trunc)

                    sig_results.update({
                        f"{metric}_{k}": v for k, v in sig.items()
                    })
            else:
                print(f"Warning: Score list is empty for {summary_json_path}, skipping significance.")

        return {
            "file": summary_json_path,
            "dataset": dataset,
            "salience_type": salience_type,       
            "token_budget": token_budget,          
            **{k: rouge_scores[k] for k in ["rouge1", "rouge2", "rougeL"]},
            **bert_scores,
            **token_stats,
            **cost_stats,
            **sig_results,         
        }
    except Exception as e:
        print(f"\n[FATAL ERROR] Crashed while evaluating file: {summary_json_path}")
        traceback.print_exc()
        raise e

def significance_test_bootstrap(full_scores, trunc_scores, n_samples=1000, seed=42):
    """
    Bootstrap significance test between full-context and truncated ROUGE scores.

    Args:
        full_scores: list of per-document ROUGE scores for full summaries
        trunc_scores: list of per-document ROUGE scores for truncated summaries
        n_samples: number of bootstrap resamples

    Returns:
        dict with:
            mean_diff: average difference full - truncated
            ci_low: lower CI 2.5%
            ci_high: upper CI 97.5%
            p_value: two-sided bootstrap p-value
    """
    assert len(full_scores) == len(trunc_scores), "Scores must align one-to-one"
    rng = np.random.default_rng(seed)
    full_scores, trunc_scores = np.array(full_scores), np.array(trunc_scores)
    n = len(full_scores)

    diffs = []
    for _ in range(n_samples):
        idx = rng.integers(0, n, n) 
        diffs.append(full_scores[idx].mean() - trunc_scores[idx].mean())

    diffs = np.array(diffs)
    return {
        "mean_diff": float(diffs.mean()),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "p_value": float(min((diffs > 0).mean(), (diffs < 0).mean()) * 2)
    }

def parse_metadata_from_filename(filename: str):
    base = os.path.basename(filename).replace(".json", "")
    known_datasets = ["cnn_dailymail", "govreport", "arxiv"]
    dataset = next((d for d in known_datasets if base.startswith(d)), "unknown")
    
    remainder = base[len(dataset):].lstrip("_")
    parts = remainder.split("_")
    
    truncation_type = "None"
    token_budget = None

    if "salience" in parts:
        try:
            truncation_type = parts[parts.index("salience") + 1]
        except IndexError:
            pass
    elif "first" in parts:
        truncation_type = "first_k"
    elif "random" in parts:
        truncation_type = "random_k"
    elif "lead" in parts:
        truncation_type = "lead_n"


    if "budget" in parts:
        try:
            token_budget = int(parts[parts.index("budget") + 1])
        except Exception:
            pass

    return dataset, truncation_type, token_budget

def save_evaluation_results(results: dict, out_path: str = "results/metrics.csv"):
    os.makedirs("results", exist_ok=True)
    FIELD_ORDER = [
        "file",
        "dataset", 
        "salience_type", 
        "token_budget",
        "bert_score_f1",
        "rouge1", "rouge2", "rougeL",
        "avg_tokens_before", "avg_tokens_after", "percentage_reduction",
        "avg_cost_usd", "total_cost_usd",
        "rouge1_mean_diff", "rouge1_ci_low", "rouge1_ci_high", "rouge1_p_value",
        "rougeL_mean_diff", "rougeL_ci_low", "rougeL_ci_high", "rougeL_p_value"
    ]

    ## Ensure only valid keys included
    row = {k: results.get(k) for k in FIELD_ORDER}

    file_exists = os.path.exists(out_path)

    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELD_ORDER)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"Saved metrics to {out_path}")