import os 
import json 
from typing import List, Dict

import numpy as np
import csv
from rouge_score import rouge_scorer
from bert_score import score as bert_score

SUMMARIES_DIR = "data/processed/summaries/"
PAIRS_DIR = "data/processed/"
TRUNCATED_DIR = os.path.join(PAIRS_DIR, "truncated_texts/")

def load_summary_file(path: str) -> Dict[str, any]:
    """Loads summary JSON file produced by summarizer.py and extracts:
    - references 
    - predictions
    - token metadata
    """

    with open(path, "r", encoding = "utf-8") as f:
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
    """ROUGE is the standard evaluation metric for summarization.
    Computes ROUGE-1, ROUGE-2, ROUGE-L F1 scores.
    
    Returns: Dict with average Rouge values.
    """

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

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


def compute_bertscore(references: List[str], predictions: List[str]):
    """Rouge is lexical -> counts word overlaps.
    BERTScore is semantic -> looks for meaning similarity using pre-trained language models.
    
    1. Load BertScore's scorer.
    2. Compute F1 per pair.
    3. Average the scores.
    
    Returns: Average BERTScore F1.
    """

    P, R, F1 = bert_score(predictions, references, lang = "en")

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
        return {
          "avg_tokens_before": None,
          "avg_tokens_after": None,
          "percentage_reduction": None
        }
    
    return {
        "avg_tokens_before": float(np.mean(before)),
        "avg_tokens_after": float(np.mean(after)),
        "percentage_reduction": 100 * (1 - np.mean(after) / np.mean(before))
    }

def evaluate_summary_file(summary_json_path: str, use_bertscore: bool = True) -> Dict[str, float]:
    """High-level evaluation for a single summary JSON file.
    Computes ROUGE, BERTScore, and token compression stats.
    
    Args:
        summary_json_path: Path to the summary JSON file.

    Returns: Dict with all evaluation metrics.
    """

    base = os.path.basename(summary_json_path)
    dataset_name = base.split("_")[0]

    token_budget = None
    if "token_budget" in base:
        try:
            token_budget = int(base.split("token_budget_")[1].split("_")[0])
        except:  # noqa: E722
            token_budget = None  
    else:
        token_budget = 100

    loaded = load_summary_file(summary_json_path)

    refs = loaded["references"]
    preds = loaded["predictions"]
    records = loaded["records"]

    rouge_scores = compute_rouge(refs, preds)
    bert_scores = compute_bertscore(refs, preds) if use_bertscore else {}
    token_stats = compute_token_stats(records)

    result = {
        "file": summary_json_path,
        "dataset": dataset_name,
        "budget": token_budget,
        **rouge_scores,
        **bert_scores,
        **token_stats
    }

    return result

def save_evaluation_results(results: dict, out_path: str = "results/metrics.csv"):
    """Appends a singel evalution dict to the metrics CSV file.
    CSV conatianing al evaluation results.
    """

    os.makedirs("results", exist_ok = True)

    # Define consistent column order
    FIELD_ORDER = [
        "file",
        "dataset",
        "token_budget",
        "rouge1", "rouge2", "rougeL",
        "bert_score_f1",
        "avg_tokens_before", "avg_tokens_after",
        "percentage_reduction"
    ]

    # Ensure only valid keys included
    row = {k: results.get(k) for k in FIELD_ORDER}

    file_exists = os.path.exists(out_path)

    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames = FIELD_ORDER)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"Saved metrics to {out_path}")

def significance_test_bootstrap(full_scores, trunc_scores, n_samples=1000, seed=42):
    """
    Bootstrap significance test between full-context and truncated ROUGE scores.

    Args:
        full_scores: list of per-document ROUGE scores for full summaries
        trunc_scores: list of per-document ROUGE scores for truncated summaries
        n_samples: number of bootstrap resamples (1000–2000 recommended)

    Returns:
        dict with:
            mean_diff: average difference full - truncated
            ci_low: lower CI 2.5%
            ci_high: upper CI 97.5%
            p_value: two-sided bootstrap p-value
    """
    assert len(full_scores) == len(trunc_scores), "Scores must align one-to-one"

    rng = np.random.default_rng(seed)
    diffs = []

    full_scores = np.array(full_scores)
    trunc_scores = np.array(trunc_scores)
    n = len(full_scores)

    ## Compute bootstrap distribution
    for _ in range(n_samples):
        idx = rng.integers(0, n, n)  ## sample with replacement
        diff = full_scores[idx].mean() - trunc_scores[idx].mean()
        diffs.append(diff)

    diffs = np.array(diffs)
    mean_diff = float(diffs.mean())
    ci_low = float(np.percentile(diffs, 2.5))
    ci_high = float(np.percentile(diffs, 97.5))

    ## p-value: proportion of bootstrap samples that cross zero
    p_value = float(min(
        (diffs > 0).mean(),  ## probability full > trunc
        (diffs < 0).mean()   ## probability trunc > full
    ) * 2)  ## two-sided

    return {
        "mean_diff": mean_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value
    }
