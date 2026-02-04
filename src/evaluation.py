"""
evaluation.py - UPDATED VERSION

Evaluates summarization quality with ROUGE-L as primary metric (as stated in paper).
Implements statistical significance testing and cost analysis.

Paper: "We evaluate performance using ROUGE-L (primary) and BERTScore."

Author: Agrim Gupta
Updated for EMNLP 2023 submission
"""

import os 
import json 
from typing import List, Dict, Optional, Tuple, Any
import torch
import numpy as np
import csv
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

SUMMARIES_DIR = "data/processed/summaries/"
PAIRS_DIR = "data/processed/"
RESULTS_DIR = "results/"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================================
# LOADING AND VALIDATION
# ============================================================================

def load_summary_file(path: str) -> Dict[str, any]:
    """
    Loads summary JSON file and validates structure.
    
    Returns:
        Dictionary with references, predictions, and metadata
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Summary file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {path}: {e}")
    
    # Validate structure
    if not isinstance(data, list):
        raise ValueError(f"Summary file must contain a list, got {type(data)}")
    
    # Extract data with validation
    references = []
    predictions = []
    ids = []
    tokens_before = []
    tokens_after = []
    tokens_input = []
    tokens_output = []
    model_names = []
    truncation_methods = []
    salience_types = []
    token_budgets = []
    
    for i, record in enumerate(data):
        # Required fields
        if "id" not in record:
            raise ValueError(f"Record {i} missing 'id' field")
        if "references" not in record and "reference_summary" not in record:
            raise ValueError(f"Record {i} missing reference summary")
        if "generated_summary" not in record:
            raise ValueError(f"Record {i} missing generated summary")
        
        # Extract with fallbacks
        ref = record.get("references") or record.get("reference_summary") or ""
        pred = record.get("generated_summary", "")
        
        # Clean whitespace
        ref = str(ref).strip()
        pred = str(pred).strip()
        
        # Skip if either is empty
        if not ref or not pred:
            print(f"⚠ Warning: Empty reference or prediction for {record.get('id')}, skipping")
            continue
        
        references.append(ref)
        predictions.append(pred)
        ids.append(record["id"])
        
        # Optional fields
        tokens_before.append(record.get("tokens_before"))
        tokens_after.append(record.get("tokens_after"))
        tokens_input.append(record.get("tokens_input"))
        tokens_output.append(record.get("tokens_output"))
        model_names.append(record.get("model_name"))
        truncation_methods.append(record.get("truncation_method"))
        salience_types.append(record.get("salience_type"))
        token_budgets.append(record.get("token_budget"))
    
    if not references:
        raise ValueError(f"No valid reference-prediction pairs found in {path}")
    
    print(f"✅ Loaded {len(references)} valid samples from {os.path.basename(path)}")
    
    return {
        "ids": ids,
        "records": data,
        "references": references,
        "predictions": predictions,
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
        "tokens_input": tokens_input,
        "tokens_output": tokens_output,
        "model_names": model_names,
        "truncation_methods": truncation_methods,
        "salience_types": salience_types,
        "token_budgets": token_budgets,
    }

# ============================================================================
# METRICS COMPUTATION
# ============================================================================

def compute_rouge(references: List[str], predictions: List[str]) -> Dict[str, any]:
    """
    Computes ROUGE scores with ROUGE-L as primary metric.
    
    Paper: "We evaluate performance using ROUGE-L (primary) and BERTScore."
    
    Returns:
        Dictionary with ROUGE scores and lists for statistical testing
    """
    print(f"  Computing ROUGE for {len(references)} samples...")
    
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True
    )
    
    r1_scores, r2_scores, rl_scores = [], [], []
    
    for ref, pred in tqdm(zip(references, predictions), total=len(references), desc="ROUGE", leave=False):
        if not ref or not pred:
            continue
        
        scores = scorer.score(ref, pred)
        r1_scores.append(scores["rouge1"].fmeasure)
        r2_scores.append(scores["rouge2"].fmeasure)
        rl_scores.append(scores["rougeL"].fmeasure)
    
    if not r1_scores:
        raise ValueError("No valid ROUGE scores computed")
    
    return {
        "rouge1": float(np.mean(r1_scores)),
        "rouge2": float(np.mean(r2_scores)),
        "rougeL": float(np.mean(rl_scores)),  # PRIMARY METRIC
        "rouge1_list": r1_scores,
        "rouge2_list": r2_scores,
        "rougeL_list": rl_scores,  # For statistical testing
        "rouge1_std": float(np.std(r1_scores)),
        "rouge2_std": float(np.std(r2_scores)),
        "rougeL_std": float(np.std(rl_scores)),
    }

def compute_bertscore(
    references: List[str], 
    predictions: List[str],
    device: Optional[str] = None
) -> Dict[str, float]:
    """
    Computes BERTScore for semantic similarity.
    
    Paper: Secondary metric to ROUGE-L.
    """
    print(f"  Computing BERTScore for {len(references)} samples...")
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # Use smaller batch size for memory efficiency
        P, R, F1 = bert_score(
            predictions,
            references,
            lang="en",
            device=device,
            batch_size=16 if device == "cuda" else 4,
            rescale_with_baseline=True,  # Better for comparison
        )
        
        return {
            "bert_score_f1": float(F1.mean()),
            "bert_score_precision": float(P.mean()),
            "bert_score_recall": float(R.mean()),
            "bert_score_f1_std": float(F1.std()),
        }
    except Exception as e:
        print(f"⚠ BERTScore failed: {e}")
        return {
            "bert_score_f1": 0.0,
            "bert_score_precision": 0.0,
            "bert_score_recall": 0.0,
            "bert_score_f1_std": 0.0,
        }

def compute_token_stats(records: List[dict]) -> Dict[str, float]:
    """
    Computes token compression statistics.
    
    Paper: "We define the compression ratio as L̂/L"
    """
    # Try different field names
    before = []
    after = []
    
    for record in records:
        # Try multiple possible field names
        b = record.get("tokens_before") or record.get("tokens_input")
        a = record.get("tokens_after") or record.get("tokens_output")
        
        if b is not None and a is not None:
            before.append(float(b))
            after.append(float(a))
    
    if not before or not after:
        return {
            "avg_tokens_before": None,
            "avg_tokens_after": None,
            "percentage_reduction": None,
            "avg_compression_ratio": None,
        }
    
    avg_before = float(np.mean(before))
    avg_after = float(np.mean(after))
    compression_ratio = avg_after / avg_before if avg_before > 0 else 0
    
    return {
        "avg_tokens_before": avg_before,
        "avg_tokens_after": avg_after,
        "percentage_reduction": 100 * (1 - compression_ratio),
        "avg_compression_ratio": compression_ratio,
        "std_tokens_before": float(np.std(before)),
        "std_tokens_after": float(np.std(after)),
    }

def compute_cost_stats(records: List[dict]) -> Dict[str, float]:
    """
    Estimates inference costs using current LLM API pricing.
    
    Paper: "Inference cost is estimated using current per-token pricing"
    """
    try:
        from src.cost_model import estimate_cost
        cost_function_available = True
    except ImportError:
        cost_function_available = False
        print("⚠ Cost model not available, skipping cost estimation")
    
    if not cost_function_available:
        return {
            "avg_cost_usd": None,
            "total_cost_usd": None,
            "cost_per_1k_tokens": None,
        }
    
    costs = []
    
    for record in records:
        tokens_in = record.get("tokens_input") or record.get("tokens_before")
        tokens_out = record.get("tokens_output") or record.get("tokens_after")
        model_name = record.get("model_name")
        
        if tokens_in is None or tokens_out is None or not model_name:
            continue
        
        try:
            cost = estimate_cost(
                tokens_in=int(tokens_in),
                tokens_out=int(tokens_out),
                model_name=model_name
            )
            if cost is not None:
                costs.append(float(cost))
        except Exception as e:
            continue
    
    if not costs:
        return {
            "avg_cost_usd": None,
            "total_cost_usd": None,
            "cost_per_1k_tokens": None,
        }
    
    total_cost = float(np.sum(costs))
    avg_cost = float(np.mean(costs))
    
    # Estimate cost per 1K tokens
    total_tokens = sum([
        (r.get("tokens_input") or r.get("tokens_before") or 0) + 
        (r.get("tokens_output") or r.get("tokens_after") or 0)
        for r in records
    ])
    
    cost_per_1k = (total_cost / total_tokens * 1000) if total_tokens > 0 else 0
    
    return {
        "avg_cost_usd": avg_cost,
        "total_cost_usd": total_cost,
        "cost_per_1k_tokens": cost_per_1k,
    }

# ============================================================================
# STATISTICAL SIGNIFICANCE TESTING
# ============================================================================

def significance_test_bootstrap(
    baseline_scores: List[float],
    treatment_scores: List[float],
    n_samples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42
) -> Dict[str, float]:
    """
    Bootstrap significance test as described in paper.
    
    Paper: "Bootstrap confidence intervals are computed"
    
    Returns:
        Dictionary with mean difference, confidence interval, and p-value
    """
    if len(baseline_scores) != len(treatment_scores):
        raise ValueError(
            f"Score length mismatch: baseline={len(baseline_scores)}, "
            f"treatment={len(treatment_scores)}"
        )
    
    print(f"  Running bootstrap test with {n_samples} samples...")
    
    np.random.seed(seed)
    n = len(baseline_scores)
    
    # Convert to numpy arrays
    baseline = np.array(baseline_scores)
    treatment = np.array(treatment_scores)
    
    # Observed difference
    observed_diff = np.mean(baseline) - np.mean(treatment)
    
    # Bootstrap sampling
    bootstrap_diffs = []
    
    for _ in range(n_samples):
        # Sample with replacement
        indices = np.random.randint(0, n, n)
        baseline_sample = baseline[indices]
        treatment_sample = treatment[indices]
        
        diff = np.mean(baseline_sample) - np.mean(treatment_sample)
        bootstrap_diffs.append(diff)
    
    bootstrap_diffs = np.array(bootstrap_diffs)
    
    # Confidence interval
    alpha = (1 - confidence_level) / 2
    ci_low = np.percentile(bootstrap_diffs, 100 * alpha)
    ci_high = np.percentile(bootstrap_diffs, 100 * (1 - alpha))
    
    # P-value (two-sided)
    # Proportion of bootstrap samples where absolute difference >= observed absolute difference
    abs_observed = abs(observed_diff)
    abs_bootstrap = abs(bootstrap_diffs - observed_diff)  # Center at observed
    p_value = np.mean(abs_bootstrap >= abs_observed)
    
    return {
        "mean_diff": float(observed_diff),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": float(p_value),
        "significant_95": p_value < 0.05,
        "significant_99": p_value < 0.01,
    }

# ============================================================================
# METADATA PARSING (FIXED)
# ============================================================================

def parse_metadata_from_filename(filename: str) -> Tuple[str, Optional[str], Optional[int]]:
    """
    Parses metadata from summary filename.
    
    Returns:
        (dataset_name, truncation_method, token_budget)
    """
    basename = os.path.basename(filename).replace(".json", "").lower()
    
    # Determine dataset
    dataset = "unknown"
    for candidate in ["cnn_dailymail", "govreport", "arxiv"]:
        if candidate in basename:
            dataset = candidate
            break
    
    # Determine truncation method and salience type
    truncation_method = None
    salience_type = None
    
    # Check for salience methods first
    if "tfidf" in basename:
        truncation_method = "salience"
        salience_type = "tfidf"
    elif "cosine" in basename:
        truncation_method = "salience"
        salience_type = "cosine"
    elif "hybrid" in basename:
        truncation_method = "salience"
        salience_type = "hybrid"
    elif "first_k" in basename or "first-k" in basename:
        truncation_method = "first_k"
    elif "random_k" in basename or "random-k" in basename:
        truncation_method = "random_k"
    elif "lead_n" in basename or "lead-n" in basename:
        truncation_method = "lead_n"
    elif "full" in basename and "summaries" in basename:
        truncation_method = "full"
    
    # Extract token budget
    token_budget = None
    parts = basename.split("_")
    
    for i, part in enumerate(parts):
        if part == "budget" and i + 1 < len(parts):
            try:
                token_budget = int(parts[i + 1])
                break
            except ValueError:
                pass
    
    # Special handling for known datasets
    if token_budget is None:
        if dataset == "cnn_dailymail":
            token_budget = 512
        elif dataset in ["govreport", "arxiv"]:
            token_budget = 4096
    
    return dataset, truncation_method, salience_type, token_budget

# ============================================================================
# MAIN EVALUATION FUNCTION
# ============================================================================

def evaluate_summary_file(
    summary_path: str,
    baseline_summary_path: Optional[str] = None,
    use_bertscore: bool = True
) -> Dict[str, any]:
    """
    Comprehensive evaluation of a summary file.
    
    Paper: "We compare full context summary against truncated inputs"
    
    Args:
        summary_path: Path to summary file to evaluate
        baseline_summary_path: Optional path to baseline (full context) for comparison
        use_bertscore: Whether to compute BERTScore
    
    Returns:
        Complete evaluation metrics dictionary
    """
    print(f"\n{'='*60}")
    print(f"EVALUATING: {os.path.basename(summary_path)}")
    print(f"{'='*60}")
    
    # Load and validate summary file
    summary_data = load_summary_file(summary_path)
    
    # Parse metadata
    dataset, trunc_method, salience_type, token_budget = parse_metadata_from_filename(summary_path)
    
    # Compute metrics
    rouge_results = compute_rouge(summary_data["references"], summary_data["predictions"])
    
    if use_bertscore:
        bert_results = compute_bertscore(summary_data["references"], summary_data["predictions"])
    else:
        bert_results = {}
    
    token_stats = compute_token_stats(summary_data["records"])
    cost_stats = compute_cost_stats(summary_data["records"])
    
    # Statistical significance testing if baseline provided
    sig_results = {}
    if baseline_summary_path and os.path.exists(baseline_summary_path):
        print(f"  Comparing with baseline: {os.path.basename(baseline_summary_path)}")
        
        try:
            baseline_data = load_summary_file(baseline_summary_path)
            baseline_rouge = compute_rouge(baseline_data["references"], baseline_data["predictions"])
            
            # Use ROUGE-L for significance testing (primary metric)
            sig_results = significance_test_bootstrap(
                baseline_rouge["rougeL_list"],
                rouge_results["rougeL_list"]
            )
            
            # Add baseline scores for reference
            sig_results.update({
                "baseline_rougeL": baseline_rouge["rougeL"],
                "baseline_rouge1": baseline_rouge["rouge1"],
                "baseline_rouge2": baseline_rouge["rouge2"],
            })
        except Exception as e:
            print(f"⚠ Significance testing failed: {e}")
    
    # Compile all results
    results = {
        "file": summary_path,
        "dataset": dataset,
        "truncation_method": trunc_method,
        "salience_type": salience_type,
        "token_budget": token_budget,
        "sample_count": len(summary_data["references"]),
        
        # ROUGE scores
        "rougeL": rouge_results["rougeL"],  # PRIMARY
        "rouge1": rouge_results["rouge1"],
        "rouge2": rouge_results["rouge2"],
        "rougeL_std": rouge_results["rougeL_std"],
        "rouge1_std": rouge_results["rouge1_std"],
        "rouge2_std": rouge_results["rouge2_std"],
        
        # Token statistics
        **token_stats,
        
        # Cost statistics
        **cost_stats,
        
        # Statistical significance
        **sig_results,
    }
    
    # Add BERTScore if computed
    if bert_results:
        results.update(bert_results)
    
    # Print summary
    print(f"\n📊 RESULTS SUMMARY:")
    print(f"  Dataset: {dataset}")
    print(f"  Method: {trunc_method} ({salience_type if salience_type else 'N/A'})")
    print(f"  Token budget: {token_budget}")
    print(f"  ROUGE-L (primary): {results['rougeL']:.4f} ± {results['rougeL_std']:.4f}")
    print(f"  ROUGE-1: {results['rouge1']:.4f}")
    print(f"  ROUGE-2: {results['rouge2']:.4f}")
    
    if token_stats["avg_tokens_before"] and token_stats["avg_tokens_after"]:
        print(f"  Token reduction: {token_stats['percentage_reduction']:.1f}%")
        print(f"  Compression ratio: {token_stats['avg_compression_ratio']:.3f}")
    
    if sig_results:
        print(f"  vs Baseline difference: {sig_results.get('mean_diff', 0):.4f}")
        print(f"  p-value: {sig_results.get('p_value', 1):.4f}")
        print(f"  Significant (p<0.05): {sig_results.get('significant_95', False)}")
    
    print(f"{'='*60}")
    
    return results

def save_evaluation_results(
    results: Dict[str, any],
    out_path: str = "results/metrics.csv",
    append: bool = True
) -> None:
    """
    Saves evaluation results to CSV file.
    
    Args:
        results: Evaluation results dictionary
        out_path: Path to CSV file
        append: Whether to append to existing file
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # Define column order
    field_order = [
        "file", "dataset", "truncation_method", "salience_type", "token_budget",
        "sample_count",
        "rougeL", "rouge1", "rouge2",
        "rougeL_std", "rouge1_std", "rouge2_std",
        "bert_score_f1", "bert_score_precision", "bert_score_recall", "bert_score_f1_std",
        "avg_tokens_before", "std_tokens_before",
        "avg_tokens_after", "std_tokens_after",
        "percentage_reduction", "avg_compression_ratio",
        "avg_cost_usd", "total_cost_usd", "cost_per_1k_tokens",
        "baseline_rougeL", "baseline_rouge1", "baseline_rouge2",
        "mean_diff", "ci_low", "ci_high", "p_value",
        "significant_95", "significant_99",
    ]
    
    # Create row with all fields
    row = {field: results.get(field) for field in field_order}
    
    # Check if file exists
    file_exists = os.path.exists(out_path) and append
    
    with open(out_path, "a" if file_exists else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=field_order)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(row)
    
    print(f"💾 Results saved to {out_path}")

# ============================================================================
# BATCH EVALUATION
# ============================================================================

def evaluate_all_summaries(
    summaries_dir: str = SUMMARIES_DIR,
    output_csv: str = "results/metrics.csv"
) -> List[Dict[str, any]]:
    """
    Evaluates all summary files in a directory.
    """
    print(f"\n🔍 EVALUATING ALL SUMMARIES IN {summaries_dir}")
    
    if not os.path.exists(summaries_dir):
        print(f"❌ Directory not found: {summaries_dir}")
        return []
    
    # Find all summary files
    summary_files = []
    for root, _, files in os.walk(summaries_dir):
        for file in files:
            if file.endswith(".json") and ("summary" in file.lower() or "summaries" in file.lower()):
                summary_files.append(os.path.join(root, file))
    
    print(f"Found {len(summary_files)} summary files")
    
    # Identify baseline files
    baseline_files = {}
    for file in summary_files:
        basename = os.path.basename(file).lower()
        if "full" in basename and "summary" in basename:
            # Extract dataset name
            for dataset in ["cnn_dailymail", "govreport", "arxiv"]:
                if dataset in basename:
                    baseline_files[dataset] = file
                    print(f"  Baseline for {dataset}: {os.path.basename(file)}")
                    break
    
    # Evaluate each file
    all_results = []
    
    for summary_file in tqdm(summary_files, desc="Evaluating summaries"):
        try:
            # Determine baseline for this dataset
            baseline = None
            for dataset, baseline_file in baseline_files.items():
                if dataset in summary_file.lower():
                    baseline = baseline_file
                    break
            
            # Evaluate
            results = evaluate_summary_file(summary_file, baseline)
            
            # Save to CSV
            save_evaluation_results(results, output_csv)
            
            all_results.append(results)
            
        except Exception as e:
            print(f"❌ Failed to evaluate {os.path.basename(summary_file)}: {e}")
            continue
    
    print(f"\n✅ Evaluated {len(all_results)} summary files")
    print(f"   Results saved to {output_csv}")
    
    return all_results

# ============================================================================
# TEST FUNCTION
# ============================================================================

def test_evaluation():
    """Test the evaluation functions."""
    print("\n🧪 TESTING EVALUATION MODULE")
    print("="*60)
    
    # Create test data
    test_summaries = [
        {
            "id": "test_1",
            "references": "Large language models have changed natural language processing.",
            "generated_summary": "LLMs transformed NLP field significantly.",
            "tokens_before": 500,
            "tokens_after": 200,
            "model_name": "facebook/bart-large-cnn",
        },
        {
            "id": "test_2",
            "references": "Machine learning requires large datasets for training.",
            "generated_summary": "ML needs big data for effective training.",
            "tokens_before": 600,
            "tokens_after": 250,
            "model_name": "facebook/bart-large-cnn",
        }
    ]
    
    # Save test file
    test_file = "test_summary.json"
    with open(test_file, "w") as f:
        json.dump(test_summaries, f, indent=2)
    
    print(f"Created test file: {test_file}")
    
    try:
        # Test loading
        print("\n1. Testing load_summary_file...")
        loaded = load_summary_file(test_file)
        print(f"   Loaded {len(loaded['references'])} samples")
        
        # Test ROUGE
        print("\n2. Testing compute_rouge...")
        rouge = compute_rouge(loaded["references"], loaded["predictions"])
        print(f"   ROUGE-L: {rouge['rougeL']:.4f}")
        print(f"   ROUGE-1: {rouge['rouge1']:.4f}")
        print(f"   ROUGE-2: {rouge['rouge2']:.4f}")
        
        # Test BERTScore
        print("\n3. Testing compute_bertscore...")
        bert = compute_bertscore(loaded["references"], loaded["predictions"])
        print(f"   BERTScore F1: {bert['bert_score_f1']:.4f}")
        
        # Test token stats
        print("\n4. Testing compute_token_stats...")
        token_stats = compute_token_stats(loaded["records"])
        print(f"   Token reduction: {token_stats.get('percentage_reduction', 0):.1f}%")
        
        # Test significance testing
        print("\n5. Testing significance_test_bootstrap...")
        baseline_scores = [0.3, 0.4, 0.35, 0.38, 0.32]
        treatment_scores = [0.28, 0.35, 0.33, 0.36, 0.30]
        sig_test = significance_test_bootstrap(baseline_scores, treatment_scores, n_samples=100)
        print(f"   Mean difference: {sig_test['mean_diff']:.4f}")
        print(f"   p-value: {sig_test['p_value']:.4f}")
        
        print("\n✅ All tests passed!")
        
    finally:
        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"\nCleaned up test file: {test_file}")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate summarization results")
    parser.add_argument("--summary", type=str, help="Path to summary JSON file")
    parser.add_argument("--baseline", type=str, help="Path to baseline summary file")
    parser.add_argument("--all", action="store_true", help="Evaluate all summaries in directory")
    parser.add_argument("--output", type=str, default="results/metrics.csv", help="Output CSV path")
    parser.add_argument("--test", action="store_true", help="Run tests")
    
    args = parser.parse_args()
    
    if args.test:
        test_evaluation()
    
    elif args.all:
        evaluate_all_summaries(output_csv=args.output)
    
    elif args.summary:
        results = evaluate_summary_file(args.summary, args.baseline)
        save_evaluation_results(results, args.output)
    
    else:
        print("Usage:")
        print("  python evaluation.py --test")
        print("  python evaluation.py --all")
        print("  python evaluation.py --summary path/to/summary.json --baseline path/to/baseline.json")
        print("  python evaluation.py --summary path/to/summary.json --output custom_results.csv")