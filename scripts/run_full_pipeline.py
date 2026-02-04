"""
run_full_pipeline.py - UPDATED VERSION

End-to-end orchestration for the salience-based token-budget truncation experiments.
Updated to use all corrected modules with constant generation parameters.

Author: Agrim Gupta
Updated for EMNLP 2023 submission
"""

import os
import sys
import json 
import numpy as np
from typing import Dict, List, Tuple, Optional
import traceback

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import updated modules
from src.data_loader import prepare_data, load_dataset
from src.embedding import load_embedding_model, build_embedding_index
from src.salience_scoring import compute_salience, save_salience_scores
from src.truncation import truncate_dataset
from src.summarizer import summarize_full_pairs, summarize_truncated_files
from src.evaluation import evaluate_summary_file, save_evaluation_results, evaluate_all_summaries

# Uncomment if you have visualization module
# from src.visualization import ...

# ============================================================================
# CONFIGURATION
# ============================================================================

# All truncation methods as per paper
TRUNCATION_METHODS = [
    ("salience", "tfidf"),
    ("salience", "cosine"),
    ("salience", "hybrid"),
    ("first_k", None),
    ("random_k", None),
    ("lead_n", None),
]

def load_dataset_config() -> Dict[str, dict]:
    """
    Dataset configuration matching paper specifications.
    
    Paper: 
      - CNN/DailyMail: 512 token budget
      - GovReport: 4096 token budget  
      - ArXiv: 4096 token budget
    """
    return {
        "cnn_dailymail": {
            "token_budget": 512,        # Paper: 512 tokens
            "baseline_model": "facebook/bart-large-cnn",
            "truncated_model": "facebook/bart-large-cnn",  # Always BART for CNN
            "input_field": "article",
            "summary_field": "highlights",
            "chunk_size": 512,          # Sentence-aligned chunking
            "skip_full": False,         # Run full context baseline
        },
        "govreport": {
            "token_budget": 4096,       # Paper: 4096 tokens
            "baseline_model": "allenai/led-base-16384",
            "truncated_model": "allenai/led-base-16384",  # LED for long docs
            "input_field": "document",
            "summary_field": "summary",
            "chunk_size": 1024,         # Sentence-aligned chunking
            "skip_full": False,         # Run full context baseline
        },
        "arxiv": {
            "token_budget": 4096,       # Paper: 4096 tokens
            "baseline_model": "allenai/led-base-16384",
            "truncated_model": "allenai/led-base-16384",  # LED for long docs
            "input_field": "article",
            "summary_field": "abstract",
            "chunk_size": 1024,         # Sentence-aligned chunking
            "skip_full": True,          # Paper: No full context baseline for ArXiv
        }
    }

# ============================================================================
# STEP 1: PREPARE DATA
# ============================================================================

def run_prepare_data(dataset_name: str, cfg: dict) -> str:
    """
    Prepare dataset with sentence-aligned chunking.
    
    Paper: "Each document D is segmented into non-overlapping, 
    sentence-aligned chunks using a deterministic greedy strategy."
    """
    print(f"\n{'='*60}")
    print(f"STEP 1: PREPARING DATA - {dataset_name}")
    print(f"{'='*60}")
    
    try:
        # Load dataset from disk
        ds = load_dataset(dataset_name)
        
        # Prepare data with sentence-aligned chunking
        print(f"  Chunk size: {cfg['chunk_size']} tokens")
        print(f"  Input field: {cfg['input_field']}")
        print(f"  Summary field: {cfg['summary_field']}")
        
        # Note: prepare_data now saves to file and returns pairs
        save_path = f"data/processed/{dataset_name}_pairs.json"
        
        # This will use our updated data_loader with proper chunking
        pairs = prepare_data(
            dataset=ds,
            dataset_name=dataset_name,
            input_field=cfg["input_field"],
            summary_field=cfg["summary_field"],
            chunk=True,  # Always chunk for salience scoring
            save_path=save_path
        )
        
        print(f"✅ Prepared {len(pairs)} chunks")
        return save_path
        
    except Exception as e:
        print(f"❌ Failed to prepare data for {dataset_name}: {e}")
        traceback.print_exc()
        raise

# ============================================================================
# STEP 2: BUILD EMBEDDINGS
# ============================================================================

def run_build_embeddings(pairs_file: str, dataset_name: str):
    """
    Build embeddings for cosine salience scoring.
    
    Paper: "Cosine: Embedding similarity with reference summaries"
    """
    print(f"\n{'='*60}")
    print(f"STEP 2: BUILDING EMBEDDINGS - {dataset_name}")
    print(f"{'='*60}")
    
    try:
        # Load embedding model
        emb_model = load_embedding_model()
        
        # Build embedding index
        emb_path, ids_path = build_embedding_index(
            pairs_file=pairs_file,
            model=emb_model,
            save_dir="data/processed/embeddings",
            batch_size=32
        )
        
        print(f"✅ Embeddings built: {emb_path}")
        return emb_path, ids_path
        
    except Exception as e:
        print(f"❌ Failed to build embeddings for {dataset_name}: {e}")
        traceback.print_exc()
        return None, None

# ============================================================================
# STEP 3: SALIENCE SCORING
# ============================================================================

def run_salience_scoring(dataset_name: str, salience_type: str, pairs_path: str, emb_path: Optional[str] = None):
    """
    Compute salience scores using reference summaries as oracle.
    
    Paper: "Scores are computed treating reference summaries as an oracle"
    """
    print(f"\n{'='*60}")
    print(f"STEP 3: SALIENCE SCORING - {dataset_name} ({salience_type})")
    print(f"{'='*60}")
    
    try:
        # Load pairs data
        with open(pairs_path, "r", encoding="utf-8") as f:
            pairs_data = json.load(f)
        
        # Load embedding model if needed
        model = None
        if salience_type in ["cosine", "hybrid"]:
            if emb_path is None:
                raise ValueError(f"emb_path required for {salience_type} salience")
            model = load_embedding_model()
        
        # Compute salience scores
        print(f"  Computing {salience_type} salience...")
        scores = compute_salience(
            pairs=pairs_data,
            method=salience_type,
            model=model,
            embedding_path=emb_path,
            alpha=0.7  # Default from paper
        )
        
        # Get chunk IDs
        if isinstance(pairs_data, dict) and "pairs" in pairs_data:
            pairs_list = pairs_data["pairs"]
        else:
            pairs_list = pairs_data
        
        chunk_ids = [p["id"] for p in pairs_list]
        
        # Save scores
        scores_path = save_salience_scores(
            scores=scores,
            ids=chunk_ids,
            dataset_name=dataset_name,
            salience_type=salience_type
        )
        
        print(f"✅ Salience scores saved: {scores_path}")
        return scores_path
        
    except Exception as e:
        print(f"❌ Failed to compute {salience_type} salience for {dataset_name}: {e}")
        traceback.print_exc()
        return None

# ============================================================================
# STEP 4: TRUNCATION
# ============================================================================

def run_truncation(dataset_name: str, token_budget: int, truncation_method: str, salience_type: Optional[str] = None):
    """
    Create truncated inputs under fixed token budget.
    
    Paper: "Given a token budget B, chunks are selected in descending order of s(c_i)"
    """
    print(f"\n{'='*60}")
    print(f"STEP 4: TRUNCATION - {dataset_name}")
    print(f"  Method: {truncation_method}")
    print(f"  Salience type: {salience_type}")
    print(f"  Token budget: {token_budget}")
    print(f"{'='*60}")
    
    try:
        stats = truncate_dataset(
            dataset_name=dataset_name,
            token_budget=token_budget,
            truncation_method=truncation_method,
            salience_type=salience_type,
            seed=42  # Fixed seed for reproducibility
        )
        
        print(f"✅ Truncation complete")
        print(f"   Avg tokens before: {stats['avg_tokens_before']:.1f}")
        print(f"   Avg tokens after: {stats['avg_tokens_after']:.1f}")
        print(f"   Reduction: {stats['percentage_reduction']:.1f}%")
        
        return stats
        
    except Exception as e:
        print(f"❌ Failed to truncate {dataset_name}: {e}")
        traceback.print_exc()
        return None

# ============================================================================
# STEP 5: SUMMARIZATION
# ============================================================================

def run_summarization(dataset_name: str, cfg: dict, truncation_method: str, salience_type: Optional[str] = None):
    """
    Generate summaries with CONSTANT generation parameters.
    
    Paper: "Decoding parameters are held constant; selection is independent 
    of salience strategy and dataset."
    """
    print(f"\n{'='*60}")
    print(f"STEP 5: SUMMARIZATION - {dataset_name}")
    print(f"  Method: {truncation_method}")
    print(f"  Salience type: {salience_type}")
    print(f"{'='*60}")
    
    try:
        # Determine model based on truncation method
        if truncation_method == "full":
            # Full context baseline
            model_name = cfg["baseline_model"]
            pairs_file = f"data/processed/{dataset_name}_pairs.json"
            
            if not os.path.exists(pairs_file):
                print(f"⚠ Pairs file not found: {pairs_file}")
                return None
            
            print(f"  Model: {model_name} (baseline)")
            print(f"  Generating full context summaries...")
            
            results = summarize_full_pairs(
                pairs_file=pairs_file,
                model_name=model_name,
                batch_size=4,  # Conservative for GPU memory
                save_intermediate=True
            )
            
            print(f"✅ Generated {len(results)} baseline summaries")
            return results
            
        else:
            # Truncated summaries
            model_name = cfg["truncated_model"]
            
            # Build truncated filename
            if truncation_method == "salience":
                truncated_file = f"data/processed/truncated_texts/{dataset_name}_{salience_type}_salience_token_budget_{cfg['token_budget']}_truncated.json"
            else:
                truncated_file = f"data/processed/truncated_texts/{dataset_name}_{truncation_method}_budget_{cfg['token_budget']}.json"
            
            if not os.path.exists(truncated_file):
                print(f"⚠ Truncated file not found: {truncated_file}")
                return None
            
            print(f"  Model: {model_name}")
            print(f"  Generating truncated summaries...")
            
            results = summarize_truncated_files(
                truncated_json_path=truncated_file,
                model_name=model_name,
                batch_size=4,  # Conservative for GPU memory
                save_intermediate=True
            )
            
            print(f"✅ Generated {len(results)} truncated summaries")
            return results
            
    except Exception as e:
        print(f"❌ Failed to generate summaries for {dataset_name}: {e}")
        traceback.print_exc()
        return None

# ============================================================================
# STEP 6: EVALUATION
# ============================================================================

def run_evaluation(dataset_name: str, cfg: dict, truncation_method: str, salience_type: Optional[str] = None):
    """
    Evaluate summaries with statistical significance testing.
    
    Paper: "Bootstrap confidence intervals are computed"
    """
    print(f"\n{'='*60}")
    print(f"STEP 6: EVALUATION - {dataset_name}")
    print(f"  Method: {truncation_method}")
    print(f"  Salience type: {salience_type}")
    print(f"{'='*60}")
    
    try:
        # Determine summary file paths
        if truncation_method == "full":
            summary_file = f"data/processed/summaries/{dataset_name}_pairs_full_summaries.json"
            baseline_file = None  # Full context is itself the baseline
        else:
            # Build summary filename
            if truncation_method == "salience":
                summary_file = f"data/processed/summaries/{dataset_name}_{salience_type}_salience_token_budget_{cfg['token_budget']}_truncated_summaries.json"
            else:
                summary_file = f"data/processed/summaries/{dataset_name}_{truncation_method}_budget_{cfg['token_budget']}_summaries.json"
            
            # Find baseline file
            baseline_file = f"data/processed/summaries/{dataset_name}_pairs_full_summaries.json"
            if not os.path.exists(baseline_file):
                print(f"⚠ Baseline file not found: {baseline_file}")
                baseline_file = None
        
        # Check if summary file exists
        if not os.path.exists(summary_file):
            print(f"⚠ Summary file not found: {summary_file}")
            return None
        
        # Evaluate
        print(f"  Summary file: {os.path.basename(summary_file)}")
        if baseline_file:
            print(f"  Baseline file: {os.path.basename(baseline_file)}")
        
        results = evaluate_summary_file(
            summary_path=summary_file,
            baseline_summary_path=baseline_file,
            use_bertscore=True
        )
        
        # Save to CSV
        save_evaluation_results(results)
        
        print(f"✅ Evaluation complete")
        print(f"   ROUGE-L: {results.get('rougeL', 0):.4f}")
        if results.get('percentage_reduction'):
            print(f"   Token reduction: {results['percentage_reduction']:.1f}%")
        
        return results
        
    except Exception as e:
        print(f"❌ Failed to evaluate {dataset_name}: {e}")
        traceback.print_exc()
        return None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_truncated_filename(dataset_name: str, truncation_method: str, salience_type: Optional[str], token_budget: int) -> str:
    """
    Generate consistent truncated filename.
    """
    if truncation_method == "salience":
        return f"{dataset_name}_{salience_type}_salience_token_budget_{token_budget}_truncated.json"
    else:
        return f"{dataset_name}_{truncation_method}_budget_{token_budget}.json"

def check_prerequisites(dataset_name: str, cfg: dict, truncation_method: str, salience_type: Optional[str] = None):
    """
    Check if prerequisites are met for each step.
    """
    issues = []
    
    # Check pairs file exists
    pairs_file = f"data/processed/{dataset_name}_pairs.json"
    if not os.path.exists(pairs_file):
        issues.append(f"Pairs file not found: {pairs_file}")
    
    # Check embeddings for salience methods
    if truncation_method == "salience" and salience_type in ["cosine", "hybrid"]:
        emb_file = f"data/processed/embeddings/{dataset_name}_embeddings.npy"
        if not os.path.exists(emb_file):
            issues.append(f"Embeddings not found: {emb_file}")
    
    # Check salience scores for salience truncation
    if truncation_method == "salience":
        salience_file = f"data/processed/salience_scores/{dataset_name}_{salience_type}_salience.json"
        if not os.path.exists(salience_file):
            issues.append(f"Salience scores not found: {salience_file}")
    
    # Check truncated file exists
    if truncation_method != "full":
        truncated_file = get_truncated_filename(dataset_name, truncation_method, salience_type, cfg["token_budget"])
        truncated_path = f"data/processed/truncated_texts/{truncated_file}"
        if not os.path.exists(truncated_path):
            issues.append(f"Truncated file not found: {truncated_path}")
    
    return issues

# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

def run_experiment_for_dataset(dataset_name: str, cfg: dict, run_full_pipeline: bool = True):
    """
    Run complete experiment pipeline for one dataset.
    """
    print(f"\n{'='*80}")
    print(f"RUNNING EXPERIMENTS FOR: {dataset_name.upper()}")
    print(f"Token budget: {cfg['token_budget']}")
    print(f"Baseline model: {cfg['baseline_model']}")
    print(f"Truncated model: {cfg['truncated_model']}")
    print(f"{'='*80}")
    
    # Track execution times and results
    results = {
        "dataset": dataset_name,
        "config": cfg,
        "methods": {}
    }
    
    try:
        # STEP 1: Prepare data (always run)
        print(f"\n📝 STEP 1: PREPARING DATA")
        pairs_file = run_prepare_data(dataset_name, cfg)
        results["pairs_file"] = pairs_file
        
        # STEP 2: Build embeddings (always run for cosine/hybrid)
        print(f"\n🔧 STEP 2: BUILDING EMBEDDINGS")
        emb_path, ids_path = run_build_embeddings(pairs_file, dataset_name)
        results["embeddings_path"] = emb_path
        
        # Run full context baseline if required
        if not cfg.get("skip_full", False):
            print(f"\n📊 STEP: FULL CONTEXT BASELINE")
            
            # Check if already exists
            baseline_summary = f"data/processed/summaries/{dataset_name}_pairs_full_summaries.json"
            if os.path.exists(baseline_summary):
                print(f"✅ Baseline already exists: {baseline_summary}")
            else:
                # Generate baseline
                baseline_results = run_summarization(
                    dataset_name=dataset_name,
                    cfg=cfg,
                    truncation_method="full",
                    salience_type=None
                )
                
                # Evaluate baseline
                if baseline_results:
                    eval_results = run_evaluation(
                        dataset_name=dataset_name,
                        cfg=cfg,
                        truncation_method="full",
                        salience_type=None
                    )
                    results["methods"]["full"] = eval_results
        
        # Run all truncation methods
        for truncation_method, salience_type in TRUNCATION_METHODS:
            print(f"\n{'~'*60}")
            print(f"METHOD: {truncation_method.upper()} {f'({salience_type})' if salience_type else ''}")
            print(f"{'~'*60}")
            
            # Skip if method not applicable
            if truncation_method == "salience" and salience_type is None:
                continue
            
            try:
                # Check prerequisites
                issues = check_prerequisites(dataset_name, cfg, truncation_method, salience_type)
                if issues:
                    print(f"⚠ Skipping due to missing prerequisites:")
                    for issue in issues:
                        print(f"   - {issue}")
                    continue
                
                # STEP 3: Salience scoring (for salience methods only)
                if truncation_method == "salience":
                    print(f"\n🎯 STEP 3: SALIENCE SCORING")
                    scores_path = run_salience_scoring(
                        dataset_name=dataset_name,
                        salience_type=salience_type,
                        pairs_path=pairs_file,
                        emb_path=emb_path
                    )
                    if not scores_path:
                        print(f"⚠ Skipping {salience_type} due to scoring failure")
                        continue
                
                # STEP 4: Truncation
                print(f"\n✂️ STEP 4: TRUNCATION")
                trunc_stats = run_truncation(
                    dataset_name=dataset_name,
                    token_budget=cfg["token_budget"],
                    truncation_method=truncation_method,
                    salience_type=salience_type
                )
                
                if not trunc_stats:
                    print(f"⚠ Skipping due to truncation failure")
                    continue
                
                # STEP 5: Summarization
                print(f"\n📝 STEP 5: SUMMARIZATION")
                summary_results = run_summarization(
                    dataset_name=dataset_name,
                    cfg=cfg,
                    truncation_method=truncation_method,
                    salience_type=salience_type
                )
                
                if not summary_results:
                    print(f"⚠ Skipping due to summarization failure")
                    continue
                
                # STEP 6: Evaluation
                print(f"\n📊 STEP 6: EVALUATION")
                eval_results = run_evaluation(
                    dataset_name=dataset_name,
                    cfg=cfg,
                    truncation_method=truncation_method,
                    salience_type=salience_type
                )
                
                if eval_results:
                    method_key = f"{truncation_method}_{salience_type}" if salience_type else truncation_method
                    results["methods"][method_key] = eval_results
                    print(f"✅ Completed {truncation_method} {salience_type if salience_type else ''}")
                
            except Exception as e:
                print(f"❌ Error in {truncation_method} {salience_type}: {e}")
                traceback.print_exc()
                continue
        
        print(f"\n✅ COMPLETED ALL METHODS FOR {dataset_name.upper()}")
        return results
        
    except Exception as e:
        print(f"❌ Fatal error for {dataset_name}: {e}")
        traceback.print_exc()
        return None

def run_visualization():
    """
    Generate plots from evaluation results.
    """
    print(f"\n{'='*60}")
    print(f"GENERATING VISUALIZATIONS")
    print(f"{'='*60}")
    
    metrics_csv = "results/metrics.csv"
    
    if not os.path.exists(metrics_csv):
        print(f"❌ Metrics CSV not found: {metrics_csv}")
        print("  Run evaluation first")
        return
    
    # Uncomment if you have visualization module
    # from src.visualization import evaluate_all_summaries
    # evaluate_all_summaries(metrics_csv)
    
    print(f"✅ Visualizations would be generated from {metrics_csv}")
    print("  (Uncomment visualization imports and function calls)")

def main():
    """
    Main execution function.
    """
    print(f"\n{'='*80}")
    print(f"SALIENCE-BASED ADAPTIVE TRUNCATION EXPERIMENTS")
    print(f"EMNLP 2023 Submission")
    print(f"{'='*80}")
    
    # Load configuration
    configs = load_dataset_config()
    
    # Create necessary directories
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/processed/embeddings", exist_ok=True)
    os.makedirs("data/processed/salience_scores", exist_ok=True)
    os.makedirs("data/processed/truncated_texts", exist_ok=True)
    os.makedirs("data/processed/summaries", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    all_results = {}
    
    # Run experiments for each dataset
    for dataset_name, cfg in configs.items():
        print(f"\n🎯 TARGET DATASET: {dataset_name.upper()}")
        
        try:
            results = run_experiment_for_dataset(dataset_name, cfg, run_full_pipeline=True)
            if results:
                all_results[dataset_name] = results
        except KeyboardInterrupt:
            print(f"\n⚠ Interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Failed to run {dataset_name}: {e}")
            traceback.print_exc()
            continue
    
    # Generate visualizations
    run_visualization()
    
    # Summary
    print(f"\n{'='*80}")
    print(f"EXPERIMENTS COMPLETE")
    print(f"{'='*80}")
    
    for dataset_name, results in all_results.items():
        print(f"\n📊 {dataset_name.upper()}:")
        if "methods" in results:
            for method, eval_results in results["methods"].items():
                if eval_results and "rougeL" in eval_results:
                    print(f"  {method:20} ROUGE-L: {eval_results['rougeL']:.4f}")
    
    print(f"\n📈 Results saved to:")
    print(f"  - Metrics: results/metrics.csv")
    print(f"  - Summaries: data/processed/summaries/")
    print(f"  - Truncated texts: data/processed/truncated_texts/")
    print(f"\n✅ ALL DONE!")

if __name__ == "__main__":
    main()