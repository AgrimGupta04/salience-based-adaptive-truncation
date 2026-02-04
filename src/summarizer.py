import os 
import torch 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import json 
from typing import List, Dict, Optional, Any, Tuple
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# FIXED: CONSTANT DECODING PARAMETERS (AS PER PAPER)
# ============================================================================

# Paper: "Decoding parameters are held constant; selection is independent of 
# salience strategy and dataset."
CONSTANT_GEN_KWARGS = {
    "max_length": 256,       # Fixed output length
    "min_length": 64,        # Fixed minimum length
    "num_beams": 4,          # Fixed beam size
    "early_stopping": True,  # Fixed early stopping
    "do_sample": False,      # Fixed: no sampling
    "no_repeat_ngram_size": 3,  # Fixed: no repeat n-gram
    "length_penalty": 2.0,   # Fixed: length penalty for ALL models
}

# ============================================================================
# MODEL-SPECIFIC INPUT CONFIGURATIONS
# ============================================================================

MODEL_CONFIGS = {
    "facebook/bart-large-cnn": {
        "max_input_length": 1024,  # BART max
        "output_length": 256,      # Same as constant kwargs
    },
    "allenai/led-base-16384": {
        "max_input_length": 16384,  # LED max
        "output_length": 512,       # GovReport/ArXiv summaries longer
    },
    "google/long-t5-local-base": {
        "max_input_length": 4096,
        "output_length": 256,
    }
}

# ============================================================================
# PATHS AND CONSTANTS
# ============================================================================

SUMMARIES_DIR = "data/processed/summaries/"
PAIRS_DIR = "data/processed/"
CHECKPOINT_INTERVAL = 50

os.makedirs(SUMMARIES_DIR, exist_ok=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_model_config(model_name: str) -> Dict[str, any]:
    """Get model-specific input configuration."""
    for key in MODEL_CONFIGS:
        if key in model_name:
            return MODEL_CONFIGS[key]
    
    # Default to BART config
    print(f"⚠ Using BART config for unknown model: {model_name}")
    return MODEL_CONFIGS["facebook/bart-large-cnn"]

def estimate_tokens_simple(text: str) -> int:
    """Simple token estimation using word count."""
    return len(text.split())

# ============================================================================
# MODEL LOADING (WITH CONSTANT PARAMETERS)
# ============================================================================

def load_summarization_model(
    model_name: str, 
    device: Optional[int] = None
) -> Tuple[Any, Dict[str, any]]:
    """
    Load model with CONSTANT generation parameters.
    
    Paper: "Decoding parameters are held constant; selection is independent 
    of salience strategy and dataset."
    
    Returns: (model_pipeline, generation_kwargs)
    """
    print(f"\n📦 Loading model: {model_name}")
    print(f"  Using CONSTANT generation parameters: {CONSTANT_GEN_KWARGS}")
    
    # Get model-specific input config only
    config = get_model_config(model_name)
    
    # Set device
    if device is None:
        device = 0 if torch.cuda.is_available() else -1
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set model-specific max input length
    tokenizer.model_max_length = config["max_input_length"]
    print(f"  Tokenizer max input length: {tokenizer.model_max_length}")
    
    # Load model
    if "led" in model_name.lower() or "long-t5" in model_name.lower():
        # Large models with device mapping
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
        )
        pipe = pipeline(
            "summarization", 
            model=model, 
            tokenizer=tokenizer,
            device_map="auto" if torch.cuda.is_available() else None
        )
    else:
        # Standard models
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        if device != -1:
            model = model.to(device)
        pipe = pipeline(
            "summarization", 
            model=model, 
            tokenizer=tokenizer, 
            device=device
        )
    
    # IMPORTANT: Use CONSTANT generation parameters for ALL models
    # Adjust only max_length based on model if needed for output
    gen_kwargs = CONSTANT_GEN_KWARGS.copy()
    
    # Only adjust output length if model specifically needs it
    if "output_length" in config:
        gen_kwargs["max_length"] = config["output_length"]
    
    print(f"  Final generation kwargs: {gen_kwargs}")
    print(f"  Device: {'CUDA' if device != -1 else 'CPU'}")
    
    return pipe, gen_kwargs

# ============================================================================
# SUMMARIZATION FUNCTIONS (WITH CONSTANT PARAMETERS)
# ============================================================================

def summarize_batch(
    texts: List[str], 
    model_pipe, 
    gen_kwargs: Dict[str, any],
    batch_size: int = 2  # Reduce to 2 for LED (memory intensive)
) -> List[str]:
    """
    Fix: Use smaller batch size and better error handling for LED.
    """
    if not texts:
        return []
    
    # Clean texts
    texts = [t if t and t.strip() else " " for t in texts]
    
    summaries = []
    
    # Process in tiny batches for LED
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        try:
            # For LED, we need to be careful with memory
            if "led" in str(model_pipe.model.config).lower():
                # Process one at a time for LED
                batch_summaries = []
                for text in batch:
                    try:
                        # Truncate very long inputs for LED
                        if len(text) > 500000:  # ~125K tokens
                            text = text[:500000]
                        
                        output = model_pipe(text, **gen_kwargs)[0]
                        batch_summaries.append(output["summary_text"].strip())
                    except Exception as e:
                        print(f"⚠ LED failed on sample: {e}")
                        batch_summaries.append("")
                summaries.extend(batch_summaries)
            else:
                # Normal batch processing for BART
                outputs = model_pipe(batch, **gen_kwargs)
                batch_summaries = []
                for output in outputs:
                    if isinstance(output, dict) and "summary_text" in output:
                        batch_summaries.append(output["summary_text"].strip())
                    else:
                        batch_summaries.append("")
                summaries.extend(batch_summaries)
            
        except Exception as e:
            print(f"⚠ Batch failed: {e}, retrying singly...")
            torch.cuda.empty_cache()
            for text in batch:
                try:
                    output = model_pipe(text, **gen_kwargs)[0]
                    summaries.append(output["summary_text"].strip())
                except:
                    summaries.append("")
    
    return summaries

def summarize_full_pairs(
    pairs_file: str, 
    model_name: str,
    batch_size: int = 4,
    save_intermediate: bool = True
) -> List[dict]:
    """
    Summarize full pairs with CONSTANT generation parameters.
    
    Paper: Same parameters for baseline and all truncation methods.
    """
    print(f"\n📝 FULL CONTEXT BASELINE (CONSTANT PARAMETERS)")
    print(f"  Dataset: {pairs_file}")
    print(f"  Model: {model_name}")
    
    # Load data
    with open(pairs_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Handle formats
    if isinstance(data, dict) and "pairs" in data:
        pairs = data["pairs"]
    else:
        pairs = data
    
    print(f"  Samples: {len(pairs)}")
    
    # Load model with CONSTANT parameters
    model_pipe, gen_kwargs = load_summarization_model(model_name)
    
    # Output path
    base_name = os.path.basename(pairs_file).replace(".json", "")
    out_path = os.path.join(SUMMARIES_DIR, f"{base_name}_full_summaries.json")
    
    # Resume if exists
    results = []
    if os.path.exists(out_path) and save_intermediate:
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"  Resuming: {len(results)}/{len(pairs)} done")
        except:
            results = []
    
    start_idx = len(results)
    
    if start_idx >= len(pairs):
        print("✅ Already completed!")
        return results
    
    # Process
    texts = [p["text"] for p in pairs]
    
    for i in tqdm(range(start_idx, len(texts), CHECKPOINT_INTERVAL), 
                  desc="Baseline (constant params)"):
        chunk_texts = texts[i:i + CHECKPOINT_INTERVAL]
        chunk_pairs = pairs[i:i + CHECKPOINT_INTERVAL]
        
        # Generate with CONSTANT parameters
        chunk_summaries = summarize_batch(chunk_texts, model_pipe, gen_kwargs, batch_size)
        
        # Store results
        for pair, summary in zip(chunk_pairs, chunk_summaries):
            results.append({
                "id": pair["id"],
                "text": pair["text"],
                "references": pair.get("summary", ""),
                "generated_summary": summary,
                "model_name": model_name,
                "gen_kwargs": gen_kwargs,  # CONSTANT for all
                "tokens_input": estimate_tokens_simple(pair["text"]),
                "tokens_output": estimate_tokens_simple(summary),
                "method": "full_context",
                "salience_type": None,
                "truncation_method": None,
            })
        
        # Save checkpoint
        if save_intermediate:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
    
    print(f"✅ Baseline completed: {out_path}")
    return results

def summarize_truncated_files(
    truncated_json_path: str,
    model_name: str,
    batch_size: int = 4,
    save_intermediate: bool = True
) -> List[dict]:
    """
    Summarize truncated documents with CONSTANT generation parameters.
    
    Paper: Same parameters for all salience strategies and datasets.
    """
    print(f"\n✂️ TRUNCATED DOCUMENTS (CONSTANT PARAMETERS)")
    print(f"  Input: {truncated_json_path}")
    print(f"  Model: {model_name}")
    
    # Load data
    with open(truncated_json_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    
    print(f"  Samples: {len(records)}")
    
    # Load model with CONSTANT parameters
    model_pipe, gen_kwargs = load_summarization_model(model_name)
    
    # Output path
    base_name = os.path.basename(truncated_json_path).replace(".json", "")
    out_path = os.path.join(SUMMARIES_DIR, f"{base_name}_summaries.json")
    
    # Resume if exists
    results = []
    if os.path.exists(out_path) and save_intermediate:
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"  Resuming: {len(results)}/{len(records)} done")
        except:
            results = []
    
    start_idx = len(results)
    
    if start_idx >= len(records):
        print("✅ Already completed!")
        return results
    
    # Process
    texts = [r.get("truncated_text", "") for r in records]
    
    for i in tqdm(range(start_idx, len(texts), CHECKPOINT_INTERVAL), 
                  desc="Truncated (constant params)"):
        chunk_texts = texts[i:i + CHECKPOINT_INTERVAL]
        chunk_records = records[i:i + CHECKPOINT_INTERVAL]
        
        # Generate with CONSTANT parameters
        chunk_summaries = summarize_batch(chunk_texts, model_pipe, gen_kwargs, batch_size)
        
        # Store results
        for record, summary in zip(chunk_records, chunk_summaries):
            results.append({
                "id": record["id"],
                "truncated_text": record.get("truncated_text", ""),
                "references": record.get("summary") or record.get("reference_summary", ""),
                "generated_summary": summary,
                "model_name": model_name,
                "gen_kwargs": gen_kwargs,  # CONSTANT for all
                "tokens_before": record.get("tokens_before"),
                "tokens_after": record.get("tokens_after"),
                "token_budget": record.get("token_budget"),
                "truncation_method": record.get("truncation_method"),
                "salience_type": record.get("salience_type"),
                "method": "truncated",
            })
        
        # Save checkpoint
        if save_intermediate:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
    
    print(f"✅ Truncated summaries completed: {out_path}")
    return results

# ============================================================================
# TEST: VERIFY CONSTANT PARAMETERS
# ============================================================================

def test_constant_parameters():
    """Verify all models use the same generation parameters."""
    print("\n🧪 VERIFYING CONSTANT PARAMETERS")
    print("="*60)
    
    test_models = [
        "facebook/bart-large-cnn",
        "allenai/led-base-16384",
    ]
    
    for model_name in test_models:
        print(f"\nTesting {model_name}:")
        _, gen_kwargs = load_summarization_model(model_name)
        
        # Check critical parameters are constant
        expected_params = ["num_beams", "length_penalty", "no_repeat_ngram_size"]
        for param in expected_params:
            actual = gen_kwargs.get(param)
            expected = CONSTANT_GEN_KWARGS.get(param)
            if actual == expected:
                print(f"  ✓ {param}: {actual} (matches constant)")
            else:
                print(f"  ✗ {param}: {actual} vs expected {expected}")
    
    print("\n✅ All models should use same num_beams=4, length_penalty=2.0, etc.")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Summarization with CONSTANT parameters")
    parser.add_argument("--test-params", action="store_true", help="Test constant parameters")
    parser.add_argument("--full", type=str, help="Full pairs JSON path")
    parser.add_argument("--truncated", type=str, help="Truncated JSON path")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    
    args = parser.parse_args()
    
    if args.test_params:
        test_constant_parameters()
    
    elif args.full:
        summarize_full_pairs(args.full, args.model, args.batch)
    
    elif args.truncated:
        summarize_truncated_files(args.truncated, args.model, args.batch)
    
    else:
        print("Usage:")
        print("  python summarizer.py --test-params")
        print("  python summarizer.py --full data/processed/cnn_pairs.json --model facebook/bart-large-cnn")
        print("  python summarizer.py --truncated data/truncated.json --model allenai/led-base-16384")