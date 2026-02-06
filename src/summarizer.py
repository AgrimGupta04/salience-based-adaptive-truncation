import os 
import torch 
from transformers import(
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline
)
import json 
from typing import List, Dict, Optional, Any
from tqdm import tqdm

import warnings
import tiktoken
import huggingface_hub

## Suppress warnings
warnings.filterwarnings("ignore", message="Both `max_new_tokens` \\(=.*\\) and `max_length` \\(=.*\\) seem to have been set")
warnings.filterwarnings("ignore", message="Your max_length is set to .*, but your input_length is only")
warnings.filterwarnings("ignore", message="The following generation flags are not valid and may be ignored:.*")

DEFAULT_MODEL = "facebook/bart-large-cnn"
SUMMARIES_DIR = "data/processed/summaries/"
PAIRS_DIR = "data/processed/"

## CHECKPOINT SETTING: Saving progress every 20 documents
CHECKPOINT_INTERVAL = 20 

# enc = tiktoken.get_encoding("cl100k_base")

OFFLOAD_DIR = "/content/offload"
os.makedirs(OFFLOAD_DIR, exist_ok=True)

# def estimate_tokens(text: str) -> int:
#     return len(enc.encode(text))

def _load_model_with_offload(model_name: str, torch_dtype):
    print(f"Attempting complex load with offload for {model_name}...")
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            offload_folder=OFFLOAD_DIR,
            offload_state_dict=True,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        return model
    except Exception as e:
        raise RuntimeError(f"Offload load failed for {model_name}: {e}")

def load_summarization_model(model_name: Optional[str] = None, device: Optional[int] = None, max_input_tokens: int = 2048):
    """Load tokenizer + model and returns a transformer pipeline for summarization."""
    if model_name is None:
        if max_input_tokens < 1024:
            model_name = "facebook/bart-large-cnn"
        elif max_input_tokens < 8000:
            model_name = "google/pegasus-large"
        else:
            model_name = "allenai/led-base-16384"
        print(f"Auto-selected model: {model_name}")

    if device is None:
        device_idx = 0 if torch.cuda.is_available() else -1
    else:
        device_idx = device

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    use_offload = "led-" in model_name.lower() or "long-t5" in model_name.lower()
    
    if "led-" in model_name.lower():
        tokenizer.model_max_length = 16384

    if use_offload:
        try:
            model = _load_model_with_offload(model_name, torch_dtype=torch_dtype)
            print(f"Loaded {model_name} with device_map='auto' and offload.")
        except Exception as e:
            raise RuntimeError(f"Failed to load long model: {e}. Please use a smaller model.")
    else:
        print(f"Loading standard model {model_name} directly to device {device_idx}...")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        if device_idx != -1:
            model.to(torch.device(device_idx))
        print(f"Loaded {model_name} directly.")

    if use_offload:
        pipe = pipeline("summarization", model=model, tokenizer=tokenizer)
    else:
        pipe = pipeline("summarization", model=model, tokenizer=tokenizer, device=device_idx)
    return pipe

def summarize_batch(texts: List[str], model_pipe, batch_size: int = 16, **gen_kwargs) -> List[str]:
    """Summarizes a list of texts (chunks or documents) in batches."""
    summaries = []
    iterator = range(0, len(texts), batch_size)
    
    for i in iterator:
        batch = texts[i: i + batch_size]
        try:
            outs = model_pipe(batch, batch_size=batch_size, **gen_kwargs) 
        except RuntimeError as e:
            print("GPU OOM — retrying with batch_size=1")
            torch.cuda.empty_cache()
            outs = [model_pipe(t, **gen_kwargs)[0] for t in batch]
        for o in outs:
            summaries.append(o["summary_text"].strip())
    return summaries

def summarize_truncated_files(truncated_json_path: str, out_name: Optional[str] = None, model_pipe = None, batch_size: int = 16, gen_kwargs: Optional[Dict[str, Any]] = None, force: bool = False) -> List[dict]:
    """Summarize records with Checkpointing support."""
    if gen_kwargs is None:
        gen_kwargs = {"max_length": 256, "min_length": 64, "num_beams": 4, "early_stopping": True, "do_sample": False, "no_repeat_ngram_size": 3} 

    os.makedirs(SUMMARIES_DIR, exist_ok=True)
    if out_name is None:
        base = os.path.basename(truncated_json_path).replace(".json", "")
        out_name = f"{base}"
    out_path = os.path.join(SUMMARIES_DIR, f"{out_name}.json")

    ## Load Input Data
    with open(truncated_json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    ## Checking for Existing Progress (Checkpointing)
    results = []
    if os.path.exists(out_path) and not force:
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"[checkpoint] Found existing file with {len(results)} summaries. Resuming...")
        except json.JSONDecodeError:
            print("[checkpoint] Existing file was corrupt. Starting from scratch.")
            results = []
    elif force and os.path.exists(out_path):
        print(f"[Force] Overwriting existing output: {out_path}")
        results = []

    ## Determine remaining work
    start_index = len(results)
    if start_index >= len(records):
        print(f"[checkpoint] All {len(records)} records already processed. Skipping.")
        return results

    records_to_process = records[start_index:]
    
    if model_pipe is None:
        model_pipe = load_summarization_model(
            model_name="allenai/led-base-16384",
            max_input_tokens=16384
        )

    is_led_model = "led" in model_pipe.model.config._name_or_path.lower() or "long" in model_pipe.model.config._name_or_path.lower()
    if is_led_model:
        batch_size = min(batch_size, 2)    ## For LED safety 

    print(f"[summarize] Starting processing from index {start_index} to {len(records)}...")

    pbar = tqdm(total=len(records_to_process), desc="Summarizing with Checkpoints")
    
    for i in range(0, len(records_to_process), CHECKPOINT_INTERVAL):
        chunk_recs = records_to_process[i : i + CHECKPOINT_INTERVAL]
        chunk_texts = [r["truncated_text"] for r in chunk_recs]

        chunk_summaries = summarize_batch(chunk_texts, model_pipe, batch_size, **gen_kwargs)

        for rec, gen in zip(chunk_recs, chunk_summaries):
            result = {
                "id": rec["id"],
                "truncated_text": rec["truncated_text"],
                "references": rec.get("summary") or rec.get("gold_summary") or rec.get("references") or "",
                "generated_summary": gen,
                "tokens_before": rec.get("tokens_before"),
                "tokens_after": rec.get("tokens_after"),
                "model_name": model_pipe.model.config._name_or_path
            }
            
            assert result["references"].strip() != "", \
                f"Empty reference summary for record {rec['id']}"

            results.append(result)
        
        ## Saving CHECKPOINT to disk
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        
        pbar.update(len(chunk_recs))

    pbar.close()
    print(f"Saved complete results ({len(results)} summaries) to {out_path}")
    return results
    
def summarize_full_pairs(pairs_file: str, out_name: Optional[str] = None, model_pipe = None, batch_size: int = 16, gen_kwargs: Optional[Dict[str, Any]] = None, force: bool = False) -> List[dict]:
    """Summarize full pairs with Checkpointing support."""
    if gen_kwargs is None:
        gen_kwargs = {"max_length": 256, "min_length": 64, "num_beams": 4, "early_stopping": True, "do_sample": False, "no_repeat_ngram_size": 3}

    os.makedirs(SUMMARIES_DIR, exist_ok=True)
    if out_name is None:
        base = os.path.basename(pairs_file).replace(".json", "")
        out_name = f"{base}_full_summaries"
    out_path = os.path.join(SUMMARIES_DIR, f"{out_name}.json")

    with open(pairs_file, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    results = []
    if os.path.exists(out_path) and not force:
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"[checkpoint] Found existing file with {len(results)} summaries. Resuming...")
        except json.JSONDecodeError:
            results = []
    elif force and os.path.exists(out_path):
        print(f"[Force] Overwriting existing output: {out_path}")
        results = []

    start_index = len(results)
    if start_index >= len(pairs):
        print(f"[checkpoint] All {len(pairs)} records already processed. Skipping.")
        return results

    pairs_to_process = pairs[start_index:]

    if model_pipe is None:
        model_pipe = load_summarization_model()

    print(f"[summarize] Starting processing from index {start_index} to {len(pairs)}...")

    pbar = tqdm(total=len(pairs_to_process), desc="Summarizing Baseline with Checkpoints")

    for i in range(0, len(pairs_to_process), CHECKPOINT_INTERVAL):
        chunk_pairs = pairs_to_process[i : i + CHECKPOINT_INTERVAL]
        chunk_texts = [p["text"] for p in chunk_pairs]

        chunk_summaries = summarize_batch(chunk_texts, model_pipe, batch_size, **gen_kwargs)

        for p, gen in zip(chunk_pairs, chunk_summaries):
            results.append({
                "id": p["id"],
                "text": p["text"],
                "references": p.get("summary", ""),
                "generated_summary": gen
            })

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        
        pbar.update(len(chunk_pairs))

    pbar.close()
    print(f"Saved complete results ({len(results)} full-context summaries) to {out_path}")
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--truncated", type = str, help = "Path to truncated JSON to summarize")
    parser.add_argument("--pairs", type=str, help="Path to full pairs JSON to summarize (baseline)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    pipe = None
    if args.truncated:
        summarize_truncated_files(args.truncated, model_pipe=None, batch_size=args.batch_size)
    if args.pairs:
        pipe = load_summarization_model(args.model)
        summarize_full_pairs(args.pairs, model_pipe=pipe, batch_size=args.batch_size)