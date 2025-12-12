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

## Suppress the redundant max_new_tokens/max_length warnings that were flooding the output
warnings.filterwarnings(
    "ignore", 
    message="Both `max_new_tokens` \\(=.*\\) and `max_length` \\(=.*\\) seem to have been set"
)
warnings.filterwarnings(
    "ignore",
    message="Your max_length is set to .*, but your input_length is only"
)

DEFAULT_MODEL = "facebook/bart-large-cnn"
SUMMARIES_DIR = "data/processed/summaries/"
PAIRS_DIR = "data/processed/"

enc = tiktoken.get_encoding("cl100k_base")

OFFLOAD_DIR = "/content/offload"
os.makedirs(OFFLOAD_DIR, exist_ok=True)

def get_device():
    return 0 if torch.cuda.is_available() else -1

def estimate_tokens(text: str) -> int:
    return len(enc.encode(text))

def _load_model_with_offload(model_name: str, torch_dtype):
    """
    Attempt to load model with device_map='auto' and offloading (for Long-T5).
    This helps avoid OOMs for very large long-input models on Colab.
    """
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
        ## If the above fails, re-raise so caller can fallback to a lighter model
        raise RuntimeError(f"Offload load failed for {model_name}: {e}")

def load_summarization_model(model_name: Optional[str] = None, device: Optional[int] = None, max_input_tokens: int = 2048):
    """Load tokenizer + model and returns a transformer pipeline for summarization.
    
    Args:
        model_name: HF model id (We have default set to BART large CNN)
        device: None -> GPU if available else CPU, int -> specific GPU id, -1 -> CPU
        
    Returns: Pipeline object callable like pipeline(texts)
    """

    if model_name is None:
        if max_input_tokens < 1024:
            model_name = "sshleifer/distilbart-cnn-12-6"
        elif max_input_tokens < 8000:
            model_name = "google/pegasus-large"
        else:
            model_name = "google/long-t5-tglobal-base"
        
        print(f"🔄 Auto-selected model: {model_name}")

    if device is None:
        device_idx = 0 if torch.cuda.is_available() else -1
    else:
        device_idx = device

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code = True)

    if "led-" in model_name.lower() or "long-t5" in model_name.lower():
        try:
            model = _load_model_with_offload(model_name, torch_dtype=torch_dtype)
            print(f"✅ Loaded {model_name} with device_map='auto' and offload.")
        except Exception as e:
            # bubble up a clear error so caller can choose an alternative
            raise RuntimeError(f"Failed to load long model with offload: {e}")

    else:
        # Normal model load (BART, PEGASUS, etc.)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        if device_idx != -1:
            model.to(torch.device(device_idx))
        print(f"✅ Loaded {model_name} directly.")

    pipe = pipeline("summarization", model = model, tokenizer = tokenizer, device = device_idx)
    return pipe

def generate_summaries(text: str, model_pipe, max_length: int = 150, min_length: int = 50, do_sample: bool = False) -> str:
    """Summarize a single text using the provided pipeline (model + tokenizer).
    
    Returns: The summary text.
    """

    out = model_pipe(       ## This pipeline accepts a single string or list ensuring string input
        text, 
        max_length = max_length,
        min_length = min_length,
        do_sample = do_sample,
        truncation = True
    )

    return out[0]["summary_text"].strip()       ## Returning the summary string removing any leading or trailing whitespaces

def summarize_batch(texts: List[str], model_pipe, batch_size: int = 32, **gen_kwargs) -> List[str]:
    """Summarizes a list of texts (chunks or documents) in batches.

    Args:
        texts: List of input texts to summarize.
        model_pipe: The summarization pipeline (from load_summarization_model).
        batch_size: Number of texts to summarize in a single batch.
        gen_kwargs: Additional generation arguments for the pipeline.
        
    Returns: List of summary strings
    """

    summaries = []
    for i in tqdm(range(0, len(texts), batch_size), desc = "Summarizing batches"):      ## Stepped range for batching
        batch = texts[i: i + batch_size]
        try:
            outs = model_pipe(batch, truncation=True, **gen_kwargs)

        except RuntimeError as e:
            # OOM recovery
            print("GPU OOM — retrying with batch_size=1")
            torch.cuda.empty_cache()
            outs = [model_pipe(t, truncation=True, **gen_kwargs)[0] for t in batch]       ## outs is a list of dicts when input list -> flatten accordingly
        
        for o in outs:
            summaries.append(o["summary_text"].strip())
    return summaries

def summarize_truncated_files(truncated_json_path: str, out_name: Optional[str] = None, model_pipe = None, batch_size: int = 32, gen_kwargs: Optional[Dict[str, Any]] = None) -> List[dict]:
    """Summarize records in a truncated JSON file and save the outputs.
    Summarize the 'truncated_text' field in the JSON file containing truncated documents.
    Main approach for the paper (versus full-document summarization as baseline).

    Args:
        truncated_json_path: Path to the JSON file with truncated texts from truncartion.py.
        out_name: Optional name for the output summary file. If None, uses the input file name with '_summaries' suffix.
        model_pipe: HF model id for summarization. If None, uses DEFAULT_MODEL.
        batch_size: Number of texts to summarize in a single batch.
        gen_kwargs: Additional generation arguments for the pipeline.
        

        Truncated JSON file format:
        {
        "id": "123",
        "truncated_text": "top salient chunks concatenated",
        "summary": "reference summary",
        "tokens_before": 780,
        "tokens_after": 240,
        "kept_chunk_count": 3
        }

    Returns: List of records with generated summaries
    """

    if gen_kwargs is None:
        gen_kwargs = {"max_length": 64, "min_length": 16, "num_beams": 1, "do_sample": False}

    with open(truncated_json_path, "r", encoding = "utf-8") as f:
        records = json.load(f)

    texts = [r["truncated_text"] for r in records]      ## Extracting texts to summarize

    if model_pipe is None:
        max_tok = max(estimate_tokens(t) for t in texts)
        model_pipe = load_summarization_model(max_input_tokens=max_tok)

    generated_summaries = summarize_batch(texts, model_pipe, batch_size, **gen_kwargs)      ## Summarizing in batches using the HF model pipeline
    
    results = []
    for rec, gen in zip(records, generated_summaries):
        results.append({
            "id": rec["id"],
            "truncated_text": rec["truncated_text"],
            "references": rec.get("summary", ""),
            "generated_summary": gen,
            "tokens_before": rec.get("tokens_before"),
            "tokens_after": rec.get("tokens_after")
        })

    os.makedirs(SUMMARIES_DIR, exist_ok = True)
    if out_name is None:
        base = os.path.basename(truncated_json_path).replace(".json", "")
        out_name = f"{base}"

    out_path = os.path.join(SUMMARIES_DIR, f"{out_name}.json")
    # Save summaries to file
    with open(out_path, "w", encoding = "utf-8") as f:
        json.dump(results, f, indent = 2)
    print(f"Saved {len(results)} summaries to {out_path}")
    return results
    
def summarize_full_pairs(pairs_file: str, out_name: Optional[str] = None, model_pipe = None, batch_size: int = 32, gen_kwargs: Optional[Dict[str, Any]] = None) -> List[dict]:
    """Summarize the 'text' field frim a pairs JSON file (full-context-baseline).
    Run the summarizer on full documents -> baseline, for comaprision with the 
    summary generated from the truncated text(summarize_truncated_files) which is the main approach.
    
    Args:
        pairs_file: Path to the JSON file with text-summary pairs from data_loader.prepare_data().
        out_name: Optional name for the output summary file. If None, uses the input file name with '_full_summaries' suffix.
        model_pipe: HF model id for summarization. If None, uses DEFAULT_MODEL.
        batch_size: Number of texts to summarize in a single batch.
        gen_kwargs: Additional generation arguments for the pipeline.

        Pairs JSON file format:     Untouched full article we downloaded from dataset
        {
        "id": "123",
        "text": "Full document text",
        "summary": "reference highlights"
        }

    Returns: List of records with generated summaries
    """

    if gen_kwargs is None:
        gen_kwargs = {"max_length": 256, "min_length": 50, "num_beams": 1, "do_sample": False}

    if model_pipe is None:
        model_pipe = load_summarization_model()

    with open(pairs_file, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    
    texts = [p["text"] for p in pairs]      ## Extracting full texts to summarize
    generated_summaries = summarize_batch(texts, model_pipe, batch_size, **gen_kwargs)

    results = []
    for p, gen in zip(pairs, generated_summaries):
        results.append({
            "id": p["id"],
            "text": p["text"],
            "references": p.get("summary", ""),
            "generated_summary": gen
        })

    os.makedirs(SUMMARIES_DIR, exist_ok = True)
    if out_name is None:
        base = os.path.basename(pairs_file).replace(".json", "")
        out_name = f"{base}_full_summaries"
    out_path = os.path.join(SUMMARIES_DIR, f"{out_name}.json")
    with open(out_path, "w", encoding = "utf-8") as f:
        json.dump(results, f, indent = 2)

    print(f"Saved {len(results)} full-context summaries to {out_path}")
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--truncated", type = str, help = "Path to truncated JSON to summarize")
    parser.add_argument("--pairs", type=str, help="Path to full pairs JSON to summarize (baseline)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    pipe = load_summarization_model(args.model)
    if args.truncated:
        summarize_truncated_files(args.truncated, model_pipe=pipe, batch_size=args.batch_size)
    if args.pairs:
        summarize_full_pairs(args.pairs, model_pipe=pipe, batch_size=args.batch_size)
