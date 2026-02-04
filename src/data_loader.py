"""
data_loader.py - FIXED VERSION

Handles loading and preprocessing of datasets for Salience-based Adaptive Truncation.
Implements sentence-aligned chunking as described in the paper.

Author: Agrim Gupta
Updated for EMNLP 2023 submission
"""

from typing import List, Dict, Any, Tuple, Optional
from datasets import load_from_disk
from tqdm import tqdm
import os
import nltk
import re

# Download NLTK data if not present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

DATA_PATH = "data/raw/"
PROCESSED_PATH = "data/processed/"
os.makedirs(PROCESSED_PATH, exist_ok=True)

# Dataset-specific configurations (from paper)
DATASET_CONFIGS = {
    "cnn_dailymail": {
        "chunk_size_tokens": 512,     # Paper: 512 tokens for CNN
        "max_tokens": 1024,           # BART max input
        "model_type": "bart",         # Use BART tokenizer
    },
    "govreport": {
        "chunk_size_tokens": 1024,    # Paper: 1024 tokens for GovReport
        "max_tokens": 16384,          # LED max input
        "model_type": "led",          # Use LED tokenizer
    },
    "arxiv": {
        "chunk_size_tokens": 1024,    # Paper: 1024 tokens for ArXiv
        "max_tokens": 16384,          # LED max input
        "model_type": "led",          # Use LED tokenizer
    }
}

def clean_text(text: str) -> str:
    """Clean text by removing excessive whitespace and normalizing."""
    text = str(text)
    # Replace multiple newlines/tabs with single space
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text

def extract_text(sample: Dict[str, Any], input_field: str) -> str:
    """
    Extracts the document text for a dataset sample.
    """
    if input_field not in sample:
        raise KeyError(
            f"[extract_text] Field '{input_field}' not found in sample.\n"
            f"Available fields: {list(sample.keys())}"
        )

    text = sample[input_field]

    # Handle different data types
    if isinstance(text, list):
        text = " ".join([str(t) for t in text])
    elif isinstance(text, dict):
        text = " ".join([str(v) for v in text.values()])

    return clean_text(text)

def extract_summary(sample: Dict[str, Any], summary_field: str) -> str:
    """
    Extracts the summary text.
    """
    if summary_field not in sample:
        raise KeyError(
            f"[extract_summary] Summary field '{summary_field}' not found.\n"
            f"Available fields: {list(sample.keys())}"
        )

    summary = sample[summary_field]

    # Handle nested summary structures
    if isinstance(summary, dict):
        # Try common keys
        for key in ["summary", "summary_text", "abstract", "highlights"]:
            if key in summary:
                summary = summary[key]
                break

    return clean_text(str(summary))

def load_dataset(dataset_name: str):
    """Loads the dataset from disk by a given name."""
    path = os.path.join(DATA_PATH, dataset_name)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset {dataset_name} not found at {path}. "
            f"Please run download_datasets.py first."
        )
    
    ds = load_from_disk(path)
    print(f"✓ Loaded dataset {dataset_name} from {path}.")
    print(f"  Size: {len(ds)} samples")
    return ds

def get_tokenizer(model_type: str):
    """Get appropriate tokenizer for model type."""
    from transformers import AutoTokenizer
    if model_type == "bart":
        return AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    elif model_type == "led":
        return AutoTokenizer.from_pretrained("allenai/led-base-16384")
    else:
        return AutoTokenizer.from_pretrained("facebook/bart-large-cnn")  # default

def chunk_text_sentence_aligned(
    text: str, 
    tokenizer, 
    max_tokens: int = 512,
    dataset_name: str = ""
) -> Tuple[List[str], List[int]]:
    """
    Splits text into sentence-aligned chunks (as described in paper).
    
    Args:
        text: Input text
        tokenizer: Model-specific tokenizer
        max_tokens: Maximum tokens per chunk
        dataset_name: For logging
    
    Returns:
        Tuple of (chunks, chunk_token_counts)
    """
    # Split into sentences
    sentences = nltk.sent_tokenize(text)
    
    chunks = []
    chunk_token_counts = []
    current_chunk = []
    current_token_count = 0
    
    for i, sentence in enumerate(sentences):
        # Tokenize sentence
        sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
        sentence_token_count = len(sentence_tokens)
        
        # Check if adding this sentence would exceed max_tokens
        if current_token_count + sentence_token_count > max_tokens and current_chunk:
            # Save current chunk
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)
            chunk_token_counts.append(current_token_count)
            
            # Start new chunk
            current_chunk = [sentence]
            current_token_count = sentence_token_count
        else:
            # Add to current chunk
            current_chunk.append(sentence)
            current_token_count += sentence_token_count
    
    # Add last chunk if any
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunks.append(chunk_text)
        chunk_token_counts.append(current_token_count)
    
    # Log chunking statistics
    if chunks:
        avg_chunk_tokens = sum(chunk_token_counts) / len(chunk_token_counts)
        print(f"  [{dataset_name}] Chunking stats: {len(chunks)} chunks, "
              f"avg {avg_chunk_tokens:.1f} tokens/chunk, "
              f"max {max(chunk_token_counts)} tokens")
    
    return chunks, chunk_token_counts

def prepare_data(
    dataset, 
    dataset_name: str, 
    input_field: str, 
    summary_field: str, 
    chunk: bool = True,
    save_path: str = None
) -> List[dict]:
    """
    Prepares text-summary pairs with sentence-aligned chunking.
    
    Args:
        dataset: HF Dataset object
        dataset_name: Name of dataset
        input_field: Document field name
        summary_field: Summary field name
        chunk: Whether to chunk documents
        save_path: Optional path to save preprocessed data
    
    Returns:
        List of dicts with id, text, summary
    """
    # Get dataset configuration
    config = DATASET_CONFIGS.get(dataset_name, DATASET_CONFIGS["cnn_dailymail"])
    chunk_size = config["chunk_size_tokens"]
    model_type = config["model_type"]
    
    print(f"\n📊 Preparing {dataset_name}:")
    print(f"  Chunk size: {chunk_size} tokens")
    print(f"  Model type: {model_type}")
    print(f"  Input field: {input_field}")
    print(f"  Summary field: {summary_field}")
    
    # Load appropriate tokenizer
    tokenizer = get_tokenizer(model_type)
    
    pairs = []
    stats = {
        "total_docs": 0,
        "total_chunks": 0,
        "avg_chunks_per_doc": 0,
        "avg_tokens_per_chunk": 0,
        "max_tokens_per_chunk": 0
    }
    
    chunk_token_counts_all = []
    
    for idx, sample in tqdm(enumerate(dataset), total=len(dataset), desc=f"[prepare] {dataset_name}"):
        text = extract_text(sample, input_field)
        summary = extract_summary(sample, summary_field)
        
        base_id = f"{dataset_name}_{idx}"
        
        if chunk:
            # Create sentence-aligned chunks
            chunks, chunk_token_counts = chunk_text_sentence_aligned(
                text, tokenizer, chunk_size, dataset_name
            )
            
            if not chunks:
                # Empty text, skip
                continue
                
            chunk_token_counts_all.extend(chunk_token_counts)
            
            # Create pairs for each chunk
            for i, (chunk_text, token_count) in enumerate(zip(chunks, chunk_token_counts)):
                pairs.append({
                    'id': f"{base_id}_{i}",
                    'text': chunk_text,
                    'summary': summary,
                    'tokens': token_count,
                    'doc_id': base_id,
                    'chunk_idx': i
                })
        else:
            # Use full document (for baseline)
            token_count = len(tokenizer.encode(text, add_special_tokens=False))
            pairs.append({
                "id": base_id,
                "text": text,
                "summary": summary,
                "tokens": token_count,
                "doc_id": base_id,
                "chunk_idx": 0
            })
    
    # Calculate statistics
    if pairs:
        total_docs = len(set([p['doc_id'] for p in pairs]))
        stats = {
            "total_docs": total_docs,
            "total_chunks": len(pairs),
            "avg_chunks_per_doc": len(pairs) / total_docs if total_docs > 0 else 0,
            "avg_tokens_per_chunk": sum(p['tokens'] for p in pairs) / len(pairs) if pairs else 0,
            "max_tokens_per_chunk": max(p['tokens'] for p in pairs) if pairs else 0
        }
    
    print(f"\n✅ Prepared {len(pairs)} text-summary pairs for {dataset_name}")
    print(f"   Total documents: {stats['total_docs']}")
    print(f"   Average chunks per document: {stats['avg_chunks_per_doc']:.2f}")
    print(f"   Average tokens per chunk: {stats['avg_tokens_per_chunk']:.1f}")
    print(f"   Maximum tokens per chunk: {stats['max_tokens_per_chunk']}")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        import json
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({
                "pairs": pairs,
                "stats": stats,
                "config": config
            }, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved preprocessed data to {save_path}")
    
    return pairs

def test_chunking():
    """Test chunking function with sample text."""
    print("\n🧪 Testing chunking function...")
    
    # Sample text
    test_text = """
    Large Language Models (LLMs) have revolutionized natural language processing. 
    They can generate human-like text and perform various tasks. However, they require significant computational resources. 
    This paper explores efficient summarization techniques. We propose adaptive truncation methods to reduce costs.
    Our experiments show promising results across multiple domains.
    """
    
    # Test with BART tokenizer
    from transformers import AutoTokenizer
    bart_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    
    chunks, token_counts = chunk_text_sentence_aligned(
        test_text, bart_tokenizer, max_tokens=50, dataset_name="test"
    )
    
    print(f"  Original text: {len(test_text.split())} words")
    print(f"  Number of chunks: {len(chunks)}")
    for i, (chunk, tokens) in enumerate(zip(chunks, token_counts)):
        print(f"  Chunk {i+1}: {tokens} tokens, preview: {chunk[:80]}...")
    
    return chunks

if __name__ == "__main__":
    # Test with CNN dataset
    print("="*60)
    print("DATA LOADER TEST")
    print("="*60)
    
    # First test chunking
    test_chunking()
    
    print("\n" + "="*60)
    print("LOADING CNN/DAILYMAIL DATASET")
    print("="*60)
    
    try:
        ds = load_dataset("cnn_dailymail")
        sample_pairs = prepare_data(
            ds, 
            dataset_name="cnn_dailymail", 
            input_field="article", 
            summary_field="highlights", 
            chunk=True,
            save_path="data/processed/cnn_dailymail_pairs.json"
        )
        
        if sample_pairs:
            print(f"\n📋 Sample pair:")
            print(f"   ID: {sample_pairs[0]['id']}")
            print(f"   Text preview: {sample_pairs[0]['text'][:150]}...")
            print(f"   Summary preview: {sample_pairs[0]['summary'][:150]}...")
            print(f"   Tokens: {sample_pairs[0]['tokens']}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Make sure you've downloaded the dataset first:")
        print("   python download_datasets.py")