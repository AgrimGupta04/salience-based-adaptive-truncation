MODEL_PRICING = {
    # prices per 1M tokens (example, adjustable)
    "facebook/bart-large-cnn": {
        "input": 0.15,
        "output": 0.15,
    },
    "google/long-t5-local-base": {
        "input": 0.10,
        "output": 0.10,
    },
    "allenai/led-base-16384": {
        "input": 0.20,
        "output": 0.20,
    },
    # Optional: simulated proprietary models
    "gpt-4o": {
        "input": 5.00,
        "output": 15.00,
    },
    "claude-3": {
        "input": 8.00,
        "output": 24.00,
    }
}

def estimate_cost(tokens_in, tokens_out, model_name):
    pricing = MODEL_PRICING.get(model_name)
    if pricing is None:
        return None

    return (
        (tokens_in / 1e6) * pricing["input"] +
        (tokens_out / 1e6) * pricing["output"]
    )

import pandas as pd

def add_cost_columns(df):
    """
    Add cost-related columns to the dataframe based on token counts.
    Using OpenAI GPT-4 pricing as default:
    - Input: $10 per 1M tokens
    - Output: $30 per 1M tokens
    """
    # Ensure we have numeric columns
    if 'avg_tokens_before' not in df.columns or 'avg_tokens_after' not in df.columns:
        return df
    
    # Calculate costs (assuming all tokens are input tokens for simplicity)
    input_cost_per_token = 10 / 1_000_000  # $10 per 1M tokens
    output_cost_per_token = 30 / 1_000_000  # $30 per 1M tokens
    
    # Calculate costs
    df['cost_before_usd'] = df['avg_tokens_before'] * input_cost_per_token
    df['cost_after_usd'] = df['avg_tokens_after'] * input_cost_per_token
    df['real_cost_savings_usd'] = df['cost_before_usd'] - df['cost_after_usd']
    df['real_cost_usd'] = df['cost_after_usd']
    
    return df

def calculate_cost_savings(original_tokens, reduced_tokens):
    """Calculate cost savings based on token reduction"""
    input_cost_per_token = 10 / 1_000_000
    original_cost = original_tokens * input_cost_per_token
    reduced_cost = reduced_tokens * input_cost_per_token
    return original_cost - reduced_cost