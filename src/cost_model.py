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
