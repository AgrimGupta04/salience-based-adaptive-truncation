# Pricing rates as of March 2026 (conservative proxy estimates)
# Actual HuggingFace Inference API / cloud provider rates may differ
# proxy=$0.01 retained as paper's primary reporting baseline for comparability
COST_PER_1K = {
    "facebook/bart-large-cnn": 0.0005,  # smaller model, cheaper
    "allenai/led-base-16384": 0.001,
    "allenai/led-large-16384-arxiv": 0.002,
    "proxy": 0.01  # keep as conservative upper bound
}

def estimate_cost(tokens_in, tokens_out, model_name="proxy"):
    if tokens_in is None:
        return None
    rate = COST_PER_1K.get(model_name, 0.01)
    return (tokens_in / 1000) * rate