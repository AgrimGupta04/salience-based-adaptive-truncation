COST_PER_1K = {
    "facebook/bart-large-cnn": 0.0005, 
    "allenai/led-base-16384": 0.001,
    "allenai/led-large-16384-arxiv": 0.002,
    "proxy": 0.01  
}

def estimate_cost(tokens_in, tokens_out, model_name="proxy"):
    if tokens_in is None:
        return None
    rate = COST_PER_1K.get(model_name, 0.01)
    return (tokens_in / 1000) * rate