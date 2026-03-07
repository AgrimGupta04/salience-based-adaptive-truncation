def estimate_cost(tokens_in, tokens_out, model_name = "proxy"):
    if tokens_in is None:
        return None

    return (
        (tokens_in / 1000) * 0.01
    )
