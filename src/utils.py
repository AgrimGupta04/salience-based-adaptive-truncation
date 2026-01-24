def get_truncated_filename(dataset, truncation_method, salience_type, budget):
    if truncation_method == "salience":
        return f"{dataset}_salience_{salience_type}_budget_{budget}.json"
    return f"{dataset}_{truncation_method}_budget_{budget}.json"
