import os
import pickle
import hashlib

def get_cache_path(name: str, dataset_type: str, top_k: int, finetune: bool) -> str:
    """Generate a unique cache filename based on params."""
    cache_dir = "evaluate/cache"
    os.makedirs(cache_dir, exist_ok=True)
    key = f"{name}_{dataset_type}_top{top_k}_{'finetune' if finetune else 'base'}"
    hashed_key = hashlib.md5(key.encode()).hexdigest()
    return os.path.join(cache_dir, f"{hashed_key}.pkl")

def load_cached_results(path: str) -> list:
    with open(path, "rb") as f:
        return pickle.load(f)

def save_results_to_cache(path: str, eval_results: list) -> None:
    """Save evaluation results to a cache file."""
    with open(path, "wb") as f:
        pickle.dump(eval_results, f)
