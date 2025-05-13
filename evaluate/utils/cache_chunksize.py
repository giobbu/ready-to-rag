import os
import pickle
import hashlib

def check_cache_questions(path_questions:str, nr_of_questions:str) -> str:
    """Generate a unique cache filename based on params."""
    os.makedirs(path_questions, exist_ok=True)
    return os.path.join(path_questions, f"nr_{nr_of_questions}.pkl")

def load_cache_questions(path_questions: str) -> list:
    """Load evaluation results from a cache file."""
    with open(path_questions, "rb") as f:
        return pickle.load(f)

def save_to_cache_questions(path_questions: str, eval_questions: list) -> None:
    """Save evaluation results to a cache file."""
    with open(path_questions, "wb") as f:
        pickle.dump(eval_questions, f)