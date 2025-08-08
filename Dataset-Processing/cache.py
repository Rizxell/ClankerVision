import pickle
import os

def save_cache(obj, fname):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)

def load_cache(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

def is_cache_valid(current_files, cache_file):
    if os.path.exists(cache_file):
        try:
            cached_files = load_cache(cache_file)
            return cached_files == current_files
        except Exception:
            return False
    return False