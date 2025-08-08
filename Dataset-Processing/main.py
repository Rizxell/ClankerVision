import os
import random
from tqdm import tqdm
from config import *
from cache import save_cache, load_cache, is_cache_valid
from log import read_yolo_classes, log_class_distribution
from loader import find_label_image_pairs, get_current_file_state
from cleaner import remove_similar_images, filter_pairs_minority
from splitter import write_split, split_and_sample, balance_by_realcase

def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    CACHE_CHECK_FILE = os.path.join(CACHE_DIR, 'dataset_file_check.pkl')
    CACHE_PAIRS_FILE = os.path.join(CACHE_DIR, 'pairs.pkl')
    CACHE_PAIRS_CLEANED_FILE = os.path.join(CACHE_DIR, 'pairs_cleaned.pkl')
    CACHE_SPLIT_FILE = os.path.join(CACHE_DIR, 'split.pkl')

    random.seed(RANDOM_SEED)

    current_files = get_current_file_state(DATASET_PATHS, DATASET_ROOT)
    use_cache = is_cache_valid(current_files, CACHE_CHECK_FILE)

    if use_cache and os.path.exists(CACHE_PAIRS_FILE):
        all_pairs = load_cache(CACHE_PAIRS_FILE)
        tqdm.write("[CACHE] Loaded image-label pairs from cache.")
    else:
        tqdm.write("[START] Scan dataset pairs ...")
        all_pairs = []
        for name, info in DATASET_PATHS.items():
            all_pairs.extend(find_label_image_pairs(name, info))
        save_cache(all_pairs, CACHE_PAIRS_FILE)
        save_cache(current_files, CACHE_CHECK_FILE)

    for pair in tqdm(all_pairs, desc="[LABEL] Reading label classes"):
        pair['classes'] = read_yolo_classes(pair['label'])

    tqdm.write("\n[LOG] Distribusi class per gambar sebelum similarity ...")
    log_class_distribution(all_pairs, "SEBELUM similarity", CLASS_PRIORITY)

    if use_cache and os.path.exists(CACHE_PAIRS_CLEANED_FILE):
        all_pairs_cleaned = load_cache(CACHE_PAIRS_CLEANED_FILE)
    else:
        all_pairs_cleaned = remove_similar_images(
            all_pairs,
            HASH_METHOD,
            HASH_SIZE,
            HAMMING_THRESHOLD,
            IGNORE_AUG_SUFFIXES,
            CLASS_PRIORITY
        )
        save_cache(all_pairs_cleaned, CACHE_PAIRS_CLEANED_FILE)

    tqdm.write("\n[LOG] Distribusi class per gambar SETELAH similarity ...")
    log_class_distribution(all_pairs_cleaned, "SETELAH similarity", CLASS_PRIORITY)

    # === Seleksi dataset dan filter minoritas ===
    labeled_pairs = [p for p in all_pairs_cleaned if p['source'] == 'labeled-cctv']
    num_labeled = len(labeled_pairs)
    selected_dataset_pairs = {'labeled-cctv': labeled_pairs}
    internet_pairs_all = []
    for ds_name in DATASET_RATIO:
        if ds_name == 'labeled-cctv':
            continue
        ds_pairs = [p for p in all_pairs_cleaned if p['source'] == ds_name]
        ds_pairs_minority = filter_pairs_minority(ds_pairs, MINORITY_CLASSES)
        num_take = int(DATASET_RATIO[ds_name] * num_labeled)
        if len(ds_pairs_minority) > num_take:
            selected = random.sample(ds_pairs_minority, num_take)
        else:
            selected = ds_pairs_minority
            tqdm.write(f"[WARN] Dataset {ds_name} hanya {len(ds_pairs_minority)} < target {num_take}.")
        selected_dataset_pairs[ds_name] = selected
        internet_pairs_all += selected
        tqdm.write(f"[INIT] {ds_name}: {len(selected)} dari {len(ds_pairs_minority)} (target: {num_take})")

    tqdm.write(f"[INIT] Total labeled-cctv: {num_labeled}")
    all_used_pairs = []
    for v in selected_dataset_pairs.values():
        all_used_pairs.extend(v)
    tqdm.write(f"[INIT] Total ALL USED pairs: {len(all_used_pairs)}")

    # Balancing internet pairs berdasarkan realcase
    balanced_internet_pairs = balance_by_realcase(
        labeled_pairs,
        internet_pairs_all,
        CLASS_PRIORITY,
        RANDOM_SEED
    )

    split_data = split_and_sample(
        {'labeled-cctv': labeled_pairs, 'internet': balanced_internet_pairs},
        OUTPUT_SPLIT, {'labeled-cctv': 1.0, 'internet': 1.0}, num_labeled, RANDOM_SEED
    )
    save_cache(split_data, CACHE_SPLIT_FILE)

    for split_name in ['train', 'val', 'test']:
        log_class_distribution(split_data[split_name], split_name, CLASS_PRIORITY)
        write_split(split_data[split_name], split_name, OUTPUT_DIR)

    tqdm.write("\n[FINISH] Split sesuai inisiasi jumlah, balancing realcase, dan cache finished!")

if __name__ == "__main__":
    main()