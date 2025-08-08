import imagehash
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
import os

def remove_similar_images(pairs, hash_method, hash_size, hamming_distance_threshold, ignore_aug_suffixes, class_priority):
    tqdm.write("[SIMILARITY] Hashing and grouping images ...")
    hasher = imagehash.phash if hash_method == 'phash' else imagehash.average_hash
    ignore_aug_suffixes_lower = [s.lower() for s in ignore_aug_suffixes]
    hash_to_pairs = defaultdict(list)
    for pair in tqdm(pairs, desc="[SIMILARITY] Hashing Images"):
        f = os.path.basename(pair['image'])
        base_name_no_ext = os.path.splitext(f)[0]
        if any(base_name_no_ext.lower().endswith(sfx) for sfx in ignore_aug_suffixes_lower):
            continue
        try:
            with Image.open(pair['image']) as img:
                if img.mode != 'RGB': img = img.convert('RGB')
                p_hash = str(hasher(img, hash_size=hash_size))
                hash_to_pairs[p_hash].append(pair)
        except Exception:
            continue
    tqdm.write("[SIMILARITY] Filtering duplicates ...")
    keep_set = set()
    remove_set = set()
    for hashval, group in tqdm(hash_to_pairs.items(), desc="[SIMILARITY] Checking Groups"):
        if len(group) == 1:
            keep_set.add(group[0]['image'])
            keep_set.add(group[0]['label'])
        else:
            group_priority = [
                (min([cp for cp in class_priority if cp in pair['classes']] + [999]), pair)
                for pair in group
            ]
            group_priority.sort(key=lambda x: class_priority.index(x[0]) if x[0] in class_priority else len(class_priority))
            keep_set.add(group_priority[0][1]['image'])
            keep_set.add(group_priority[0][1]['label'])
            for prio, pair in group_priority[1:]:
                remove_set.add(pair['image'])
                remove_set.add(pair['label'])
    cleaned_pairs = [pair for pair in pairs if pair['image'] in keep_set]
    tqdm.write(f"[SIMILARITY] Found and removed {len(remove_set)//2} identical images.")
    return cleaned_pairs

def has_minority_object(pair, minority_classes):
    return any(cls in minority_classes for cls in pair['classes'])

def filter_pairs_minority(pairs, minority_classes):
    return [p for p in pairs if has_minority_object(p, minority_classes)]