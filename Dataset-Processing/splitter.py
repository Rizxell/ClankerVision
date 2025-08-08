import os
import random
from tqdm import tqdm
from collections import Counter

def write_split(pairs, split_name, out_dir):
    labels_out_dir = os.path.join(out_dir, split_name, 'labels')
    images_out_dir = os.path.join(out_dir, split_name, 'images')
    os.makedirs(labels_out_dir, exist_ok=True)
    os.makedirs(images_out_dir, exist_ok=True)
    for pair in tqdm(pairs, desc=f"[WRITE] {split_name} copying files"):
        label_dst = os.path.join(labels_out_dir, os.path.basename(pair['label']))
        with open(pair['label'], 'r') as fr, open(label_dst, 'w') as fw:
            fw.write(fr.read())
        image_dst = os.path.join(images_out_dir, os.path.basename(pair['image']))
        if not os.path.exists(image_dst):
            with open(pair['image'], 'rb') as fr, open(image_dst, 'wb') as fw:
                fw.write(fr.read())

def split_and_sample(selected_dataset_pairs, output_split, dataset_ratio, num_labeled, random_seed=42):
    random.seed(random_seed)
    # split realcase
    n_realcase = len(selected_dataset_pairs['labeled-cctv'])
    train_n = int(n_realcase * output_split['train'])
    val_n = int(n_realcase * output_split['val'])
    test_n = n_realcase - train_n - val_n
    shuffled = selected_dataset_pairs['labeled-cctv'][:]
    random.shuffle(shuffled)
    train_realcase = shuffled[:train_n]
    val_realcase = shuffled[train_n:train_n+val_n]
    test_realcase = shuffled[train_n+val_n:]
    # other dataset in train only
    if 'internet' in selected_dataset_pairs:
        other_train = selected_dataset_pairs['internet']
    else:
        other_train = []
        for ds_name in dataset_ratio:
            if ds_name == 'labeled-cctv':
                continue
            other_train += selected_dataset_pairs[ds_name]
    train_balanced = train_realcase + other_train
    val_balanced = val_realcase
    test_balanced = test_realcase
    return {'train': train_balanced, 'val': val_balanced, 'test': test_balanced}

def compute_class_distribution(pairs, class_priority):
    class_count = Counter()
    for pair in pairs:
        for c in pair['classes']:
            if c in class_priority:
                class_count[c] += 1
    return class_count

def balance_by_realcase(realcase_pairs, internet_pairs, class_priority, random_seed=42):
    realcase_dist = compute_class_distribution(realcase_pairs, class_priority)
    total_realcase = sum(realcase_dist.values())
    num_samples_per_class = {c: realcase_dist[c] for c in class_priority}
    # Kumpulkan gambar internet per kelas
    class_to_pairs = {c: [] for c in class_priority}
    for pair in internet_pairs:
        for c in pair['classes']:
            if c in class_priority:
                class_to_pairs[c].append(pair)
    # Sampling internet agar proporsi kelas mendekati realcase
    balanced_pairs = []
    random.seed(random_seed)
    for c in class_priority:
        available = class_to_pairs[c]
        take_n = num_samples_per_class[c]
        if len(available) > take_n:
            balanced_pairs += random.sample(available, take_n)
        else:
            balanced_pairs += available
            tqdm.write(f"[WARN] Jumlah gambar kelas {c} pada internet hanya {len(available)} < target {take_n}")
    return balanced_pairs