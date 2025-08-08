from collections import Counter
from tqdm import tqdm

def log_class_distribution(pairs, split_name, class_priority):
    class_count = Counter()
    multi_label_count = 0
    for pair in pairs:
        for c in pair['classes']:
            class_count[c] += 1
        if len(pair['classes']) > 1:
            multi_label_count += 1
    tqdm.write(f"--- Distribusi kelas di {split_name} ---")
    for c in class_priority:
        tqdm.write(f"  class_{c} (ID {c}): {class_count[c]} gambar")
    tqdm.write(f"  Gambar multi-label: {multi_label_count}")
    tqdm.write("")

def read_yolo_classes(label_path):
    classes = set()
    try:
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    classes.add(int(parts[0]))
    except Exception:
        pass
    return list(classes)