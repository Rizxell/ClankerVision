import os
from glob import glob
from tqdm import tqdm

def find_label_image_pairs(dataset_name, dataset_info):
    pairs = []
    labels_dir = dataset_info['labels']
    images_dir = dataset_info['images']
    if not os.path.exists(labels_dir) or not os.path.exists(images_dir):
        tqdm.write(f"[SCAN] {dataset_name}: Labels or images folder missing, skip.")
        return pairs
    all_images = set(glob(os.path.join(images_dir, "*.jpg"))) | set(glob(os.path.join(images_dir, "*.png")))
    all_labels = set(glob(os.path.join(labels_dir, "*.txt")))
    img_to_label = {}
    for label_path in all_labels:
        base = os.path.splitext(os.path.basename(label_path))[0]
        img_path_jpg = os.path.join(images_dir, f"{base}.jpg")
        img_path_png = os.path.join(images_dir, f"{base}.png")
        img_path = None
        if os.path.exists(img_path_jpg):
            img_path = img_path_jpg
        elif os.path.exists(img_path_png):
            img_path = img_path_png
        if img_path:
            pairs.append({'image': img_path, 'label': label_path, 'source': dataset_name})
            img_to_label[img_path] = label_path
    # UA-DETRAC special handling
    if dataset_name == 'UA-DETRAC':
        label_seqs = [d for d in os.listdir(labels_dir) if os.path.isdir(os.path.join(labels_dir, d))]
        for seq in tqdm(label_seqs, desc=f"[SCAN] UA-DETRAC seqs", leave=False):
            lbl_seq_dir = os.path.join(labels_dir, seq)
            img_seq_dir = os.path.join(images_dir, seq)
            label_files = glob(os.path.join(lbl_seq_dir, "*.txt"))
            for label_path in label_files:
                base = os.path.splitext(os.path.basename(label_path))[0]
                img_path_jpg = os.path.join(img_seq_dir, f"{base}.jpg")
                img_path_png = os.path.join(img_seq_dir, f"{base}.png")
                img_path = None
                if os.path.exists(img_path_jpg):
                    img_path = img_path_jpg
                elif os.path.exists(img_path_png):
                    img_path = img_path_png
                if img_path:
                    pairs.append({'image': img_path, 'label': label_path, 'source': dataset_name})
                    img_to_label[img_path] = label_path
            all_images |= set(glob(os.path.join(img_seq_dir, "*.jpg"))) | set(glob(os.path.join(img_seq_dir, "*.png")))
    # Cek gambar tanpa label
    no_label_images = [img for img in all_images if img not in img_to_label]
    if no_label_images:
        tqdm.write(f"[WARNING] {len(no_label_images)} images in '{dataset_name}' have NO label file!")
        for img in no_label_images[:10]:
            tqdm.write(f"  No label: {img}")
    return pairs

def get_current_file_state(paths, dataset_root):
    files = []
    for ds, info in paths.items():
        img_files = glob(os.path.join(info['images'], '**/*.jpg'), recursive=True) + glob(os.path.join(info['images'], '**/*.png'), recursive=True)
        lbl_files = glob(os.path.join(info['labels'], '**/*.txt'), recursive=True)
        files.extend(img_files)
        files.extend(lbl_files)
    files.sort()
    return [os.path.relpath(f, dataset_root) for f in files]