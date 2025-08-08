import numpy as np
from ultralytics.data.dataset import YOLODataset

def detect_split_from_im_files(self):
    # Cek 10 file pertama, cari pattern 'train', 'val', atau 'test' di pathnya
    if len(self.im_files) == 0:
        return None
    for f in self.im_files[:10]:
        fname = f.replace("\\", "/").lower()
        if "train" in fname:
            return "train"
        if "val" in fname:
            return "val"
        if "test" in fname:
            return "test"
    return None

class YOLOWeightedDataset(YOLODataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # DETEKSI SPLIT DARI PATH
        self.split = detect_split_from_im_files(self)
        if self.split is None:
            print("[YOLOWeightedDataset] Tidak dapat mendeteksi split dari file gambar. Menggunakan split 'train' sebagai default.")
            self.split = "train"
        else:
            print(f"[YOLOWeightedDataset] Split terdeteksi: {self.split}")

        self.counts = np.zeros(len(self.data["names"]), dtype=int)
        for label in self.labels:
            cls = label['cls'].reshape(-1).astype(int)
            for cid in cls:
                self.counts[cid] += 1

        smoothing = 0.1
        min_weight = 0.5
        max_weight = 3.0
        counts_smooth = self.counts + smoothing
        class_weights = np.sum(counts_smooth) / (len(self.counts) * counts_smooth)
        class_weights = np.clip(class_weights, min_weight, max_weight)
        self.class_weights = class_weights

        self.weights = []
        for label in self.labels:
            cls = label['cls'].reshape(-1).astype(int)
            if cls.size == 0:
                self.weights.append(1.0)
            else:
                w = np.mean(self.class_weights[cls])
                self.weights.append(w)
        self.weights = np.array(self.weights)
        self.probabilities = self.weights / self.weights.sum()

        print("\n--- [YOLOWeightedDataset LOG] ---")
        print(f"  - Nama Kelas: {self.data['names']}")
        print(f"  - Jumlah Instans per Kelas: {self.counts.tolist()}")
        print(f"  - Bobot Kelas (minoritas lebih tinggi): {np.round(self.class_weights, 2).tolist()}")
        print(f"  - Bobot per Gambar: min={self.weights.min():.2f}, max={self.weights.max():.2f}, mean={self.weights.mean():.2f}")
        print(f"  - Sampling Probabilitas (10 pertama): {np.round(self.probabilities[:10], 4).tolist()}")
        print("--- [LOG END] ---\n")
        self._printed_split_log = False

    def __getitem__(self, index):
        split = getattr(self, "split", None)
        # Log sekali per split
        if not hasattr(self, "_split_logged") or not getattr(self, "_split_logged"):
            print(f"[YOLOWeightedDataset] Weighted sampling {'AKTIF' if split == 'train' else 'NONAKTIF'} (split={split})")
            self._split_logged = True
        if split == "train":
            sampled_index = np.random.choice(len(self.labels), p=self.probabilities)
            return super().__getitem__(sampled_index)
        return super().__getitem__(index)