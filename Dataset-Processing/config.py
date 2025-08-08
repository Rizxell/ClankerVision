import os

DATASET_ROOT = r'C:\Users\mr250\Documents\SEM8\SkripShit\Calonproject 1\costume-dataset\Dataset-import\coba\Dataset\Dataset_sumber'
DATASET_PATHS = {
    'COCO_Filtered_YOLO_Export': {
        'labels': os.path.join(DATASET_ROOT, 'COCO_Filtered_YOLO_Export/train/labels'),
        'images': os.path.join(DATASET_ROOT, 'COCO_Filtered_YOLO_Export/train/images')
    },
    'labeled-cctv': {
        'labels': os.path.join(DATASET_ROOT, 'labeled-cctv/train/labels'),
        'images': os.path.join(DATASET_ROOT, 'labeled-cctv/train/images')
    },
    'MIO-TCD-Localization': {
        'labels': os.path.join(DATASET_ROOT, 'MIO-TCD-Localization/train/labels'),
        'images': os.path.join(DATASET_ROOT, 'MIO-TCD-Localization/train/images')
    },
    'UA-DETRAC': {
        'labels': os.path.join(DATASET_ROOT, 'UA-DETRAC/labels'),
        'images': os.path.join(DATASET_ROOT, 'UA-DETRAC/images')
    }
}
PRIORITY = ['labeled-cctv', 'MIO-TCD-Localization', 'UA-DETRAC', 'COCO_Filtered_YOLO_Export']
CLASS_PRIORITY = [0, 3, 2, 1] # urutan prioritas class

DATASET_RATIO = {
    'labeled-cctv': 1.0,
    'MIO-TCD-Localization': 0.25,
    'UA-DETRAC': 0.25,
    'COCO_Filtered_YOLO_Export': 0.25
}
OUTPUT_SPLIT = {'train': 0.7, 'val': 0.2, 'test': 0.1}
OUTPUT_DIR = 'output/'
HASH_METHOD = 'phash'
HASH_SIZE = 8
HAMMING_THRESHOLD = 3
IGNORE_AUG_SUFFIXES = ['_aug0', '_aug1', '_aug2']
CACHE_DIR = './cache_split_clean/'
MINORITY_CLASSES = [0,3]  # contoh class minoritas, sesuaikan
RANDOM_SEED = 42