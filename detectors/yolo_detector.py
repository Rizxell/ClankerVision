# === File: detectors/yolo_detector.py ===
from ultralytics import YOLO
import torch

class YOLODetector:
    #def __init__(self, weight_path= r"../Yolo-Weights/yolov8l.pt",
    def __init__(self, weight_path= 'C:\\Users\\mr250\\Documents\\SEM8\\SkripShit\\Calonproject 1\\costume-dataset\\Dataset-import\\coba\\script\\runs\\detect\\Percobaan Ke-10\\weights\\best.pt',
                 device=None, vehicle_classes=None, conf_thres=0.28) : #0.4):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(weight_path).to(device)
        # 1. Ambil peta nama dari model
        self.names = self.model.names 
        # 2. Tentukan ID default jika tidak ada yang diberikan
        default_vehicle_ids = [0,1, 2, 3, 5, 7]
        vehicle_ids = vehicle_classes or default_vehicle_ids
         # 3. Terjemahkan ID menjadi NAMA KELAS dan simpan
        self.vehicle_classes = set([self.names[i] for i in vehicle_ids if i in self.names])
        self.conf_thres = conf_thres

    def detect(self, frame):
        """
        Mendeteksi objek dalam satu frame dan mengembalikan kotak, kelas, dan kepercayaan.
        """
        results = self.model(frame, stream=True, verbose=False) # verbose=False agar tidak print log terus-menerus
        boxes, classes, confidences = [], [], []

        for r in results:
            for b in r.boxes:
                # Dapatkan ID dan confidence
                cls_id = int(b.cls[0])
                conf = float(b.conf[0])

                # --- PERBAIKAN DI SINI ---
                # 1. Terjemahkan ID kelas menjadi NAMA kelas
                class_name = self.names[cls_id]
                
                # 2. Bandingkan NAMA dengan NAMA di dalam filter
                if class_name in self.vehicle_classes and conf >= self.conf_thres:
                    # Jika cocok, baru proses dan simpan hasilnya
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    boxes.append((x1, y1, x2, y2))
                    classes.append(class_name) # <- Saran: Simpan nama kelasnya langsung
                    confidences.append(conf)
                    
        return boxes, classes, confidences