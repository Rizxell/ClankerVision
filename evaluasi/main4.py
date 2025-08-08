import cv2, time, numpy as np, os, glob
from detectors.yolo_detector import YOLODetector
from trackers.multi_tracker import MultiObjectTracker as KalmanTracker
from trackers.multi_tracker_sort import MultiObjectTracker as SortTracker

# === KONFIGURASI DASAR ===
VIDEO_PATH = "../costume-dataset/Dataset-import/coba/Dataset/Dataset_sumber/UA-DETRAC/images/MVI_20011/output_video.mp4"

MASK_FOLDER = "./final_dataset/Mask"
CLASS_NAMES = ["bus", "car", "motorbike", "truck"]
SAVE_PATH = "./tracking_results_mask.txt"

USE_TRACKER = True
USE_MASK = True
DEBUG_MODE = False
SAVE_TRACKING = True
USE_KALMAN = True  # Jika False, gunakan SORT

COST_MODES = ["iou", "centroid", "hybrid"]
cost_index = 2
COST_MODE = COST_MODES[cost_index]

# === INISIALISASI DASAR ===
cap = cv2.VideoCapture(VIDEO_PATH)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30

detector = YOLODetector()

def reset_tracker_kalman():
    global tracker_kalman
    tracker_kalman = KalmanTracker(
        max_disappeared=5,           # ID bertahan walau hilang 
        max_distance=140,              # jarak maksimum untuk mengasosiasikan deteksi
        class_mismatch_penalty=500,    # penalti class berubah (stabilkan class)
        iou_cost_scale=70,             # skala IOU cost
        max_area_change_ratio=10,       # area bbox tidak boleh berubah drastis
        frame_width=frame_width,
        frame_height=frame_height,
        cost_mode='hybrid'
    )

reset_tracker_kalman()
tracker_sort = SortTracker(max_age=1000, min_hits=1, iou_threshold=0.5, fps=fps)
COLORS = np.random.randint(0, 255, size=(10000, 3), dtype="uint8")

# === MASK ===
mask_paths = sorted(glob.glob(os.path.join(MASK_FOLDER, "*.png")))
if not mask_paths:
    raise FileNotFoundError("Mask tidak ditemukan di folder: " + MASK_FOLDER)
mask_combined = np.zeros((frame_height, frame_width), dtype=np.uint8)
for mpath in mask_paths:
    mask = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Gagal membaca mask: {mpath}")
    if len(mask.shape) == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if len(mask.shape) == 3 and mask.shape[2] == 1:
        mask = mask[:, :, 0]
    if len(mask.shape) == 3 and mask.shape[2] == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.shape != (frame_height, frame_width):
        raise ValueError(f"Ukuran mask tidak cocok! {mask.shape[::-1]} ≠ ({frame_width}, {frame_height})")
    mask_combined = cv2.bitwise_or(mask_combined, mask)
mask_overlay = cv2.cvtColor(mask_combined, cv2.COLOR_GRAY2BGR)
mask_overlay[:, :, 1:] = 0  # channel merah saja

# === LAINNYA ===
prev_t = time.time()
frame_id = 0
is_paused = False
track_file = open(SAVE_PATH, 'w') if SAVE_TRACKING else None

print("Spasi=Jeda | q=Keluar | t=Toggle Tracker | d=Debug | m=Mask | s=Save Tracking | c=Cost Mode")

while True:
    if not is_paused:
        ok, frame = cap.read()
        if not ok:
            print("Video telah selesai.")
            break
        frame_id += 1
        boxes, classes, confs = detector.detect(frame)

        if USE_TRACKER:
            tracker = tracker_kalman if USE_KALMAN else tracker_sort
            if USE_KALMAN:
                tracked = tracker.update(boxes, classes, confs)
                for obj_id, (kf, cls, conf) in tracked.items():
                    state = kf.kalman.statePost
                    cx, cy, w, h = state[0,0], state[1,0], state[2,0], state[3,0]
                    x1, y1, x2, y2 = int(cx-w/2), int(cy-h/2), int(cx+w/2), int(cy+h/2)
                    cx_i, cy_i = (x1+x2)//2, (y1+y2)//2

                    if USE_MASK:
                        if not (0 <= cx_i < frame_width and 0 <= cy_i < frame_height):
                            continue
                        if mask_combined[cy_i, cx_i] == 0:
                            continue

                    # === STABILISASI CLASS: Majority Voting ===
                    major_cls = tracker.get_majority_class(obj_id) if hasattr(tracker, "get_majority_class") else cls
                    # Tampilkan class hasil voting (lebih stabil)
                    cls_name = CLASS_NAMES[major_cls] if isinstance(major_cls, int) and major_cls < len(CLASS_NAMES) else str(major_cls)

                    color = COLORS[obj_id % len(COLORS)].tolist()
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"ID:{obj_id} {cls_name} {conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    if SAVE_TRACKING and track_file:
                        w, h = x2 - x1, y2 - y1
                        class_id = 1  # untuk mobil
                        track_file.write(f"{frame_id},{obj_id},{x1},{y1},{w},{h},1,{class_id},-1\n")

            else:
                tracked = tracker.update(boxes, confs, classes)
                for obj in tracked:
                    x1, y1, x2, y2 = map(int, obj[:4])
                    obj_id = int(obj[4])
                    conf = obj[5]
                    class_id = int(obj[6])
                    cls_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else 'N/A'
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    if USE_MASK:
                        if not (0 <= cx < frame_width and 0 <= cy < frame_height):
                            continue
                        if mask_combined[cy, cx] == 0:
                            continue

                    color = COLORS[obj_id % len(COLORS)].tolist()
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"ID:{obj_id} {cls_name} {conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    if SAVE_TRACKING and track_file:
                        w, h = x2 - x1, y2 - y1
                        class_id = 1  # untuk mobil
                        track_file.write(f"{frame_id},{obj_id},{x1},{y1},{w},{h},1,{class_id},-1\n")

            if DEBUG_MODE:
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 1)

        else:
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                cls = classes[i]
                conf = confs[i]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"{cls} {conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        if USE_MASK and mask_overlay is not None:
            frame = cv2.addWeighted(mask_overlay, 0.3, frame, 0.7, 0)

        fps_disp = 1 / (time.time() - prev_t + 1e-6)
        prev_t = time.time()
        cv2.putText(frame, f"FPS: {int(fps_disp)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100,255,0), 2)
        cv2.putText(frame, f"Tracker: {'Kalman' if USE_KALMAN else 'SORT'}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        cv2.putText(frame, f"Debug: {'ON' if DEBUG_MODE else 'OFF'}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128,128,128), 2)
        cv2.putText(frame, f"Mask: {'ON' if USE_MASK else 'OFF'}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255) if USE_MASK else (128,128,128), 2)
        cv2.putText(frame, f"Save: {'ON' if SAVE_TRACKING else 'OFF'}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, f"Cost: {COST_MODE.upper()}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,200,100), 2)

    cv2.imshow("Tracking", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord(' '): is_paused = not is_paused
    elif key == ord('d'): DEBUG_MODE = not DEBUG_MODE
    elif key == ord('m'): USE_MASK = not USE_MASK
    elif key == ord('s'):
        SAVE_TRACKING = not SAVE_TRACKING
        if SAVE_TRACKING and not track_file:
            track_file = open(SAVE_PATH, 'a')
        elif not SAVE_TRACKING and track_file:
            track_file.close()
            track_file = None
    elif key == ord('t'):
        USE_KALMAN = not USE_KALMAN
        print(f"Tracker sekarang: {'Kalman' if USE_KALMAN else 'SORT'}")
    elif key == ord('c'):
        cost_index = (cost_index + 1) % len(COST_MODES)
        COST_MODE = COST_MODES[cost_index]
        reset_tracker_kalman()
        print(f"Cost mode sekarang: {COST_MODE}")

cap.release()
cv2.destroyAllWindows()
if track_file: track_file.close()