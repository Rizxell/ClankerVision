import cv2, time, numpy as np, subprocess
from yt_dlp import YoutubeDL
from detectors.yolo_detector import YOLODetector
from trackers.multi_tracker import MultiObjectTracker

# ===================== KONFIGURASI AWAL =====================
VIDEO_SOURCE = "../costume-dataset/dataset/video/JembatanTirtonadi/JembatanTirtonadi_2025-06-10_15-56-28.mp4"
USE_TRACKER = True                   # Status tracker aktif/tidak
DEBUG_MODE = False                   # Mode debug
DRAW_TRACKING_LINE = True            # Opsi menggambar garis pelacakan
TARGET_WIDTH, TARGET_HEIGHT = 640, 360
TARGET_FPS = 30
frame_delay = int(1000 / TARGET_FPS)
is_paused = False
prev_t = 0

# ===================== FUNGSI AMBIL STREAM YOUTUBE =====================
def get_youtube_stream_url(youtube_url):
    """
    Mengambil URL stream dari video YouTube menggunakan yt_dlp.
    """
    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'quiet': True,
        'noplaylist': True,
        'simulate': True,
    }
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=False)
            if 'url' in info_dict:
                print(f"URL stream YouTube ditemukan: {info_dict['url']}")
                return info_dict['url']
            else:
                print(f"Tidak dapat menemukan URL stream untuk {youtube_url}")
                return None
    except Exception as e:
        print(f"Error saat mendapatkan URL YouTube stream: {e}")
        return None

# ===================== FUNGSI BUKA VIDEO/STREAM =====================
def open_stream(source, width=640, height=360):
    """
    Membuka sumber video lokal, RTSP, HTTP, atau YouTube.
    Menggunakan FFmpeg untuk stream eksternal.
    """
    if "youtube.com" in source or "youtu.be" in source:
        print(f"Mendeteksi sumber YouTube: {source}")
        source_url = get_youtube_stream_url(source)
        if not source_url:
            raise Exception(f"Gagal mendapatkan URL stream YouTube dari: {source}")
        source = source_url

    if source.startswith("rtsp://") or source.startswith("http"):
        print(f"Membuka stream eksternal dengan FFmpeg: {source}")
        ffmpeg_cmd = [
            'ffmpeg',
            '-i', source,
            '-loglevel', 'quiet',
            '-an',
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-vf', f'scale={width}:{height}',
            '-'
        ]
        try:
            process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            time.sleep(1)
            if process.poll() is not None and process.returncode != 0:
                stderr_output = process.stderr.read().decode('utf-8')
                raise Exception(f"FFmpeg gagal memulai. Pastikan FFmpeg terinstal dan URL stream valid. Error: {stderr_output}")
            return process, width, height
        except FileNotFoundError:
            raise Exception("FFmpeg tidak ditemukan. Pastikan FFmpeg terinstal dan ada di PATH sistem Anda.")
    else:
        print(f"Membuka file/stream lokal dengan OpenCV: {source}")
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise Exception(f"Gagal membuka video lokal: {source}")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return cap, w, h

# ===================== FUNGSI BACA FRAME =====================
def read_frame(stream, width, height):
    """
    Membaca satu frame dari stream.
    Mendukung stream eksternal (FFmpeg) dan lokal (OpenCV).
    """
    if isinstance(stream, subprocess.Popen):
        raw_frame_size = width * height * 3
        raw_frame = stream.stdout.read(raw_frame_size)
        if not raw_frame or len(raw_frame) != raw_frame_size:
            return False, None
        frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3)).copy()
        return True, frame
    else:
        return stream.read()

# ===================== INISIALISASI DETEKTOR DAN TRACKER =====================
try:
    stream, frame_width, frame_height = open_stream(VIDEO_SOURCE, TARGET_WIDTH, TARGET_HEIGHT)
except Exception as e:
    print(f"Fatal Error: {e}")
    exit()

detector = YOLODetector()
tracker = MultiObjectTracker(
        max_disappeared=5,           # ID bertahan walau hilang 
        max_distance=140,            # jarak maksimum asosiasi deteksi
        class_mismatch_penalty=500,  # penalti class berubah (stabilkan class)
        iou_cost_scale=70,           # skala IOU cost
        max_area_change_ratio=10,    # area bbox tidak boleh berubah drastis
        frame_width=frame_width,
        frame_height=frame_height,
        cost_mode='hybrid'
)

# ===================== DICTIONARY WARNA PER CLASS =====================
# Atur warna unik untuk setiap kelas. Format BGR.
CLASS_COLORS = {
    'car': (255, 0, 0),      # Biru
    'motorbike': (0, 0, 255),     # Merah
    'truck': (0, 255, 0),    # Hijau
    'bus': (255, 255, 0),  # Biru Kuning
    
    # Tambahkan kelas lain sesuai kebutuhan
}

# ===================== GENERATE WARNA TRACKER =====================
# Array warna random untuk fallback jika class tidak ada di CLASS_COLORS
COLORS = np.random.randint(0, 255, size=(1000, 3), dtype="uint8")

# ===================== FUNGSI IOU DAN DUPLIKAT TRACK =====================
def calculate_iou_for_boxes(boxA, boxB):
    """
    Menghitung IOU antara dua bounding box.
    Format box: [x1, y1, x2, y2]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0: return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def remove_duplicate_tracks(objects, iou_threshold=0.5):
    """
    Menghapus track yang duplikat berdasarkan IOU > threshold.
    """
    filtered_objects = {}
    used_ids = set()
    object_items = list(objects.items())

    for i in range(len(object_items)):
        id1, (kf1, cls1, conf1) = object_items[i]
        if id1 in used_ids: continue
        bbox1 = kf1.kalman.statePost[:4].flatten()
        cx1, cy1, w1, h1 = bbox1
        box1 = [cx1 - w1/2, cy1 - h1/2, cx1 + w1/2, cy1 + h1/2]

        for j in range(i+1, len(object_items)):
            id2, (kf2, cls2, conf2) = object_items[j]
            if id2 in used_ids: continue
            bbox2 = kf2.kalman.statePost[:4].flatten()
            cx2, cy2, w2, h2 = bbox2
            box2 = [cx2 - w2/2, cy2 - h2/2, cx2 + w2/2, cy2 + h2/2]
            iou = calculate_iou_for_boxes(box1, box2)
            if iou > iou_threshold:
                used_ids.add(id2)

        filtered_objects[id1] = (kf1, cls1, conf1)
    return filtered_objects

# ===================== TRACK HISTORY UNTUK GARIS PELACAKAN =====================
track_history = {}

print("Tekan Spasi = jeda/lanjut | 't' = tracker ON/OFF | 'l' = garis pelacakan ON/OFF | 'd' = debug ON/OFF | 'q' = keluar")

# ===================== LOOP UTAMA FRAME =====================
while True:
    if not is_paused:
        ok, frame = read_frame(stream, frame_width, frame_height)
        if not ok:
            print("Stream selesai atau gagal membaca frame.")
            break

        # --------- DETEKSI OBJEK DI FRAME ---------
        boxes, classes, confs = detector.detect(frame)

        # --------- JIKA TRACKER AKTIF ---------
        if USE_TRACKER:
            # Jika debug mode, gambar bounding box deteksi awal (hijau tipis)
            if DEBUG_MODE:
                for i in range(len(boxes)):
                    x1_raw, y1_raw, x2_raw, y2_raw = boxes[i]
                    cv2.rectangle(frame, (x1_raw, y1_raw), (x2_raw, y2_raw), (0, 255, 0), 1)
                    cv2.putText(frame, "Deteksi", (x1_raw, y1_raw - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

            # Update tracker dengan hasil deteksi
            objects = tracker.update(boxes, classes, confs)
            objects = remove_duplicate_tracks(objects)

            # Gambar hasil tracking (warna sesuai class atau unik per object)
            for obj_id, (kf, cls, conf) in objects.items():
                # Ambil state bbox dari Kalman filter
                state = kf.kalman.statePost
                cx, cy, w, h = state[0, 0], state[1, 0], state[2, 0], state[3, 0]
                x1, y1 = int(cx - w/2), int(cy - h/2)
                x2, y2 = int(cx + w/2), int(cy + h/2)

                # Pilih warna: jika class terdaftar di CLASS_COLORS pakai warna class, jika tidak pakai random tetap unik per obj_id
                color = CLASS_COLORS.get(cls, COLORS[obj_id % len(COLORS)].tolist())
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID:{obj_id} {cls} {conf:.2f}", (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # --- Update dan gambar garis pelacakan (track history) ---
                if DRAW_TRACKING_LINE:
                    if obj_id not in track_history:
                        track_history[obj_id] = []
                    track_history[obj_id].append((int(cx), int(cy)))
                    if len(track_history[obj_id]) > 50:  # Batas panjang jejak
                        track_history[obj_id] = track_history[obj_id][-50:]
                    for j in range(1, len(track_history[obj_id])):
                        cv2.line(frame, track_history[obj_id][j - 1], track_history[obj_id][j], color, 2)
                else:
                    # Jika garis pelacakan dimatikan, hapus history
                    track_history[obj_id] = []
        else:
            # --------- JIKA TRACKER NONAKTIF (BOX HIJAU) ---------
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                cls = classes[i]
                conf = confs[i]
                # Semua objek diberi bounding box hijau saat tracker nonaktif
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{cls} {conf:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            # Reset semua jejak jika tracker off
            track_history = {}

        # --------- TAMPILKAN INFO FPS DAN STATUS MODE ---------
        fps = 1 / (time.time() - prev_t + 1e-6)
        prev_t = time.time()
        cv2.putText(frame, f"FPS: {int(fps)}", (7, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2)
        cv2.putText(frame, f"Tracker: {'ON' if USE_TRACKER else 'OFF'}", (7, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0) if USE_TRACKER else (0, 0, 255), 2)
        cv2.putText(frame, f"Line: {'ON' if DRAW_TRACKING_LINE else 'OFF'}", (7, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 0) if DRAW_TRACKING_LINE else (128, 128, 128), 2)
        cv2.putText(frame, f"Debug: {'ON' if DEBUG_MODE else 'OFF'}", (7, 120), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 255) if DEBUG_MODE else (128, 128, 128), 2)

    # --------- TAMPILKAN FRAME HASIL TRACKING ---------
    cv2.imshow("Tracking", frame)

    # ===================== KONTROL KEYBOARD =====================
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('t'):
        USE_TRACKER = not USE_TRACKER
        print(f"Tracker sekarang {'diaktifkan' if USE_TRACKER else 'dinonaktifkan'}.")
    elif key == ord('l'):
        DRAW_TRACKING_LINE = not DRAW_TRACKING_LINE
        print(f"Garis pelacakan sekarang {'diaktifkan' if DRAW_TRACKING_LINE else 'dinonaktifkan'}.")
        if not DRAW_TRACKING_LINE:
            track_history = {}
    elif key == ord('d'):
        DEBUG_MODE = not DEBUG_MODE
        print(f"Debug sekarang {'diaktifkan' if DEBUG_MODE else 'dinonaktifkan'}.")
    elif key == ord(' '):
        is_paused = not is_paused
        print("Video dijeda." if is_paused else "Video dilanjutkan.")

# ===================== BERSIHKAN DAN TUTUP =====================
if isinstance(stream, subprocess.Popen):
    stream.terminate()
    stream.wait()
else:
    stream.release()
cv2.destroyAllWindows()