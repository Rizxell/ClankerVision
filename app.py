import streamlit as st
import numpy as np
import cv2
import time
import json
import os
from stream_handler import open_stream, read_frame
from detectors.yolo_detector import YOLODetector
from trackers.multi_tracker import MultiObjectTracker as KalmanTracker
from trackers.multi_tracker_sort import MultiObjectTracker as SortTracker

STREAM_JSON = "list_stream.json"
TARGET_WIDTH, TARGET_HEIGHT = 640, 360
CLASS_NAMES = ["bus", "car", "motorbike", "truck"]

def load_stream_dict(json_path=STREAM_JSON):
    if not os.path.exists(json_path):
        with open(json_path, "w") as f:
            json.dump({}, f)
    with open(json_path, "r") as f:
        return json.load(f)

def save_stream_dict(data, json_path=STREAM_JSON):
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

def add_stream_entry(name, url, json_path=STREAM_JSON):
    data = load_stream_dict(json_path)
    if name and url:
        data[name] = url
        save_stream_dict(data, json_path)
    return data

CLASS_COLORS = {
    'car': (255, 0, 0),
    'motorbike': (0, 0, 255),
    'truck': (0, 255, 0),
    'bus': (255, 255, 0),
}
COLORS = np.random.randint(0, 255, size=(1000, 3), dtype="uint8")

def remove_duplicate_tracks(objects, iou_threshold=0.5):
    def calculate_iou_for_boxes(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0: return 0.0
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea)
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

st.title("CCTV  Live Video Object Tracking")

# Sidebar menus
st.sidebar.header("Pengaturan Tracking")
USE_TRACKER = st.sidebar.checkbox("Tracker Aktif", value=True)
DRAW_TRACKING_LINE = st.sidebar.checkbox("Garis Tracking Aktif", value=True)
DEBUG_MODE = st.sidebar.checkbox("Debug Mode", value=False)
tracker_mode = st.sidebar.selectbox("Pilih Tracker", ["Hybrid", "SORT"], index=0)
cost_modes = ["iou", "centroid", "hybrid"]
cost_mode = st.sidebar.selectbox("Cost Mode (hanya Kalman)", cost_modes, index=2)
st.sidebar.header("Tambah Stream Baru")
new_name = st.sidebar.text_input("Nama Lokasi Stream")
new_url = st.sidebar.text_input("URL Stream (YouTube/RTSP/FLV/m3u8/File)")
if st.sidebar.button("Tambah Stream"):
    add_stream_entry(new_name, new_url)
    st.sidebar.success(f"Stream '{new_name}' berhasil ditambahkan!")

# Input lokasi (tidak disimpan)
input_link = st.text_input("Masukkan Link Stream")

# Daftar stream (main area)
stream_dict = load_stream_dict()
location_names = list(stream_dict.keys())
selected_name = st.selectbox("Pilih lokasi kamera", location_names)
video_source = input_link if input_link else (stream_dict[selected_name] if selected_name else None)

run_tracking = st.button("Mulai Tracking")

video_placeholder = st.empty()
status_placeholder = st.empty()

if run_tracking and video_source:
    try:
        stream, frame_width, frame_height = open_stream(video_source, TARGET_WIDTH, TARGET_HEIGHT)
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()
    detector = YOLODetector()

    if tracker_mode == "Kalman":
        tracker = KalmanTracker(
            max_disappeared=5,
            max_distance=140,
            class_mismatch_penalty=500,
            iou_cost_scale=70,
            max_area_change_ratio=10,
            frame_width=frame_width,
            frame_height=frame_height,
            cost_mode=cost_mode
        )
    else:
        fps = 30
        try:
            if hasattr(stream, 'get'):
                fps_val = int(stream.get(cv2.CAP_PROP_FPS))
                if fps_val > 0: fps = fps_val
        except Exception:
            pass
        tracker = SortTracker(max_age=1000, min_hits=1, iou_threshold=0.5, fps=fps)

    track_history = {}
    prev_t = time.time()
    frame_count = 0

    while True:
        ok, frame = read_frame(stream, frame_width, frame_height)
        if not ok:
            status_placeholder.warning("Stream selesai atau gagal membaca frame.")
            break
        boxes, classes, confs = detector.detect(frame)
        # Selalu gambar garis tracking (jejak), baik tracker aktif maupun tidak
        if USE_TRACKER:
            if tracker_mode == "Kalman":
                objects = tracker.update(boxes, classes, confs)
                objects = remove_duplicate_tracks(objects)
                for obj_id, (kf, cls, conf) in objects.items():
                    state = kf.kalman.statePost
                    cx, cy, w, h = state[0, 0], state[1, 0], state[2, 0], state[3, 0]
                    x1, y1 = int(cx - w/2), int(cy - h/2)
                    x2, y2 = int(cx + w/2), int(cy + h/2)
                    color = CLASS_COLORS.get(cls, COLORS[obj_id % len(COLORS)].tolist())
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"ID:{obj_id} {cls} {conf:.2f}", (x1, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    # Simpan track history
                    if obj_id not in track_history:
                        track_history[obj_id] = []
                    track_history[obj_id].append((int(cx), int(cy)))
                    if len(track_history[obj_id]) > 50:
                        track_history[obj_id] = track_history[obj_id][-50:]
            else:  # SORT
                tracked = tracker.update(boxes, confs, classes)
                for obj in tracked:
                    x1, y1, x2, y2 = map(int, obj[:4])
                    obj_id = int(obj[4])
                    conf = obj[5]
                    class_id = int(obj[6])
                    cls_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else 'N/A'
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    color = CLASS_COLORS.get(cls_name, COLORS[obj_id % len(COLORS)].tolist())
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"ID:{obj_id} {cls_name} {conf:.2f}", (x1, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    if obj_id not in track_history:
                        track_history[obj_id] = []
                    track_history[obj_id].append((int(cx), int(cy)))
                    if len(track_history[obj_id]) > 50:
                        track_history[obj_id] = track_history[obj_id][-50:]
            if DEBUG_MODE:
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 1)
        else:
            # Tracker nonaktif, hanya gambar bbox hijau + simpan history
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                cls = classes[i]
                conf = confs[i]
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{cls} {conf:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                obj_id = i
                cx, cy = (x1 + x2)//2, (y1 + y2)//2
                if obj_id not in track_history:
                    track_history[obj_id] = []
                track_history[obj_id].append((int(cx), int(cy)))
                if len(track_history[obj_id]) > 50:
                    track_history[obj_id] = track_history[obj_id][-50:]

        # Gambar garis pelacakan (track history) untuk semua objek
        if DRAW_TRACKING_LINE:
            for obj_id, points in track_history.items():
                color = COLORS[obj_id % len(COLORS)].tolist()
                for j in range(1, len(points)):
                    cv2.line(frame, points[j - 1], points[j], color, 2)

        fps = 1 / (time.time() - prev_t + 1e-6)
        prev_t = time.time()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB")
        frame_count += 1

    if isinstance(stream, subprocess.Popen):
        stream.terminate()
        stream.wait()
    else:
        stream.release()