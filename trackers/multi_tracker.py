import numpy as np
from .kalman_filter import KalmanFilter
from scipy.optimize import linear_sum_assignment

class MovingAverageFilter:
    def __init__(self, window_size=3):
        self.window_size = window_size
        self.positions = []

    def update(self, position):
        self.positions.append(position)
        if len(self.positions) > self.window_size:
            self.positions.pop(0)
        avg = np.mean(self.positions, axis=0)
        return tuple(avg)

class MultiObjectTracker:
    def __init__(self, max_disappeared=5, max_distance=80,
                 appearance_weight=50, class_mismatch_penalty=1000,
                 frame_width=1920, frame_height=1080,
                 iou_cost_scale=100, max_area_change_ratio=3.0,
                 cost_mode='hybrid'):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.object_history = {}
        self.appearance_features = {}
        self.lost_tracks = {}
        self.lost_track_patience = int(max_disappeared * 1.5)
        self.class_history = {}  # Tambahkan untuk stabilisasi class

        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.appearance_weight = appearance_weight
        self.class_mismatch_penalty = class_mismatch_penalty
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.edge_margin = 50
        self.iou_cost_scale = iou_cost_scale
        self.max_area_change_ratio = max_area_change_ratio
        self.cost_mode = cost_mode
        if self.cost_mode not in ['centroid', 'iou', 'hybrid']:
            raise ValueError("cost_mode harus salah satu dari: 'centroid', 'iou', 'hybrid'")

    def calculate_iou(self, boxA, boxB):
        def to_corners(box):
            cx, cy, w, h = box
            return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
        boxA_corners, boxB_corners = to_corners(boxA), to_corners(boxB)
        xA, yA = max(boxA_corners[0], boxB_corners[0]), max(boxA_corners[1], boxB_corners[1])
        xB, yB = min(boxA_corners[2], boxB_corners[2]), min(boxA_corners[3], boxB_corners[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0
        boxAArea, boxBArea = boxA[2] * boxA[3], boxB[2] * boxB[3]
        return interArea / float(boxAArea + boxBArea - interArea)

    def register(self, box, class_id, confidence):
        self.next_object_id += 1
        self._register_with_id(self.next_object_id - 1, box, class_id, confidence)

    def _register_with_id(self, object_id, box, class_id, confidence):
        kf = KalmanFilter()
        x1, y1, x2, y2 = box
        cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        kf.init(np.array([cx, cy, w, h], np.float32))
        self.objects[object_id] = (kf, class_id, confidence)
        self.disappeared[object_id] = 0
        self.object_history[object_id] = [(cx, cy, w, h)]
        self.appearance_features[object_id] = {
            'aspect_ratio': w/h if h > 0 else 1.0,
            'area': w*h,
            'class_id': class_id
        }
        # Inisialisasi class history
        self.class_history[object_id] = [class_id]
        if object_id >= self.next_object_id:
            self.next_object_id = object_id + 1

    def deregister(self, object_id):
        if object_id in self.objects:
            kf, cls, conf = self.objects[object_id]
            if conf > 0.3:
                self.lost_tracks[object_id] = {
                    'state': kf.kalman.statePost.flatten(),
                    'features': self.appearance_features.get(object_id),
                    'age': 0,
                    'class_id': cls
                }
        for d in [self.objects, self.disappeared, self.object_history, self.appearance_features, self.class_history]:
            d.pop(object_id, None)

    def get_majority_class(self, object_id):
        history = self.class_history.get(object_id, [])
        if not history:
            return None
        return max(set(history), key=history.count)

    def update(self, boxes, classes, confidences):
        if len(boxes) == 0:
            for object_id in list(self.objects.keys()):
                self.deregister(object_id)
            return self.objects

        input_boxes = [
            ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1])
            for b in boxes
        ]
        if len(self.objects) == 0:
            for i in range(len(boxes)):
                self.register(boxes[i], classes[i], confidences[i])
            return self.objects

        object_ids = list(self.objects.keys())
        predicted_states = [self.objects[oid][0].predict() for oid in object_ids]

        D = np.zeros((len(object_ids), len(input_boxes)))
        for i, pred_state in enumerate(predicted_states):
            for j, det_box in enumerate(input_boxes):
                penalty = 0
                if self.appearance_features[object_ids[i]]['class_id'] != classes[j]:
                    penalty += self.class_mismatch_penalty

                # --- COST BBOX APPEARANCE ---
                tracked_features = self.appearance_features[object_ids[i]]
                detected_aspect = det_box[2] / det_box[3] if det_box[3] > 0 else 1.0
                detected_area = det_box[2] * det_box[3]
                aspect_diff = abs(tracked_features['aspect_ratio'] - detected_aspect)
                area_diff = abs(tracked_features['area'] - detected_area) / max(tracked_features['area'], detected_area, 1)
                appearance_cost = aspect_diff + area_diff

                if self.cost_mode == 'centroid':
                    cost = np.linalg.norm(pred_state[:2] - np.array(det_box[:2]))
                elif self.cost_mode == 'iou':
                    cost = self.iou_cost_scale * (1 - self.calculate_iou(pred_state, det_box))
                else:
                    cost = (
                        np.linalg.norm(pred_state[:2] - np.array(det_box[:2]))
                        + self.iou_cost_scale * (1 - self.calculate_iou(pred_state, det_box))
                    )
                # Tambahkan cost appearance
                D[i, j] = cost + penalty + self.appearance_weight * appearance_cost
                
        row_indices, col_indices = linear_sum_assignment(D)
        used_rows, used_cols = set(), set()
        for row, col in zip(row_indices, col_indices):
            if D[row, col] > self.max_distance:
                continue
            object_id = object_ids[row]
            kf, _, _ = self.objects[object_id]
            kf.correct(np.array(input_boxes[col], np.float32))
            # Simpan history class untuk stabilisasi
            if object_id not in self.class_history:
                self.class_history[object_id] = []
            self.class_history[object_id].append(classes[col])
            if len(self.class_history[object_id]) > 10:
                self.class_history[object_id] = self.class_history[object_id][-10:]
            self.objects[object_id] = (kf, classes[col], confidences[col])
            self.disappeared[object_id] = 0
            used_rows.add(row)
            used_cols.add(col)

        for row in set(range(len(object_ids))) - used_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)

        unused_cols = set(range(len(input_boxes))) - used_cols
        for object_id in list(self.lost_tracks.keys()):
            self.lost_tracks[object_id]['age'] += 1
            if self.lost_tracks[object_id]['age'] > self.lost_track_patience:
                self.lost_tracks.pop(object_id, None)

        if len(unused_cols) > 0 and len(self.lost_tracks) > 0:
            lost_ids = list(self.lost_tracks.keys())
            det_indices = list(unused_cols)
            reid_D = np.zeros((len(lost_ids), len(det_indices)))
            for i, lost_id in enumerate(lost_ids):
                for j, det_idx in enumerate(det_indices):
                    dist = np.linalg.norm(self.lost_tracks[lost_id]['state'][:2] - np.array(input_boxes[det_idx][:2]))
                    reid_D[i, j] = dist

            rows_reid, cols_reid = linear_sum_assignment(reid_D)
            for row, col in zip(rows_reid, cols_reid):
                lost_id = lost_ids[row]
                det_idx = det_indices[col]
                if det_idx in used_cols:
                    continue
                if reid_D[row, col] < self.max_distance / 2:
                    self._register_with_id(lost_id, boxes[det_idx], classes[det_idx], confidences[det_idx])
                    self.lost_tracks.pop(lost_id, None)
                    used_cols.add(det_idx)

        for col in set(range(len(input_boxes))) - used_cols:
            self.register(boxes[col], classes[col], confidences[col])

        return self.objects