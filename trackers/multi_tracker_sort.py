import numpy as np
from scipy.optimize import linear_sum_assignment
from .kalman_filter import KalmanFilter


def iou_batch(bb_test, bb_gt):
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h

    area_test = np.maximum((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1]), 1e-6)
    area_gt = np.maximum((bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]), 1e-6)

    o = wh / (area_test + area_gt - wh)
    return np.nan_to_num(o, nan=0.0, posinf=0.0, neginf=0.0)


def is_similar(bbox1, bbox2, size_thresh=0.5, center_thresh=50):
    w1, h1 = bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]
    w2, h2 = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]
    size_diff = abs(w1 - w2) / max(w1, w2 + 1e-6) + abs(h1 - h2) / max(h1, h2 + 1e-6)

    cx1, cy1 = (bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2
    cx2, cy2 = (bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2
    center_dist = np.linalg.norm([cx1 - cx2, cy1 - cy2])

    return size_diff < size_thresh and center_dist < center_thresh


class MultiObjectTracker:
    def __init__(self, max_age=100, min_hits=3, iou_threshold=0.3, fps=30):
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.next_object_id = 0
        self.removed_static_trackers = []  # cache for reactivation
        self.fps = fps
        self.max_age = int(max_age * fps / 30)
        self.static_frame_thresh = int(1.0 * fps)  # 1 detik

    def update(self, dets_xyxy, confs, classes):
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []

        for t, trk in enumerate(trks):
            pred_state = self.trackers[t].predict()
            if pred_state is not None:
                cx, cy, w, h = pred_state[0], pred_state[1], pred_state[2], pred_state[3]
                x1, y1 = cx - w / 2, cy - h / 2
                x2, y2 = cx + w / 2, cy + h / 2
                trk[:] = [x1, y1, x2, y2, 0]
                if np.any(np.isnan(pred_state)):
                    to_del.append(t)
            else:
                to_del.append(t)

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        dets = np.array(dets_xyxy)
        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(dets, trks)

        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])
            self.trackers[m[1]].class_name = classes[m[0]]
            self.trackers[m[1]].confidence = confs[m[0]]

        for i in unmatched_dets:
            if confs[i] < 0.4:
                continue

            matched_from_cache = False
            for old_trk in self.removed_static_trackers:
                old_bbox = old_trk.get_state()[0]
                if is_similar(dets[i], old_bbox):
                    old_trk.kf.init(np.array([(dets[i, 0] + dets[i, 2]) / 2, (dets[i, 1] + dets[i, 3]) / 2,
                                               dets[i, 2] - dets[i, 0], dets[i, 3] - dets[i, 1]], np.float32))
                    old_trk.time_since_update = 0
                    old_trk.hit_streak = 1
                    old_trk.confidence = confs[i]
                    old_trk.class_name = classes[i]
                    self.trackers.append(old_trk)
                    self.removed_static_trackers.remove(old_trk)
                    matched_from_cache = True
                    break

            if matched_from_cache:
                continue

            kf = KalmanFilter()
            kf.init(np.array([(dets[i, 0] + dets[i, 2]) / 2, (dets[i, 1] + dets[i, 3]) / 2,
                              dets[i, 2] - dets[i, 0], dets[i, 3] - dets[i, 1]], np.float32))
            new_tracker = TrackerState(kf, self.next_object_id)
            new_tracker.class_name = classes[i]
            new_tracker.confidence = confs[i]
            new_tracker.hit_streak = -self.min_hits
            self.trackers.append(new_tracker)
            self.next_object_id += 1

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= 0):
                ret.append(np.concatenate((d, [trk.id, trk.confidence], [trk.get_class_id()])).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_age:
                if (trk.is_potential_static(frame_thresh=self.static_frame_thresh) and trk.hit_streak >= self.min_hits):
                    self.removed_static_trackers.append(trk)
                    continue
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 7))

    def associate_detections_to_trackers(self, detections, trackers):
        if len(detections) == 0:
            return np.empty((0, 2), dtype=int), np.empty((0, 1), dtype=int), np.arange(len(trackers))
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)

        iou_matrix = iou_batch(detections, trackers)
        matched_indices = []

        min_match_iou = max(self.iou_threshold, 0.4)

        if min(iou_matrix.shape) > 0:
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            matched_indices = np.array(list(zip(row_ind, col_ind)))
        else:
            matched_indices = np.empty((0, 2), dtype=int)

        unmatched_detections = list(range(len(detections)))
        unmatched_trackers = list(range(len(trackers)))
        matches = []

        for d, t in matched_indices:
            if iou_matrix[d, t] >= min_match_iou and is_similar(detections[d], trackers[t]):
                matches.append([d, t])
                unmatched_detections.remove(d)
                unmatched_trackers.remove(t)

        return np.array(matches), np.array(unmatched_detections), np.array(unmatched_trackers)


class TrackerState:
    def __init__(self, kf, track_id):
        self.kf = kf
        self.id = track_id
        self.time_since_update = 0
        self.hit_streak = 0
        self.age = 0
        self.class_name = None
        self.confidence = 0.0
        self.class_id_map = {"bus": 0, "car": 1, "motorbike": 2, "truck": 3}
        self.static_counter = 0
        self.is_static = False

    def update(self, bbox):
        self.time_since_update = 0
        self.hit_streak += 1
        cx, cy, w, h = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2, bbox[2] - bbox[0], bbox[3] - bbox[1]
        self.kf.correct(np.array([cx, cy, w, h], np.float32))

    def predict(self):
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        if self.velocity() < 0.05:
            self.static_counter += 1
            if self.static_counter > 30:
                self.is_static = True
        else:
            self.static_counter = 0
            self.is_static = False

        return self.kf.predict()

    def get_state(self):
        cx, cy, w, h = self.kf.kalman.statePost[:4].flatten()
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]).reshape((1, 4))

    def get_class_id(self):
        return self.class_id_map.get(self.class_name, -1)

    def velocity(self):
        vx, vy = self.kf.kalman.statePost[4:6].flatten()
        return np.sqrt(vx ** 2 + vy ** 2)

    def is_potential_static(self, frame_thresh, velocity_thresh=0.1):
        return self.static_counter >= frame_thresh or self.velocity() < velocity_thresh
