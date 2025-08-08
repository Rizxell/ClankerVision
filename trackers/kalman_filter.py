# === File: trackers/kalman_filter.py ===
import cv2
import numpy as np

class KalmanFilter:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(8, 4)
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ], np.float32)
        self.kalman.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ], np.float32)
        self.kalman.processNoiseCov = np.eye(8, dtype=np.float32) * 0.01
        self.kalman.processNoiseCov[0:4, 0:4] *= 0.5
        self.kalman.processNoiseCov[4:8, 4:8] *= 2.0
        self.kalman.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.05
        self.kalman.errorCovPost = np.eye(8, dtype=np.float32)
        self.last_measurement = None
        self.initialized = False

    def init(self, measurement):
        self.kalman.statePost = np.array([
            [measurement[0]], [measurement[1]], [measurement[2]], [measurement[3]],
            [0], [0], [0], [0]
        ], np.float32)
        self.initialized = True
        self.last_measurement = measurement

    def predict(self):
        if not self.initialized:
            return None
        predicted = self.kalman.predict()
        return predicted[:4].flatten()

    def correct(self, measurement):
        if not self.initialized:
            self.init(measurement)
            return measurement
        corrected = self.kalman.correct(np.array([[measurement[0]], [measurement[1]],
                                                  [measurement[2]], [measurement[3]]], np.float32))
        self.last_measurement = measurement
        return corrected[:4].flatten()