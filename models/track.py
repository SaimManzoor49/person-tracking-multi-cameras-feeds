from collections import deque
import numpy as np

class Track:
    def __init__(self, track_id: int, bbox, feature_vector):
        self.id = track_id
        self.bbox = bbox
        self.feature_history = deque(maxlen=30)
        self.feature_history.append(feature_vector)
        self.missed_frames = 0
        self.last_position = self._get_center(bbox)
        
    def _get_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def update(self, bbox, feature_vector):
        self.bbox = bbox
        self.feature_history.append(feature_vector)
        self.missed_frames = 0
        self.last_position = self._get_center(bbox)
    
    def get_average_feature(self):
        return np.mean(self.feature_history, axis=0)