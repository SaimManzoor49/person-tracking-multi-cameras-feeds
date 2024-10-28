# detector.py
from dataclasses import dataclass
from typing import List, Tuple
from ultralytics import YOLO

@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]
    confidence: float

class YOLODetector:
    def __init__(self, model_path: str, conf_threshold: float, person_class_id: int = 0):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.person_class_id = person_class_id
    
    def detect(self, frame) -> List[Detection]:
        results = self.model(frame, conf=self.conf_threshold)
        detections = []
        
        if len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                if int(box.cls[0]) == self.person_class_id:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confidence = float(box.conf[0])
                    detections.append(Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=confidence
                    ))
        
        return detections
