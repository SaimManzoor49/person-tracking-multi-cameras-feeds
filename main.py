import cv2
from typing import Dict
import torch
from models.detector import YOLODetector
from models.feature_extractor import (
    initialize_feature_extractor, clean_feature_file
)
from processors.camera_processor import CameraProcessor
from utils.visualization import draw_detections

def initialize_system(config: Dict):
    """Initialize all components of the tracking system."""
    detector = YOLODetector(
        model_path=config['model_paths']['yolo'],
        conf_threshold=config['detection_threshold']
    )
    
    feature_extractor = initialize_feature_extractor(
        model_name='osnet_x1_0',
        model_path=config['model_paths']['reid'],
        device=config['device']
    )
    
    clean_feature_file(config['feature_file'])
    
    processors = []
    for cam_id, source in enumerate(config['camera_sources']):
        processor = CameraProcessor(
            camera_id=cam_id,
            video_source=source,
            detector=detector,
            feature_extractor=feature_extractor,
            feature_file=config['feature_file'],
            detection_threshold=config['detection_threshold'],
            similarity_threshold=config['similarity_threshold']
        )
        processors.append(processor)
    
    return processors

def run_tracking_system(config: Dict):
    """Main function to run the multi-camera tracking system."""
    processors = initialize_system(config)
    caps = [cv2.VideoCapture(source) for source in config['camera_sources']]
    
    try:
        while True:
            for processor, cap in zip(processors, caps):
                ret, frame = cap.read()
                if not ret:
                    continue
                
                detections = processor.process_frame(frame)
                frame = draw_detections(frame, detections)
                cv2.imshow(f"Camera {processor.camera_id}", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        for cap in caps:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    from config import DEFAULT_CONFIG
    run_tracking_system(DEFAULT_CONFIG)