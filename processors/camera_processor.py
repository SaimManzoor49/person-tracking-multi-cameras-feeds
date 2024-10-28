from typing import Dict, List
import numpy as np
from scipy.optimize import linear_sum_assignment
from models.track import Track
from models.feature_extractor import (
    load_existing_features, save_features, extract_features
)
from models.detector import YOLODetector
from models.feature_extractor import *
class CameraProcessor:
    def __init__(self, camera_id: int, video_source, 
                 detector: YOLODetector,
                 feature_extractor,
                 feature_file: str,
                 detection_threshold: float = 0.5,
                 similarity_threshold: float = 0.7,
                 max_missed_frames: int = 30):
        self.camera_id = camera_id
        self.video_source = video_source
        self.detector = detector
        self.feature_extractor = feature_extractor
        self.feature_file = feature_file
        self.detection_threshold = detection_threshold
        self.similarity_threshold = similarity_threshold
        self.max_missed_frames = max_missed_frames
        self.next_track_id = 1
        self.active_tracks = {}  # track_id -> Track object
        
    def _calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_distance(self, pos1, pos2):
        """Calculate Euclidean distance between two points."""
        return np.sqrt(((pos1[0] - pos2[0]) ** 2) + ((pos1[1] - pos2[1]) ** 2))
    
    def _calculate_cost_matrix(self, detections, features):
        """Calculate cost matrix for Hungarian algorithm using multiple metrics."""
        cost_matrix = np.zeros((len(detections), len(self.active_tracks)))
        
        for i, (det, feat) in enumerate(zip(detections, features)):
            det_center = ((det.bbox[0] + det.bbox[2]) / 2, (det.bbox[1] + det.bbox[3]) / 2)
            feat_vector = np.array(feat[1:])  # Remove track_id
            
            for j, track in enumerate(self.active_tracks.values()):
                # Spatial distance cost
                distance_cost = self._calculate_distance(det_center, track.last_position) / 100.0
                
                # Feature similarity cost
                avg_track_feat = track.get_average_feature()
                similarity = np.dot(feat_vector, avg_track_feat) / (
                    np.linalg.norm(feat_vector) * np.linalg.norm(avg_track_feat)
                )
                feature_cost = 1 - similarity
                
                # IOU cost
                iou = self._calculate_iou(det.bbox, track.bbox)
                iou_cost = 1 - iou
                
                # Combined cost (weighted sum)
                cost_matrix[i, j] = (0.4 * distance_cost + 
                                   0.4 * feature_cost + 
                                   0.2 * iou_cost)
        
        return cost_matrix
    
    def process_frame(self, frame):
        # Get detections from YOLO
        detections = self.detector.detect(frame)
        
        # Extract features for all detections
        features = []
        valid_detections = []
        
        for detection in detections:
            if detection.confidence < self.detection_threshold:
                continue
                
            x1, y1, x2, y2 = detection.bbox
            person_crop = frame[y1:y2, x1:x2]
            features.append(extract_features(self.feature_extractor, person_crop, -1))
            valid_detections.append(detection)
        
        # Load existing features for cross-camera matching
        existing_features = load_existing_features(self.feature_file)
        
        # Update missed frames counter for all tracks
        for track in self.active_tracks.values():
            track.missed_frames += 1
        
        # Remove tracks that have been missing for too long
        self.active_tracks = {
            track_id: track 
            for track_id, track in self.active_tracks.items()
            if track.missed_frames < self.max_missed_frames
        }
        
        processed_detections = []
        
        if valid_detections and self.active_tracks:
            # Calculate cost matrix
            cost_matrix = self._calculate_cost_matrix(valid_detections, features)
            
            # Use Hungarian algorithm for optimal assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # Process assignments
            for i, j in zip(row_ind, col_ind):
                cost = cost_matrix[i, j]
                if cost < 0.7:  # Threshold for accepting a match
                    track = list(self.active_tracks.values())[j]
                    track.update(valid_detections[i].bbox, features[i][1:])
                    
                    processed_detections.append({
                        'track_id': track.id,
                        'bbox': valid_detections[i].bbox,
                        'confidence': valid_detections[i].confidence
                    })
                    
                    # Save updated features
                    features[i][0] = track.id
                    save_features(self.feature_file, features[i])
                else:
                    # Create new track
                    new_track_id = self._assign_new_track_id(features[i], existing_features)
                    new_track = Track(new_track_id, valid_detections[i].bbox, features[i][1:])
                    self.active_tracks[new_track_id] = new_track
                    
                    processed_detections.append({
                        'track_id': new_track_id,
                        'bbox': valid_detections[i].bbox,
                        'confidence': valid_detections[i].confidence
                    })
                    
                    # Save new features
                    features[i][0] = new_track_id
                    save_features(self.feature_file, features[i])
        
        elif valid_detections:
            # If no active tracks, create new tracks for all detections
            for det, feat in zip(valid_detections, features):
                new_track_id = self._assign_new_track_id(feat, existing_features)
                new_track = Track(new_track_id, det.bbox, feat[1:])
                self.active_tracks[new_track_id] = new_track
                
                processed_detections.append({
                    'track_id': new_track_id,
                    'bbox': det.bbox,
                    'confidence': det.confidence
                })
                
                # Save new features
                feat[0] = new_track_id
                save_features(self.feature_file, feat)
        
        return processed_detections
    
    def _assign_new_track_id(self, features, existing_features):
        """Assign new track ID, checking for matches in existing features first."""
        match_found, matched_id = compare_features(features, existing_features, self.similarity_threshold)
        if match_found:
            return matched_id
        
        new_id = self.next_track_id
        self.next_track_id += 1
        return new_id