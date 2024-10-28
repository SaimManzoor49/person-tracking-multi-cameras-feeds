from torchreid.utils import FeatureExtractor
import numpy as np
import os

def initialize_feature_extractor(model_name, model_path, device):
    return FeatureExtractor(
        model_name=model_name,
        model_path=model_path,
        device=device
    )

def clean_feature_file(file_path):
    """Remove the feature file if it exists, to start fresh."""
    if os.path.exists(file_path):
        os.remove(file_path)

def load_existing_features(file_path):
    existing_features = {}
    if not os.path.exists(file_path):
        return existing_features  # Ensure a dictionary is returned
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = list(map(float, line.strip().split(',')))
        track_id = int(parts[0])  # First element as track ID
        feature_vector = parts[1:]  # Remaining as feature vector
        
        if track_id not in existing_features:
            existing_features[track_id] = []
        existing_features[track_id].append(feature_vector)
    
    return existing_features

def save_features(file_path, features):
    """Append new features to the file to avoid reloading all features each time."""
    with open(file_path, 'a') as f:
        f.write(','.join(map(str, features)) + '\n')

def extract_features(extractor, frame, track_id):
    """Extract features from a frame and attach a tracking ID to the features."""
    features = extractor(frame).flatten().tolist()
    features.insert(0, track_id)  # Prepend track ID for easy identification
    return features

def compare_features(new_features, existing_features, threshold):
    """
    Compare new features with existing ones.
    Returns the ID of the matched person if similarity exceeds threshold, otherwise None.
    """
    new_feature_vector = np.array(new_features[1:])  # Remove track ID for comparison
    best_match_id = None
    highest_similarity = threshold

    # Compare against each stored feature vector
    for existing_id, feature_list in existing_features.items():
        for existing_feature_vector in feature_list:
            existing_feature_vector = np.array(existing_feature_vector)
            similarity = np.dot(new_feature_vector, existing_feature_vector) / (
                np.linalg.norm(new_feature_vector) * np.linalg.norm(existing_feature_vector)
            )
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match_id = existing_id
                if similarity >= 0.99:  # Early stopping if a near-perfect match
                    return True, best_match_id

    return (best_match_id is not None), best_match_id
