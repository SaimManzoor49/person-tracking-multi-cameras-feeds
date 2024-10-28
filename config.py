import torch

DEFAULT_CONFIG = {
    'model_paths': {
        'yolo': './yolo11n.onnx',
        'reid': './osnet_x1_0_imagenet.pth'
    },
    'camera_sources': ['./111.mp4', './2.mp4'],
    'feature_file': 'person_features.txt',
    'detection_threshold': 0.5,
    'similarity_threshold': 0.5,
     'reid_timeout': 30.0,  # Time window for re-identification in seconds
    'similarity_threshold': 0.7,  # Adjust based on your needs
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}