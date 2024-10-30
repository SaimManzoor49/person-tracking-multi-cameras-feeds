import cv2
import numpy as np
import requests
from openvino.runtime import Core
from PIL import Image
from io import BytesIO
from scipy.spatial.distance import cosine  # For similarity calculation

# Set absolute paths for the model and weights
model_xml = "C:/Saim/python/fyp_3/test_models/tim/vit-small-ov/dino-vits16.xml"  # Update with your actual path
model_bin = "C:/Saim/python/fyp_3/test_models/tim/vit-small-ov/dino-vits16.bin"   # Update with your actual path

# Load the OpenVINO runtime
ie = Core()
model = ie.read_model(model=model_xml, weights=model_bin)
compiled_model = ie.compile_model(model=model, device_name="CPU")

# Prepare the input shape
input_shape = model.input(0).shape  # e.g., [1, 3, 224, 224]
input_width, input_height = input_shape[3], input_shape[2]

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((input_width, input_height))  # Resize
    image = np.array(image)  # Convert to numpy array
    image = np.transpose(image, (2, 0, 1))  # Change data layout from HWC to CHW
    image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
    return image

# Function to extract features from an image
def extract_features(image_url):
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    input_image = preprocess_image(image)
    input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension
    result = compiled_model([input_image])
    return result[0]  # Return the extracted features

# Step 1: Extract features from two images
image_url1 = 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
image_url2 = 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'  # Same image for testing

features1 = extract_features(image_url1)
features2 = extract_features(image_url2)

# Step 2: Compute similarity between features
similarity_score = 1 - cosine(features1.flatten(), features2.flatten())  # Using cosine similarity

# Output the results
print(f"Similarity score between the two images: {similarity_score:.4f}")
