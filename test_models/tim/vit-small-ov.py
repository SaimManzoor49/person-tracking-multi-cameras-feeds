import cv2
import numpy as np
import time
import requests
from openvino.runtime import Core
from PIL import Image
from io import BytesIO

# Step 1: Set absolute paths for the model and weights
model_xml = "C:/Saim/python/fyp_3/test_models/tim/vit-small-ov/dino-vits16.xml"  # Update with your actual path
model_bin = "C:/Saim/python/fyp_3/test_models/tim/vit-small-ov/dino-vits16.bin"   # Corrected to .bin

# Load the OpenVINO runtime
ie = Core()

# Read the model
try:
    model = ie.read_model(model=model_xml, weights=model_bin)
except Exception as e:
    print(f"Error reading model: {e}")
    exit()

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

# Step 2: Load the image from the URL
img_url = 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
response = requests.get(img_url)
image = Image.open(BytesIO(response.content))

# Prepare the image for inference
input_image = preprocess_image(image)
input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension

# Start timing
start_time = time.time()

# Run inference on the same image 60 times
for _ in range(500):
    # Run inference
    result = compiled_model([input_image])
    # Process result if needed
    # For example, extracting the last hidden state
    last_hidden_states = result[0]

# End timing
end_time = time.time()

# Calculate total and average time taken
total_time_taken = end_time - start_time
average_time_per_image = total_time_taken / 60  # Average for 60 images

print(f"Total time taken to extract features from 500 images: {total_time_taken:.4f} seconds")
print(f"Average time per extraction: {average_time_per_image:.4f} seconds")

# Total time taken to extract features from 500 images: 54.7566 seconds
# Average time per extraction: 0.9126 seconds
#  9.13 FPS