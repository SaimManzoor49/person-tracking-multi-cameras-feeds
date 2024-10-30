from openvino.runtime import Core
from transformers import ViTImageProcessor
from PIL import Image
import requests
import time
import numpy as np

# Load the image
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

# Initialize the image processor
processor = ViTImageProcessor.from_pretrained('facebook/dino-vits16')

# Load the OpenVINO model
ie = Core()
model_xml = "C:/Saim/python/fyp_3/test_models/facebook/dino-vits-ov/dino-vits16.xml"  # Update with your actual path
model_bin = "C:/Saim/python/fyp_3/test_models/facebook/dino-vits-ov/dino-vits16.bin"   # Update with your actual path

# Read and compile the model for inference
try:
    model = ie.read_model(model=model_xml, weights=model_bin)
    compiled_model = ie.compile_model(model, "CPU")  # Change to "GPU" if you have a compatible GPU
except Exception as e:
    print(f"Error loading the model: {e}")
    exit()

# Prepare the input for inference
inputs = processor(images=image, return_tensors="np")  # Return as NumPy array
input_data = inputs['pixel_values']  # Shape: (1, 3, 224, 224)

# Start the timer
start_time = time.time()

# Run feature extraction for the same image 60 times
for _ in range(500):
    # Perform inference
    output_data = compiled_model(input_data)

# End the timer
end_time = time.time()

# Calculate the total time taken for feature extraction
total_time_taken = end_time - start_time

# Retrieve the last hidden states
last_hidden_states = output_data[0]  # Assuming the output is in the first position

print(f"Total time taken to extract features from 500 images: {total_time_taken:.4f} seconds")
print(f"Average time per extraction: {total_time_taken / 60:.4f} seconds")

# Total time taken to extract features from 500 images: 49.5800 seconds
# Average time per extraction: 0.8263 seconds
# 10fps avg