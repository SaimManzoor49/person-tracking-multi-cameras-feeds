from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests
import torch  # Import PyTorch
import time  # Import the time module

# Load the image
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

# Initialize the processor and model
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
model = AutoModel.from_pretrained('facebook/dinov2-small')

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Start the timer
start_time = time.time()

# Run feature extraction for the same image 60 times
for _ in range(60):
    # Prepare the inputs for each iteration and move to GPU
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    # Extract features
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state

# End the timer
end_time = time.time()

# Calculate the total time taken for feature extraction
total_time_taken = end_time - start_time

print(f"Total time taken to extract features from 60 images: {total_time_taken:.4f} seconds")
print(f"Average time per extraction: {total_time_taken / 60:.4f} seconds")


# Total time taken to extract features from 60 images: 9.9949 seconds
# Average time per extraction: 0.1666 seconds