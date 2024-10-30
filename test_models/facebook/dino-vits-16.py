from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import requests
import time  # Import the time module
import torch  # Import PyTorch

# Load the image
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

# Initialize the processor and model
processor = ViTImageProcessor.from_pretrained('facebook/dino-vits16')
model = ViTModel.from_pretrained('facebook/dino-vits16')

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # Set model to evaluation mode

# Start the timer
start_time = time.time()

# Run feature extraction for the same image 60 times
for _ in range(500):
    # Prepare the inputs for each iteration and move to GPU
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    # Extract features
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state

# End the timer
end_time = time.time()

# Calculate the total time taken for feature extraction
total_time_taken = end_time - start_time

print(f"Total time taken to extract features from 500 images: {total_time_taken:.4f} seconds")
print(f"Average time per extraction: {total_time_taken / 60:.4f} seconds")


# Total time taken to extract features from 500 images: 66.6447 seconds
# Average time per extraction: 1.1107 seconds
# avg 7fps