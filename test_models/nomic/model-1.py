import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
from PIL import Image
import requests
import time  # Import time module for measuring performance

# Load the image
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

# Initialize the image processor and model
processor = AutoImageProcessor.from_pretrained("nomic-ai/nomic-embed-vision-v1")
vision_model = AutoModel.from_pretrained("nomic-ai/nomic-embed-vision-v1", trust_remote_code=True)

# Start the timer
start_time = time.time()

# Run feature extraction for the same image 60 times
for _ in range(60):
    # Process the image
    inputs = processor(image, return_tensors="pt")
    
    # Get image embeddings
    img_emb = vision_model(**inputs).last_hidden_state
    img_embeddings = F.normalize(img_emb[:, 0], p=2, dim=1)

# End the timer
end_time = time.time()

# Calculate the total time taken for feature extraction
total_time_taken = end_time - start_time

print(f"Total time taken to extract features from 60 images: {total_time_taken:.4f} seconds")
print(f"Average time per extraction: {total_time_taken / 60:.4f} seconds")

# Total time taken to extract features from 60 images: 29.9756 seconds
# Average time per extraction: 0.4996 seconds