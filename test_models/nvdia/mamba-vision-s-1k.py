from transformers import AutoModelForImageClassification
from PIL import Image
from timm.data.transforms_factory import create_transform
import requests
import time  # Import the time module
import torch  # Import PyTorch

# Load the model
model = AutoModelForImageClassification.from_pretrained("nvidia/MambaVision-S-1K", trust_remote_code=True)

# Set the model to evaluation mode and move it to GPU
model.cuda().eval()

# Prepare image for the model
url = 'http://images.cocodataset.org/val2017/000000020247.jpg'
image = Image.open(requests.get(url, stream=True).raw)
input_resolution = (3, 224, 224)  # MambaVision supports any input resolutions

# Create the transformation
transform = create_transform(input_size=input_resolution,
                             is_training=False,
                             mean=model.config.mean,
                             std=model.config.std,
                             crop_mode=model.config.crop_mode,
                             crop_pct=model.config.crop_pct)

# Start the timer
start_time = time.time()

# Run model inference for the same image 60 times
for _ in range(60):
    # Prepare inputs for the model and move to GPU
    inputs = transform(image).unsqueeze(0).cuda()
    
    # Model inference
    outputs = model(inputs)
    logits = outputs['logits']
    predicted_class_idx = logits.argmax(-1).item()

# End the timer
end_time = time.time()

# Calculate the total time taken for inference
total_time_taken = end_time - start_time

print(f"Total time taken to extract features from 60 images: {total_time_taken:.4f} seconds")
print(f"Average time per extraction: {total_time_taken / 60:.4f} seconds")
print("Predicted class:", model.config.id2label[predicted_class_idx])
