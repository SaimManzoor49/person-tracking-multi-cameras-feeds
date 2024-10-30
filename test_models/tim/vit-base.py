from urllib.request import urlopen
from PIL import Image
import timm
import torch
import time

# Load the image
img_url = 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
img = Image.open(urlopen(img_url))

# Initialize the model
model = timm.create_model('vit_base_patch16_224.orig_in21k', pretrained=True)
model.eval()  # Set model to evaluation mode

# Get model-specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

# Prepare the image for the model (transformations)
input_tensor = transforms(img).unsqueeze(0)  # Add batch dimension

# Start timing
start_time = time.time()

# Run inference for the same image 60 times
for _ in range(60):
    with torch.no_grad():  # Disable gradient calculations for inference
        output = model(input_tensor)

# End timing
end_time = time.time()

# Calculate total time and average time per inference
total_time_taken = end_time - start_time
average_time_per_inference = total_time_taken / 60

# Get top 5 probabilities and class indices
top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)

print(f"Total time taken to process the image 60 times: {total_time_taken:.4f} seconds")
print(f"Average time per inference: {average_time_per_inference:.4f} seconds")
print("Top 5 probabilities:", top5_probabilities)
print("Top 5 class indices:", top5_class_indices)

# Total time taken to process the image 60 times: 25.3796 seconds
# Average time per inference: 0.4230 seconds