import torch
from urllib.request import urlopen
from PIL import Image
import timm
import time  # Import time module for measuring performance

# Load the image
url = 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
image = Image.open(urlopen(url))

# Initialize the model
model = timm.create_model(
    'vit_small_patch16_224.dino',
    pretrained=True,
    num_classes=0,  # remove classifier nn.Linear
)
model.eval()

# Get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

# Start the timer
start_time = time.time()

# Run feature extraction for the same image 60 times
for _ in range(60):
    # Transform and unsqueeze the image to add a batch dimension
    input_tensor = transforms(image).unsqueeze(0)
    
    # Perform inference
    output = model(input_tensor)  # output is (batch_size, num_features) shaped tensor
    
    # Alternatively, using the model's features
    output_features = model.forward_features(input_tensor)  # (1, 197, 384) shaped tensor
    output_head = model.forward_head(output_features, pre_logits=True)  # (1, num_features) shaped tensor

# End the timer
end_time = time.time()

# Calculate the total time taken for feature extraction
total_time_taken = end_time - start_time

print(f"Total time taken to extract features from 60 images: {total_time_taken:.4f} seconds")
print(f"Average time per extraction: {total_time_taken / 60:.4f} seconds")

# Total time taken to extract features from 60 images: 18.1134 seconds
# Average time per extraction: 0.3019 seconds