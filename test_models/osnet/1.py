import torch
from urllib.request import urlopen
from PIL import Image
import timm
import time  # Import time module for measuring performance
from torchvision import transforms

# Load the image
url = 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
image = Image.open(urlopen(url))

# Initialize the OSNet model using timm
model = timm.create_model('osnet_x1_0', pretrained=True, num_classes=0)  # Set num_classes to 0 to remove the classifier
model.eval()

# Define the transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the input size of the model
    transforms.ToTensor(),            # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Start the timer
start_time = time.time()

# Run feature extraction for the same image 60 times
for _ in range(60):
    # Transform and unsqueeze the image to add a batch dimension
    input_tensor = transform(image).unsqueeze(0)
    
    # Perform inference
    with torch.no_grad():  # Disable gradient calculation
        output = model(input_tensor)  # output is (batch_size, num_features) shaped tensor

# End the timer
end_time = time.time()

# Calculate the total time taken for feature extraction
total_time_taken = end_time - start_time

print(f"Total time taken to extract features from 60 images: {total_time_taken:.4f} seconds")
print(f"Average time per extraction: {total_time_taken / 60:.4f} seconds")
