import torch
from transformers import ViTModel, ViTImageProcessor

# Load the model and processor
model = ViTModel.from_pretrained('facebook/dino-vits16')
processor = ViTImageProcessor.from_pretrained('facebook/dino-vits16')

# Set the model to evaluation mode
model.eval()


# Dummy input for the model
dummy_image = torch.randn(1, 3, 224, 224)  # Change to appropriate size if needed

# Export the model
torch.onnx.export(
    model,
    dummy_image,
    "vit_model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)
