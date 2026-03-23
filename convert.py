import torch
import torch.onnx

# Load the PyTorch model
model = torch.load("rmvpe.pt")
model.eval()

# Create dummy input (adjust shape based on your model)
dummy_input = torch.randn(1, 1, 16000)  # Example: 1 second of audio

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "rmvpe.onnx",
    input_names=["audio"],
    output_names=["f0"],
    dynamic_axes={"audio": {2: "length"}, "f0": {1: "frames"}},
    opset_version=17
)