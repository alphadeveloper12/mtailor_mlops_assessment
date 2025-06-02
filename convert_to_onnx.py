import torch
from pytorch_model import Classifier  # Assuming your model is in pytorch_model.py

def convert_model_to_onnx():
    model = Classifier()  # Initialize your model
    model.load_state_dict(torch.load("./resnet18-f37072fd.pth"))
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)  # Dummy input for ONNX
    torch.onnx.export(model, dummy_input, "model.onnx",
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})

if __name__ == "__main__":
    convert_model_to_onnx()
