import onnx
import onnxruntime as ort
import numpy as np
from torchvision import transforms
from PIL import Image

class OnnxModel:
    def __init__(self, model_path: str):
        self.model = onnx.load(model_path)
        self.session = ort.InferenceSession(model_path)

    def predict(self, img: np.ndarray):
        input_name = self.session.get_inputs()[0].name
        return self.session.run(None, {input_name: img})[0]

class ImagePreprocessor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def preprocess(self, img: Image.Image) -> np.ndarray:
        img = self.transform(img)
        return img.numpy().astype(np.float32).reshape(1, 3, 224, 224)  # Reshape for ONNX model
