from model import OnnxModel, ImagePreprocessor
from PIL import Image
import numpy as np

def test_model():
    model = OnnxModel("model.onnx")
    preprocessor = ImagePreprocessor()

    img = Image.open("./n01667114_mud_turtle.JPEG").convert("RGB")  # Ensure RGB format
    img_tensor = preprocessor.preprocess(img)
    predictions = model.predict(img_tensor)

    print("Predicted class ID:", np.argmax(predictions))

if __name__ == "__main__":
    test_model()
