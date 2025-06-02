import base64
from io import BytesIO
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from model import ImagePreprocessor, OnnxModel

app = FastAPI()

# Load once
model = OnnxModel("model.onnx")
preprocessor = ImagePreprocessor()

class ImageRequest(BaseModel):
    param_1: str

@app.post("/predict")
async def predict(data: ImageRequest):
    try:
        img_bytes = base64.b64decode(data.param_1)
        img = Image.open(BytesIO(img_bytes)).convert("RGB")

        img_tensor = preprocessor.preprocess(img)
        predictions = model.predict(img_tensor)
        class_id = int(predictions.argmax())

        return {"my_result": {"class_id": class_id}}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}
