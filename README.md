# MTailor ML Ops Assessment - Image Classification Deployment on Cerebrium

This repository contains all the necessary components to convert a PyTorch classification model to ONNX, test it locally, deploy it using Cerebrium’s serverless GPU platform, and interact with it through API calls.

---

## 🚀 Project Structure

```
mtailor_mlops_assessment/
│
├── accessment/                  # Folder for Cerebrium deployment
│   ├── cerebrium.toml           # Cerebrium config file
│   ├── Dockerfile               # Custom Docker image for deployment
│   ├── main.py                  # Main FastAPI app for ONNX model inference
│   ├── model.onnx               # Converted ONNX model
│   ├── model.py                 # Contains inference and preprocessing logic
│   ├── requirements.txt         # Python dependencies
│
├── convert_to_onnx.py           # Script to convert PyTorch model to ONNX
├── model.py                     # Modularized model code (ONNX runtime + preprocessing)
├── pytorch_model.py             # Original PyTorch model definition and weights usage
├── test.py                      # Local test script for inference using ONNX model
├── test_server.py               # Script to test deployed Cerebrium API
├── n01440764_tench.jpeg         # Sample image - class id 0
├── n01667114_mud_turtle.JPEG    # Sample image - class id 35
├── model.onnx                   # ONNX model (top-level copy)
├── resnet18-f37072fd.pth        # PyTorch weights
└── README.md                    # This file
```


## 📌 Requirements

- Python 3.8+
- [Cerebrium CLI](https://docs.cerebrium.ai/)
- Docker
- Git

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r accessment/requirements.txt
```


# 🔧 Step-by-Step Instructions

## ✅ 1. Convert PyTorch to ONNX

```bash
python convert_to_onnx.py
 ```
 ## ✅ 2. Local Testing
 Run the test.py file:

```bash
python test.py --image_path n01440764_tench.jpeg
```
Expected output:
Class ID: 0

## ✅ 3. Deploy on Cerebrium
Navigate to the accessment/ folder:
```bash
cd accessment
```
Login to Cerebrium and initialize project:
```bash
cerebrium login
cerebrium init
```
Then deploy:
```bash
cerebrium deploy
```
## ✅ 4. Test Deployed API
```bash
python test_server.py --image_path ../n01440764_tench.jpeg
```
Expected output:
```bash
{
    "my_result": {
        "class_id": 35
    }
}
```
You can also run internal tests using the --run_tests flag:
```bash
python test_server.py --run_tests --api_key <YOUR_API_KEY> --api_url <MODEL_API_URL>
```
# 🔍 Files Overview
## convert_to_onnx.py
- Loads the PyTorch model.
- Converts the model to ONNX format.
- Ensures dynamic input shape.
- Applies proper export settings.

##model.py
Contains:

- ImagePreprocessor class: Handles image resizing, normalization.
- OnnxModelHandler class: Loads ONNX model and performs inference.

## main.py
- FastAPI app.
- Exposes /predict endpoint that accepts base64-encoded image and returns class ID.

##test.py
- Tests the model locally using model.onnx.

## test_server.py
- Sends image to the deployed Cerebrium endpoint.
- Has a --run_tests flag to test platform-level behavior.

# 🧪 Testing
- **Unit Tests**: In `test.py`
- **API Tests**: In `test_server.py` with live Cerebrium model
- Supports expected image classification outputs
- Validates:
  - Output class
  - Latency (<3s)

# 🔁 Future Improvements
- Add GitHub Actions CI for Docker build testing
- Integrate ONNX pre-processing into the model graph itself for faster inference
- Use TorchScript instead of ONNX if needed for speed

# ✅ Submission Checklist
- ONNX conversion
- Local model testing
- FastAPI server with ONNX inference
- Dockerfile for deployment
- Cerebrium deployment with `cerebrium.toml`
- `test_server.py` with live API calls
- `README` with full steps
- Loom walkthrough recorded





