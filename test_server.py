import requests
import argparse
import base64
import json

# Default API key and model URL — override with CLI args if desired
DEFAULT_API_KEY = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJwcm9qZWN0SWQiOiJwLTE4ZDNhMzlmIiwiaWF0IjoxNzQ4ODk1NjYyLCJleHAiOjIwNjQ0NzE2NjJ9.ZvNGYI53rIoxX8GuHZqmlEiLUGpbCYj4JKfQrojZldxYvMdivOjqcfvG2li-vys-1gisgC5PqTEgwgbmrX_DqmtG1wIrA2IuAIG37OK5ilxma-Fiu86HFEaKbtXZZmIlPp0Zha_YTVkRfWB3S4i5e418aLu6j8_Vw_SAmHImb0aJuZO9WfFeJn0EUrYnkQ_U-QjbhjetaafobtgLLPcMt9KBNbfp61D_3e7xQAdM26W83rwwC_SN4wui64q1nbwsew26Vkv1_pSmMIAeoik82hzHyNjZUzU56-TcU4PsWF-KnLDkMIZtecxFp6OP_h_qPIJg04bez3x-1QexwkIAnQ"
DEFAULT_MODEL_URL = "https://api.cortex.cerebrium.ai/v4/p-18d3a39f/5-dockerfile/predict"

def test_deployed_model(image_path: str, api_key: str, model_url: str):
    with open(image_path, "rb") as img_file:
        img_bytes = img_file.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    payload = {
        "param_1": img_b64,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    response = requests.post(model_url, headers=headers, json=payload)

    try:
        res_json = response.json()
        print(json.dumps(res_json, indent=4))

    except Exception as e:
        print("\n❌ Failed to parse JSON response. Raw response:")
        print(response.text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image file")
    parser.add_argument("--api_key", type=str, default=DEFAULT_API_KEY, help="API key for authentication")
    parser.add_argument("--model_url", type=str, default=DEFAULT_MODEL_URL, help="URL of the deployed model")
    args = parser.parse_args()

    test_deployed_model(args.image_path, args.api_key, args.model_url)
