import requests
import json
import os
from pathlib import Path

# API配置
API_URL = "http://127.0.0.1:5001"
CURRENT_DIR = Path(__file__).parent
DATA_PATH = CURRENT_DIR / "data" / "SMK_CAN_187.mat"


def test_api():
    try:
        # 检查数据文件
        if not DATA_PATH.exists():
            print(f"Error: Data file not found at {DATA_PATH}")
            print(f"Current working directory: {os.getcwd()}")
            return

        # 准备配置
        config = {
            "classifiers": ["RandomForestClassifier", "SVCClassifier_2"],
            "splits": {
                "train": 0.2,
                "val": 0.5,
                "test": 0.3
            }
        }

        # 发送训练请求
        with open(DATA_PATH, 'rb') as f:
            print("Sending request to:", f"{API_URL}/api/train")
            print("Config:", json.dumps(config, indent=2))

            response = requests.post(
                f"{API_URL}/api/train",
                files={'file': ('SMK_CAN_187.mat', f, 'application/octet-stream')},
                data={'config': json.dumps(config)}
            )

            # 打印响应信息
            print(f"Response Status Code: {response.status_code}")
            print(f"Response Headers: {dict(response.headers)}")

            try:
                print(f"Response Content: {response.content.decode()}")
            except:
                print(f"Raw Response Content: {response.content}")

    except requests.exceptions.ConnectionError:
        print(f"Connection Error: Could not connect to {API_URL}")
        print("Please make sure the server is running")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    test_api()