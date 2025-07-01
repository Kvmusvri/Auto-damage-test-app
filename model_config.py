import os
from hub_sdk import HUBClient
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("ULTRALYTICS_API_KEY")

client = HUBClient({"api_key": api_key})

def get_model_map():
    private_models_list = client.model_list().results
    return {
        model['meta']['name']: model['id']
        for model in private_models_list
    }

MODEL_MAP = get_model_map()

if __name__ == "__main__":
    print(api_key)
