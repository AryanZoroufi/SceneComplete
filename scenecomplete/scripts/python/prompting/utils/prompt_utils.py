import base64
import os
import os.path as osp
import requests
import sys
from typing import List


def read_image_as_base64(image_path: str) -> str:
    """
    Reads an image from disk and returns its base64-encoded string.
    Raises FileNotFoundError if the image does not exist.
    """
    if not osp.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def call_openai_chat_completion(
    api_key: str,
    model: str,
    user_prompt: str,
    base64_image: str,
    max_tokens: int = 300
) -> str:
    """
    Calls the OpenAI ChatCompletion API with the given parameters.

    :param api_key: The OpenAI API key used for authorization. Generate your API key here https://platform.openai.com/api-keys.
    :param model: The model to use.
    :param user_prompt: The content the user is asking about.
    :param base64_image: The base64-encoded image string.
    :param max_tokens: The maximum number of tokens in the completion.
    :return: The text content of the model's response.
    :raises requests.RequestException: If the request fails.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": max_tokens
    }

    url = "https://api.openai.com/v1/chat/completions"

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Request to OpenAI API failed: {e}", file=sys.stderr)
        raise

    data = response.json()
    if "choices" not in data or not data["choices"]:
        raise ValueError("No completion choices received from API.")

    return data["choices"][0]["message"]["content"]


def write_prompts_to_file(prompts: List[str], output_filepath: str) -> None:
    """
    Writes each prompt on a new line to the given output file.
    """
    os.makedirs(osp.dirname(output_filepath), exist_ok=True)
    with open(output_filepath, "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(f"{p}\n")