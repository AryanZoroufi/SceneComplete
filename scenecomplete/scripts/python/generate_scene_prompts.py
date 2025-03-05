import argparse
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

    # Construct the payload using the ChatCompletion API spec. 
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{user_prompt}"
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
        response.raise_for_status()  # Raise an exception for 4xx/5xx errors
    except requests.RequestException as e:
        print(f"Request to OpenAI API failed: {e}", file=sys.stderr)
        raise

    data = response.json()
    # Basic sanity check
    if "choices" not in data or len(data["choices"]) == 0:
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


def main():
    parser = argparse.ArgumentParser(
        description="Generate object prompts for scene segmentation using OpenAI's ChatCompletion API."
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to the input image file."
    )
    parser.add_argument(
        "--output_filepath",
        type=str,
        required=True,
        help="Path to output the generated prompts (one per line)."
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="OpenAI API key (if not provided, will attempt to use OPENAI_API_KEY environment variable)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model to use."
    )
    parser.add_argument(
        "--user_prompt",
        type=str,
        default=(
            "Describe the objects in the image as prompts (one per line) that's useful for an image segmentation model like SAM. "
            "Describe object using its generic name along with its color. "
            "Don't include the prompts background and table. Don't include any extra symbols in the response."
        ),
        help="User-level prompt that instructs the model on what to provide."
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=300,
        help="Maximum tokens to be used by the model for the response."
    )

    args = parser.parse_args()

    # Read the API key from command line or environment variable
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: No API key provided. Use --api_key or set OPENAI_API_KEY env var.", file=sys.stderr)
        sys.exit(1)

    # Read and encode the image
    base64_image = read_image_as_base64(args.image_path)

    # Call the OpenAI API
    try:
        completion_text = call_openai_chat_completion(
            api_key=api_key,
            model=args.model,
            user_prompt=args.user_prompt,
            base64_image=base64_image,
            max_tokens=args.max_tokens
        )
    except Exception as e:
        print(f"Failed to obtain prompts: {e}", file=sys.stderr)
        sys.exit(1)

    # Split the response into lines
    prompts_list = [line.strip() for line in completion_text.split("\n") if line.strip()]

    # Write prompts to file
    write_prompts_to_file(prompts_list, args.output_filepath)

    print(f"Prompts saved to: {args.output_filepath}")
    print(f"Prompts:\n  - " + "\n  - ".join(prompts_list))


if __name__ == "__main__":
    main()
