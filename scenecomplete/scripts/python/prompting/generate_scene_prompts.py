import argparse
import os
import sys

from scenecomplete.scripts.python.prompting.utils.prompt_utils import (
    read_image_as_base64,
    call_openai_chat_completion,
    write_prompts_to_file
)


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

    try:
        # Read and encode the image
        base64_image = read_image_as_base64(args.image_path)

        # Call the OpenAI API
        completion_text = call_openai_chat_completion(
            api_key=api_key,
            model=args.model,
            user_prompt=args.user_prompt,
            base64_image=base64_image,
            max_tokens=args.max_tokens
        )

        # Split the response into lines
        prompts_list = [line.strip() for line in completion_text.split("\n") if line.strip()]

        # Write prompts to file
        write_prompts_to_file(prompts_list, args.output_filepath)

        print(f"Prompts saved to: {args.output_filepath}")
        print(f"Prompts:\n  - " + "\n  - ".join(prompts_list))

    except Exception as e:
        print(f"Failed to generate prompts: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
