"""Run shoe-brand classification through the Roboflow Serverless API."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from dotenv import load_dotenv
from inference_sdk import InferenceConfiguration, InferenceHTTPClient

API_URL = "https://serverless.roboflow.com"
MODEL_ID = "shoe-classification-k9ajr/2"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify a shoe image as adidas, converse, or nike."
    )
    parser.add_argument("image", help="Local image path or public image URL.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for saving the JSON response.",
    )
    return parser.parse_args()


def is_http_url(value: str) -> bool:
    parsed_url = urlparse(value)
    return parsed_url.scheme in {"http", "https"} and bool(parsed_url.netloc)


def validate_image_source(image_source: str) -> str:
    if is_http_url(image_source):
        return image_source

    image_path = Path(image_source).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    return str(image_path)


def get_api_key() -> str:
    load_dotenv(Path(__file__).with_name(".env"))
    api_key = os.getenv("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "ROBOFLOW_API_KEY is not set. Add it to the environment or modeling/.env."
        )
    return api_key


def run_inference(image_source: str, api_key: str) -> Any:
    client = InferenceHTTPClient(
        api_url=API_URL,
        api_key=api_key,
    ).configure(
        InferenceConfiguration(api_key_transport="header")
    )
    return client.infer(image_source, model_id=MODEL_ID)


def save_result(result: Any, output_path: Path) -> None:
    resolved_output = output_path.expanduser().resolve()
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    resolved_output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    arguments = parse_arguments()
    image_source = validate_image_source(arguments.image)
    result = run_inference(image_source, get_api_key())

    print(json.dumps(result, indent=2, ensure_ascii=False))
    if arguments.output:
        save_result(result, arguments.output)


if __name__ == "__main__":
    main()

