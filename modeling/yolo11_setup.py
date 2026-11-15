"""Validate the local dataset and initialize YOLO11 classification."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

EXPECTED_CLASSES = {"adidas", "converse", "nike"}
REQUIRED_SPLITS = ("train", "val")
MODEL_CHECKPOINT = "yolo11n-cls.pt"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check YOLO11 shoe-classification prerequisites without training."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).parent / "datasets" / "shoe_brands",
        help="Dataset root containing train/ and val/ class directories.",
    )
    parser.add_argument(
        "--model",
        default=MODEL_CHECKPOINT,
        help="YOLO11 classification checkpoint to initialize.",
    )
    return parser.parse_args()


def validate_dataset(dataset_root: Path) -> None:
    missing_directories: list[Path] = []

    for split_name in REQUIRED_SPLITS:
        split_path = dataset_root / split_name
        for class_name in sorted(EXPECTED_CLASSES):
            class_path = split_path / class_name
            if not class_path.is_dir():
                missing_directories.append(class_path)

    if missing_directories:
        formatted_paths = "\n".join(f"  - {path}" for path in missing_directories)
        raise FileNotFoundError(
            "Dataset structure is incomplete. Create these directories:\n"
            f"{formatted_paths}"
        )


def initialize_model(checkpoint: str) -> YOLO:
    return YOLO(checkpoint)


def main() -> None:
    arguments = parse_arguments()
    dataset_root = arguments.dataset.expanduser().resolve()
    validate_dataset(dataset_root)
    model = initialize_model(arguments.model)

    print(f"Dataset ready: {dataset_root}")
    print(f"Classes: {', '.join(sorted(EXPECTED_CLASSES))}")
    print(f"YOLO11 checkpoint ready: {model.ckpt_path}")
    print("Setup complete. Training was not started.")


if __name__ == "__main__":
    main()

