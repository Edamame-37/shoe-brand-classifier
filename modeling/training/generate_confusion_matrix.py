"""Generate a publication-ready confusion matrix for the final model report."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = (
    PROJECT_ROOT
    / "modeling"
    / "runs"
    / "training"
    / "yolo11s_robust"
    / "confusion_matrix.csv"
)
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "docs" / "assets" / "confusion_matrix.png"
EXPECTED_CLASSES = ("adidas", "converse", "nike")


def parse_arguments() -> argparse.Namespace:
    """Parse input and output paths from the command line."""
    parser = argparse.ArgumentParser(
        description="Generate the final model confusion matrix as a PNG image."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to the confusion-matrix CSV file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination path for the generated PNG image.",
    )
    return parser.parse_args()


def load_confusion_matrix(input_path: Path) -> pd.DataFrame:
    """Load and validate the confusion matrix exported by the evaluation pipeline."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Confusion-matrix CSV not found: {input_path}")

    confusion_matrix = pd.read_csv(input_path, index_col=0)
    expected_labels = list(EXPECTED_CLASSES)
    if confusion_matrix.index.tolist() != expected_labels:
        raise ValueError(
            "Unexpected row labels: "
            f"{confusion_matrix.index.tolist()}; expected {expected_labels}."
        )
    if confusion_matrix.columns.tolist() != expected_labels:
        raise ValueError(
            "Unexpected column labels: "
            f"{confusion_matrix.columns.tolist()}; expected {expected_labels}."
        )
    if confusion_matrix.isna().any().any():
        raise ValueError("Confusion matrix contains missing values.")

    numeric_values = confusion_matrix.to_numpy(dtype=float)
    if np.any(numeric_values < 0) or not np.all(numeric_values == np.floor(numeric_values)):
        raise ValueError("Confusion matrix must contain non-negative integer counts.")

    return confusion_matrix.astype(int)


def build_annotations(confusion_matrix: pd.DataFrame) -> np.ndarray:
    """Create annotations containing counts and row-normalized percentages."""
    counts = confusion_matrix.to_numpy()
    row_totals = counts.sum(axis=1, keepdims=True)
    if np.any(row_totals == 0):
        raise ValueError("Every actual class must contain at least one sample.")

    row_percentages = counts / row_totals * 100
    return np.asarray(
        [
            f"{count}\n({percentage:.1f}%)"
            for count, percentage in zip(counts.flat, row_percentages.flat)
        ]
    ).reshape(counts.shape)


def save_confusion_matrix(confusion_matrix: pd.DataFrame, output_path: Path) -> None:
    """Render and save a high-resolution confusion-matrix heatmap."""
    annotations = build_annotations(confusion_matrix)
    counts = confusion_matrix.to_numpy()
    accuracy = np.trace(counts) / counts.sum()

    sns.set_theme(style="white", font_scale=1.05)
    figure, axis = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(
        confusion_matrix,
        annot=annotations,
        fmt="",
        cmap="Blues",
        vmin=0,
        vmax=int(counts.max()),
        square=True,
        linewidths=1.0,
        linecolor="white",
        cbar_kws={"label": "Number of images", "shrink": 0.82},
        annot_kws={"fontsize": 11, "fontweight": "semibold"},
        ax=axis,
    )
    axis.set_title(
        f"YOLO11s Test Confusion Matrix\nAccuracy: {accuracy:.2%} (n={counts.sum()})",
        fontsize=15,
        fontweight="bold",
        pad=16,
    )
    axis.set_xlabel("Predicted brand", fontsize=12, labelpad=10)
    axis.set_ylabel("Actual brand", fontsize=12, labelpad=10)
    axis.set_xticklabels([label.title() for label in confusion_matrix.columns], rotation=0)
    axis.set_yticklabels([label.title() for label in confusion_matrix.index], rotation=0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    """Generate the confusion-matrix image from the evaluation CSV."""
    arguments = parse_arguments()
    confusion_matrix = load_confusion_matrix(arguments.input.resolve())
    output_path = arguments.output.resolve()
    save_confusion_matrix(confusion_matrix, output_path)
    print(f"Confusion matrix saved to: {output_path}")


if __name__ == "__main__":
    main()
