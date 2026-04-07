"""Simple CLI wrapper for training and prediction."""

from __future__ import annotations

import argparse
import sys

from predict import format_prediction_report, predict_event
from train_model import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tamil sentence event classification application."
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("train", help="Train models using dataset.csv")

    predict_parser = subparsers.add_parser("predict", help="Predict from a Tamil sentence")
    predict_parser.add_argument("--text", type=str, required=True, help="Tamil sentence")

    return parser.parse_args()


def ensure_utf8_output() -> None:
    """Make Tamil text printable in Windows terminals when possible."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    ensure_utf8_output()
    args = parse_args()

    if args.command == "train":
        train()
        return

    if args.command == "predict":
        result = predict_event(args.text)
        print(format_prediction_report(result))
        return

    print("Choose one command:")
    print("1. python app.py train")
    print('2. python app.py predict --text "தமிழ்நாட்டில் கடும் மழை"')


if __name__ == "__main__":
    main()
