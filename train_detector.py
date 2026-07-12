"""Fine-tune YOLOv8 on the Roboflow football-players-detection dataset.

Requires a free Roboflow account and API key (ROBOFLOW_API_KEY env var):
https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc

Not needed to run the pipeline: the models/ folder already carries
checkpoints trained on this exact dataset (from the
Darkmyter/Football-Players-Tracking project). Use this script to retrain
on a newer dataset version or a different base model. Realistically this
needs a GPU -- on CPU, 100 epochs would take days.
"""

import argparse
import os

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="yolov8m.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--dataset-version", type=int, default=12)
    args = parser.parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("set ROBOFLOW_API_KEY (free key from roboflow.com)")

    from roboflow import Roboflow  # optional dep, only needed for training

    rf = Roboflow(api_key=api_key)
    project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
    dataset = project.version(args.dataset_version).download("yolov8", location="datasets/football-players")

    model = YOLO(args.base_model)
    model.train(
        data=f"{dataset.location}/data.yaml",
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
    )
    print("best weights under runs/detect/train*/weights/best.pt -- copy to models/")


if __name__ == "__main__":
    main()
