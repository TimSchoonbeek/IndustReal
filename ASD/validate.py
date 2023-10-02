"""
Runs the validation script with the correct hyperparameters and outputs metrics.

author: Tim Houben
email: timhouben@gmail.com
date: 25/06/2023
"""

import argparse
import os
from ultralytics import YOLO

def setup_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', default='/', type=str, help='Input folder for the images to make qualitative results of')
    parser.add_argument('--model_path', default='pretrained/yolov8l.pt', type=str, help='Path to the weights of the model')
    parser.add_argument('--data_path', default='datasets/synthetic_rgb_mb_16.yaml', type=str, help='Path to the weights of the model')
    parser.add_argument('--phase', default='val', type=str, help='train/val/test')
    parser.add_argument('--get_metrics', action='store_true', help='When set the metrics are drawn')

    return parser.parse_args()

def create_dir(dir: str):
    """Create a directory if one does not exist.

    Args:
        dir (str): Directory to be created.
    """
    if not os.path.exists(dir):
        os.makedirs(dir)


if __name__ == "__main__":
    args = setup_options()

    weights_path = os.path.join(args.base_folder, args.model_path)
    data_path = os.path.join(args.base_folder, args.data_path)

    model = YOLO(weights_path) #load model

    # Validate the model
    metrics = model.val(data=data_path, save_json=True, split=args.phase)  # no arguments needed, dataset and settings remembered

    if args.get_metrics:
        print(metrics.box.map)    # map50-95
        print(metrics.box.map50)  # map50
        print(metrics.box.map75)  # map75
        print(metrics.box.maps)   # a list contains map50-95 of each category