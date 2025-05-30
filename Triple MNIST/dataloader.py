"""
Loading and preprocessing of the Triple MNIST dataset.
"""

import os
import logging
from typing import Tuple
import numpy as np
from PIL import Image
from utils import IMAGE_SIZE

def load_images(folder: str, flatten: bool=False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load grayscale images from subfolders, resize to (IMAGE_SIZE x IMAGE_SIZE).

    Args:
        folder: path to directory containing subfolders named by integer labels
        flatten: return flat arrays if True

    Returns:
        images: array of shape (N, IMAGE_SIZE, IMAGE_SIZE) or (N, IMAGE_SIZE*IMAGE_SIZE)
        labels: array of shape (N,)
    """
    images, labels = [], []
    if not os.path.isdir(folder):
        logging.error(f"Folder not found: {folder}")
        raise FileNotFoundError(folder)

    for label_name in os.listdir(folder):
        label_path = os.path.join(folder, label_name)
        if os.path.isdir(label_path) and label_name.isdigit():
            for fname in os.listdir(label_path):
                fpath = os.path.join(label_path, fname)
                try:
                    img = Image.open(fpath).convert('L').resize((IMAGE_SIZE, IMAGE_SIZE))
                    arr = np.array(img)
                    if flatten:
                        arr = arr.flatten()
                    images.append(arr)
                    labels.append(int(label_name))
                except Exception as e:
                    logging.warning(f"Skipping file {fpath}: {e}")

    images_arr = np.array(images)
    labels_arr = np.array(labels)
    logging.info(f"Loaded {len(images_arr)} images from {folder}")
    return images_arr, labels_arr

def split_labels(labels: np.ndarray) -> np.ndarray:
    """
    Convert integer labels to array of 3 digits each.
    """
    return np.array([[int(d) for d in f"{lbl:03d}"] for lbl in labels])
