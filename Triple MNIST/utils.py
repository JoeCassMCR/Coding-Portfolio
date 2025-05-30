"""
Utility functions, constants, and logging configuration.
"""

import os
import logging
from typing import Tuple, List
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping


# Configuration constants
IMAGE_SIZE = 84
BATCH_SIZE = 32
EPOCHS = 100
NOISE_DIM = 100
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def ensure_directory(path: str) -> None:
    """
    Create directory if it doesn't exist.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        logging.info(f"Created directory: {path}")

def save_figure(fig, out_path: str) -> None:
    """
    Save a Matplotlib figure to disk.
    """
    ensure_directory(os.path.dirname(out_path))
    fig.savefig(out_path)
    logging.info(f"Saved figure: {out_path}")
