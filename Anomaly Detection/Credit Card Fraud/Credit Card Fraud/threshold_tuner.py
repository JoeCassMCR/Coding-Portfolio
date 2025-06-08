"""
threshold_tuner.py

Script to tune probability threshold on validation set for minimum expected cost.
"""

import argparse
import logging
import pickle
from sklearn.model_selection import train_test_split
from data_loader import load_data
from preprocessing import preprocess_data, split_data
from evaluation import tune_threshold
from config import DEFAULT_MODEL_PATH, RANDOM_STATE, TEST_SIZE

# Configure logging
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATEFMT)
LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Tune probability threshold.")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH, help="Path to trained model.")
    parser.add_argument("--cost-fp", type=float, default=1.0, help="Cost of false positive.")
    parser.add_argument("--cost-fn", type=float, default=10.0, help="Cost of false negative.")
    return parser.parse_args()


def main():
    args = parse_args()
    # Load data
    df = load_data()
    X, y = preprocess_data(df)
    # Split off a validation set
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

    # Load model
    with open(args.model_path, 'rb') as f:
        model = pickle.load(f)

    best_threshold = tune_threshold(model, X_val, y_val, cost_fp=args.cost_fp, cost_fn=args.cost_fn)
    LOGGER.info(f"Best threshold: {best_threshold}")


if __name__ == '__main__':
    main()
