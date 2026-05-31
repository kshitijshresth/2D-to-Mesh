import os
import logging
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_image(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV XXXXXXXX: {path}")

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    logger.info(f"Loaded image {path}: shape={rgb_image.shape}, dtype={rgb_image.dtype}")
    return rgb_image


if __name__ == "__main__":
    try:
        img = load_image("test.jpg")
        logger.info(f"Success: {img.shape}")
    except Exception as e:
        logger.error(f"Failed XXXXXXX: {e}")
        raise
