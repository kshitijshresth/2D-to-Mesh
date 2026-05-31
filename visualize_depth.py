import logging

import cv2
import numpy as np
from depth_predictor import predict_depth
from image_loader import load_image
from model_loader import load_depth_pro

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def save_depth_image(depth_np: np.ndarray, output_path: str):
    depth_min = depth_np.min()
    depth_max = depth_np.max()
    if depth_max - depth_min < 1e-6:
        normalized = np.zeros_like(depth_np, dtype=np.uint8)
    else:
        normalized = ((depth_np - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)

    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
    cv2.imwrite(output_path, colored)
    logger.info(f"Viz depth @ {output_path}: shape={colored.shape}")


if __name__ == "__main__":
    try:
        img = load_image("test.jpg")
        model, transform = load_depth_pro()
        depth, focal = predict_depth(model, transform, img)
        save_depth_image(depth, "depth_preview.png")
        logger.info("Saved depth preview")
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise
