import logging
import time

import numpy as np
import torch

from image_loader import load_image
from model_loader import load_depth_pro

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def predict_depth(model, transform, img_rgb: np.ndarray):
    try:
        device = next(model.parameters()).device

        image = transform(img_rgb)
        image = image.to(device).unsqueeze(0)

        start = time.perf_counter()
        with torch.inference_mode():
            prediction = model.infer(image)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        depth = prediction["depth"].squeeze().cpu().numpy().astype(np.float32)
        focallength_px = float(prediction["focallength_px"])

        torch.cuda.empty_cache()

        logger.info(f"Inference time: {elapsed_ms:.2f} ms")
        logger.info(f"Depth shape={depth.shape}")
        logger.info(f"Depth min={depth.min():.4f} max={depth.max():.4f} mean={depth.mean():.4f}")
        logger.info(f"Focal length (px): {focallength_px:.2f}")

        return depth, focallength_px
    except Exception as e:
        logger.error(f"Depth prediction XXXXXXX: {e}")
        raise


if __name__ == "__main__":
    try:
        img = load_image("test.jpg")
        model, transform = load_depth_pro()
        depth, focal = predict_depth(model, transform, img)
        logger.info(f"Depth prediction complete. Shape: {depth.shape}, focal: {focal:.2f}px")
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise
