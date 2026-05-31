import logging
import time
import numpy as np
import torch
import torch.nn.functional as F
from image_loader import load_image
from model_loader import load_midas

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def predict_depth(model, transform, img_rgb: np.ndarray):
    try:
        h, w = img_rgb.shape[:2]

        input_batch = transform(img_rgb).to("cuda")
        if input_batch.dim() == 3:
            input_batch = input_batch.unsqueeze(0)
        input_batch = input_batch.half()

        start = time.perf_counter()
        with torch.inference_mode():
            prediction = model(input_batch)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        prediction = F.interpolate(
            prediction.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth = prediction.cpu().numpy().astype(np.float32)

        torch.cuda.empty_cache()

        logger.info(f"Inference time: {elapsed_ms:.2f} ms")
        logger.info(f"Depth min={depth.min():.4f} max={depth.max():.4f} mean={depth.mean():.4f}")

        return depth
    except Exception as e:
        logger.error(f"Depth prediction XXXXXXX: {e}")
        raise


if __name__ == "__main__":
    try:
        img = load_image("test.jpg")
        model, transform = load_midas()
        depth = predict_depth(model, transform, img)
        logger.info(f"Depth prediction complete. Shape: {depth.shape}")
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise
