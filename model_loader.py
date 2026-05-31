import logging
import os

import torch
import depth_pro
from depth_pro.depth_pro import DepthProConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINT_PATH = "depth_pro.pt"


def _ensure_checkpoint():
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found at {CHECKPOINT_PATH}. "
            "Download depth_pro.pt and place it in the project root."
        )


def load_depth_pro():
    try:
        logger.info("Loading DepthPro...")
        _ensure_checkpoint()

        config = DepthProConfig(
            patch_encoder_preset='dinov2l16_384',
            image_encoder_preset='dinov2l16_384',
            decoder_features=256,
            checkpoint_uri=CHECKPOINT_PATH,
            fov_encoder_preset='dinov2l16_384',
            use_fov_head=True,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, transform = depth_pro.create_model_and_transforms(
            config=config, device=device, precision=torch.float16
        )
        model = model.eval()

        vram_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        logger.info(f"VRAM used after loading: {vram_mb:.2f} MB")

        return model, transform
    except Exception as e:
        logger.error(f"Failed DepthPro: {e}")
        raise


def load_midas():
    return load_depth_pro()


if __name__ == "__main__":
    try:
        model, transform = load_depth_pro()
        logger.info("Model ready")
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise
