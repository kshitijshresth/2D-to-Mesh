import logging
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_midas():
    try:
        logger.info("Loading MiDaS DPT_Large via torch.hub...")
        model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        transform = midas_transforms.dpt_transform

        model = model.cuda().half().eval()

        vram_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        logger.info(f"VRAM used after loading: {vram_mb:.2f} MB")

        return model, transform
    except Exception as e:
        logger.error(f"Failed MiDaS: {e}")
        raise


if __name__ == "__main__":
    try:
        model, transform = load_midas()
        logger.info("Model ready")
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise
