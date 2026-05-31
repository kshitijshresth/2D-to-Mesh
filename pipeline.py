import argparse
import logging
import sys
import time
import traceback
from depth_predictor import predict_depth
from exporter import export_mesh
from image_loader import load_image
from mesh_builder import build_mesh
from model_loader import load_midas
from pointcloud_builder import build_point_cloud
from pointcloud_cleaner import clean_point_cloud


def setup_logging():
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log", mode="w"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )
    return logging.getLogger(__name__)


def run_pipeline(args):
    logger = setup_logging()
    start = time.perf_counter()

    try:
        logger.info("Step [1/7]: Load Image")
        img = load_image(args.input)

        logger.info("Step [2/7]: Load MiDaS Model")
        model, transform = load_midas()

        logger.info("Step [3/7]: Depth Prediction")
        depth = predict_depth(model, transform, img)

        logger.info("Step [4/7]: Build Point Cloud")
        pcd = build_point_cloud(depth, img)

        logger.info("Step [5/7]: Clean Point Cloud")
        pcd = clean_point_cloud(pcd)

        logger.info("Step [6/7]: Build Mesh")
        mesh = build_mesh(pcd)

        logger.info("Step [7/7]: Export Mesh")
        export_mesh(mesh, args.output)

        elapsed = time.perf_counter() - start
        logger.info(f"Pipeline finished in {elapsed:.2f}s")
        return 0
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.error(traceback.format_exc())
        return 1


def main():
    parser = argparse.ArgumentParser(description="2D-to-Mesh pipeline")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--output", default="output.obj", help="Output mesh path")
    parser.add_argument(
        "--model",
        default="DPT_Large",
        choices=["DPT_Large", "MiDaS_small"],
        help="MiDaS model variant",
    )
    args = parser.parse_args()
    sys.exit(run_pipeline(args))


if __name__ == "__main__":
    main()
