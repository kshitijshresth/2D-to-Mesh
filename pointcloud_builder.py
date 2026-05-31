import logging
import numpy as np
import open3d as o3d
from depth_predictor import predict_depth
from image_loader import load_image
from model_loader import load_midas

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_point_cloud(depth_np: np.ndarray, img_rgb: np.ndarray):
    try:
        h, w = depth_np.shape
        fx = fy = w * 1.2
        cx = w / 2.0
        cy = h / 2.0

        u, v = np.meshgrid(np.arange(w), np.arange(h))
        u = u.astype(np.float32)
        v = v.astype(np.float32)
        z = depth_np.astype(np.float32)

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
        colors = img_rgb.reshape(-1, 3).astype(np.float64) / 255.0

        total_before = points.shape[0]
        mask = points[:, 2] >= 0.01
        points = points[mask]
        colors = colors[mask]
        total_after = points.shape[0]

        logger.info(f"Points before filtering: {total_before}, after filtering: {total_after}")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        bbox = pcd.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()
        logger.info(f"Bounding box dimensions: X={extent[0]:.4f} Y={extent[1]:.4f} Z={extent[2]:.4f}")

        return pcd
    except Exception as e:
        logger.error(f"Point cloud build XXXX: {e}")
        raise


if __name__ == "__main__":
    try:
        img = load_image("test.jpg")
        model, transform = load_midas()
        depth = predict_depth(model, transform, img)
        pcd = build_point_cloud(depth, img)
        logger.info("Point cloud built successfully")
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise
