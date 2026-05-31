import logging
import open3d as o3d
from depth_predictor import predict_depth
from image_loader import load_image
from model_loader import load_depth_pro
from pointcloud_builder import build_point_cloud

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def clean_point_cloud(pcd):
    try:
        initial_count = len(pcd.points)

        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        removed = initial_count - len(pcd.points)
        logger.info(f"Removed {removed} outliers ({initial_count} -> {len(pcd.points)})")

        pcd.estimate_normals()
        logger.info("Normals estimated")

        pcd = pcd.voxel_down_sample(voxel_size=0.005)
        logger.info(f"Final point count after voxel downsampling: {len(pcd.points)}")

        return pcd
    except Exception as e:
        logger.error(f"Point cloud cleaning XXXX: {e}")
        raise


if __name__ == "__main__":
    try:
        img = load_image("test.jpg")
        model, transform = load_depth_pro()
        depth, focal = predict_depth(model, transform, img)
        pcd = build_point_cloud(depth, img, focal)
        pcd = clean_point_cloud(pcd)
        logger.info("point cloud cleaned successfully")
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise
