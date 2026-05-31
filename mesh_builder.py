import logging
import numpy as np
import open3d as o3d
from depth_predictor import predict_depth
from image_loader import load_image
from model_loader import load_midas
from pointcloud_builder import build_point_cloud
from pointcloud_cleaner import clean_point_cloud

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_mesh(pcd):
    try:
        logger.info("Running Poisson surface reconstruction")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9
        )
        raw_triangles = len(mesh.triangles)
        logger.info(f"Raw mesh triangles: {raw_triangles}")

        logger.info("Running quadric decimation")
        target = min(50000, raw_triangles)
        mesh = mesh.simplify_quadric_decimation(target)
        decimated_triangles = len(mesh.triangles)
        logger.info(f"Triangles after decimation: {decimated_triangles}")

        logger.info("Removing disconnects")
        with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug
        ):
            triangle_clusters, cluster_n_triangles, cluster_area = (
                mesh.cluster_connected_triangles()
            )

        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        total_triangles = len(mesh.triangles)
        threshold = total_triangles * 0.01

        triangles_to_remove = cluster_n_triangles[triangle_clusters] < threshold
        mesh.remove_triangles_by_mask(triangles_to_remove)
        mesh.remove_unreferenced_vertices()

        removed_components = np.sum(cluster_n_triangles < threshold)
        logger.info(f"Removed {removed_components} small disconnected components")
        logger.info(f"Final mesh triangles: {len(mesh.triangles)}")

        return mesh
    except Exception as e:
        logger.error(f"Mesh build failed: {e}")
        raise


if __name__ == "__main__":
    try:
        img = load_image("test.jpg")
        model, transform = load_midas()
        depth = predict_depth(model, transform, img)
        pcd = build_point_cloud(depth, img)
        pcd = clean_point_cloud(pcd)
        mesh = build_mesh(pcd)
        logger.info("Mesh built successfully")
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise
