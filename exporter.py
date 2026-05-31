import logging
import os
import numpy as np
import open3d as o3d
import trimesh
from depth_predictor import predict_depth
from image_loader import load_image
from mesh_builder import build_mesh
from model_loader import load_depth_pro
from pointcloud_builder import build_point_cloud
from pointcloud_cleaner import clean_point_cloud

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def export_mesh(mesh, output_path: str):
    try:
        original_triangles = len(mesh.triangles)
        original_vertices = len(mesh.vertices)

        o3d.io.write_triangle_mesh(output_path, mesh)

        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"File size: {file_size_mb:.2f} MB")

        loaded_mesh = o3d.io.read_triangle_mesh(output_path)
        loaded_triangles = len(loaded_mesh.triangles)
        loaded_vertices = len(loaded_mesh.vertices)

        logger.info(f"Saved file vertices: {loaded_vertices}, triangles: {loaded_triangles}")

        if loaded_triangles != original_triangles:
            raise AssertionError(
                f"Triangle count mismatch: original={original_triangles}, loaded={loaded_triangles}"
            )

        return output_path
    except Exception as e:
        logger.error(f"Mesh export failed: {e}")
        raise


def export_glb(mesh, output_path: str):
    try:
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        tm = trimesh.Trimesh(vertices=vertices, faces=triangles)
        tm.export(output_path)

        if not os.path.exists(output_path):
            raise AssertionError(f"GLB file not found after save: {output_path}")

        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"GLB file size: {file_size_mb:.2f} MB")

        return output_path
    except Exception as e:
        logger.error(f"GLB export failed: {e}")
        raise


if __name__ == "__main__":
    try:
        img = load_image("test.jpg")
        model, transform = load_depth_pro()
        depth, focal = predict_depth(model, transform, img)
        pcd = build_point_cloud(depth, img, focal)
        pcd = clean_point_cloud(pcd)
        mesh = build_mesh(pcd)
        export_mesh(mesh, "output_mesh.obj")
        logger.info("Pipeline complete")
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise
