import argparse
import logging

import open3d as o3d

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def view_mesh(path: str):
    logger.info(f"Loading mesh: {path}")
    mesh = o3d.io.read_triangle_mesh(path)
    if len(mesh.triangles) == 0:
        raise ValueError(f"Mesh has no triangles: {path}")
    logger.info(
        f"Mesh loaded — vertices: {len(mesh.vertices)}, triangles: {len(mesh.triangles)}"
    )
    logger.info("Opening 3D viewer... (close window to exit)")
    o3d.visualization.draw_geometries([mesh])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View a mesh in Open3D")
    parser.add_argument("--mesh", default="output_mesh.obj", help="Path to .obj or .glb mesh")
    args = parser.parse_args()
    view_mesh(args.mesh)
