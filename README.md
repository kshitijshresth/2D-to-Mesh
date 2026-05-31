# 2D-to-Mesh

A Python pipeline that converts a single RGB image into a textured 3D mesh using deep monocular depth estimation and surface reconstruction.

## What it does

1. Loads an image with OpenCV.
2. Predicts per-pixel depth with Intel's MiDaS (DPT_Large) on CUDA.
3. Projects pixels into 3D space using the predicted depth map.
4. Cleans the point cloud (outlier removal, normal estimation, voxel downsampling).
5. Reconstructs a watertight mesh with Poisson surface reconstruction.
6. Decimates and removes floating fragments.
7. Exports the final mesh as a `.obj` file.

## Install

Requires Python 3.10+ and a CUDA-capable GPU.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python open3d timm numpy
```

## Example usage

Run the full pipeline from the command line:

```bash
python pipeline.py --input test.jpg --output output.obj --model DPT_Large
```

Or run individual stages:

```bash
python image_loader.py
python depth_predictor.py
python visualize_depth.py
python pointcloud_builder.py
python pointcloud_cleaner.py
python mesh_builder.py
python exporter.py
```

## Architecture

The pipeline is split into small, single-responsibility modules that pass data forward like a conveyor belt:

- `image_loader.py` handles file I/O and OpenCV color conversion.
- `model_loader.py` pulls MiDaS from torch.hub, moves it to CUDA, and quantizes to FP16.
- `depth_predictor.py` runs the model in `torch.inference_mode()`, resizes the output back to the original image dimensions, and returns a CPU float32 depth map.
- `pointcloud_builder.py` unprojects every pixel into `(X, Y, Z)` using a simple pinhole camera model with an estimated focal length, then packs the result into an Open3D `PointCloud`.
- `pointcloud_cleaner.py` applies statistical outlier removal, estimates normals, and voxel-downsamples to keep the cloud manageable.
- `mesh_builder.py` uses Poisson reconstruction to get a watertight mesh, simplifies it with quadric decimation to ~50k triangles, and prunes tiny disconnected fragments.
- `exporter.py` writes the mesh to `.obj`, reloads it, and asserts the triangle count matches to guard against silent corruption.
- `pipeline.py` ties everything together with `argparse`, section logging, dual console/file logging, and total runtime measurement.

<img width="1280" height="960" alt="image" src="https://github.com/user-attachments/assets/cd4cc3b4-75dd-4cec-ab34-de945c4b0824" />
When using MIDAS
<img width="1280" height="960" alt="image" src="https://github.com/user-attachments/assets/c04553df-8c24-4b5c-9a3f-38c1d2d29960" />
When using DEPTH PRO
<img width="1280" height="960" alt="image" src="https://github.com/user-attachments/assets/980c9a05-4a9c-43e6-8d16-b2443724337d" />


