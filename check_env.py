import importlib

libraries = {
    "torch": "torch",
    "cv2": "cv2",
    "open3d": "open3d",
    "timm": "timm",
    "numpy": "numpy",
}

for name, module_name in libraries.items():
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, "__version__", "unknown")
        print(f"OK: {name} {version}")
    except ImportError as e:
        print(f"FAIL: {name} — {e}")

try:
    import torch
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"CUDA check failed: {e}")
