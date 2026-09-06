from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path


def version(name):
    return importlib.metadata.version(name)


def verify_runtime(checkpoint=None, require_cuda=False):
    import cv2
    import numpy
    import PIL
    import PySide6
    import torch
    import torchvision
    import ultralytics
    from ultralytics import SAM
    from ultralytics.models.sam import SAM3SemanticPredictor

    packages = {
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "PySide6": PySide6.__version__,
        "Pillow": PIL.__version__,
        "numpy": numpy.__version__,
        "opencv-python": cv2.__version__,
    }
    print("Production runtime imports: OK")
    for name, package_version in packages.items():
        print(f"  {name}: {package_version}")
    print(f"  SAM3SemanticPredictor: {SAM3SemanticPredictor.__module__}.{SAM3SemanticPredictor.__name__}")

    cuda_available = bool(torch.cuda.is_available())
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    if require_cuda and not cuda_available:
        raise RuntimeError("CUDA was required but torch.cuda.is_available() is false.")

    if checkpoint is not None:
        checkpoint = Path(checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        if checkpoint.suffix.lower() != ".pt":
            raise ValueError("SAM3 checkpoint must be a .pt file.")
        # This exercises Ultralytics' real checkpoint load path. Only trusted
        # checkpoints should ever be supplied to this command.
        SAM(str(checkpoint))
        print(f"Checkpoint load: OK ({checkpoint})")


def main():
    parser = argparse.ArgumentParser(
        description="Verify the installed SAM3 AutoAnnotator production runtime."
    )
    parser.add_argument(
        "--checkpoint",
        help="Optional trusted SAM3 .pt checkpoint to load through Ultralytics.",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail unless PyTorch reports a CUDA device.",
    )
    args = parser.parse_args()
    verify_runtime(args.checkpoint, args.require_cuda)


if __name__ == "__main__":
    main()
