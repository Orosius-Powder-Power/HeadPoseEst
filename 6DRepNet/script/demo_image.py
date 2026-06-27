#!/usr/bin/env python3
"""Run a deterministic 6DRepNet image demo on one AFLW2000 sample."""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


ROOT_DIR = Path(__file__).resolve().parents[1]
SIXD_DIR = ROOT_DIR / "sixdrepnet"
sys.path.insert(0, str(SIXD_DIR))

import utils  # noqa: E402
from model import SixDRepNet  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Run 6DRepNet on one AFLW2000 image.")
    parser.add_argument("--snapshot", default=str(ROOT_DIR / "checkpoints" / "6DRepNet_300W_LP_AFLW2000.pth"))
    parser.add_argument("--data_dir", default=str(ROOT_DIR / "datasets" / "AFLW2000"))
    parser.add_argument("--sample", default="", help="Sample id without suffix, e.g. image00002.")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device id. Use -1 for CPU.")
    parser.add_argument("--output", default=str(ROOT_DIR / "output" / "demo" / "aflw2000_demo.png"))
    return parser.parse_args()


def choose_sample(data_dir: Path, sample: str) -> str:
    if sample:
        return sample

    files_txt = data_dir / "files.txt"
    if files_txt.exists():
        for line in files_txt.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                return line.lstrip("./")

    first_image = sorted(data_dir.glob("*.jpg"))
    if not first_image:
        raise FileNotFoundError(f"No .jpg files found in {data_dir}")
    return first_image[0].stem


def load_model(snapshot: Path, device: torch.device) -> SixDRepNet:
    model = SixDRepNet(
        backbone_name="RepVGG-B1g2",
        backbone_file="",
        deploy=True,
        pretrained=False,
    )
    saved_state_dict = torch.load(snapshot, map_location="cpu")
    if "model_state_dict" in saved_state_dict:
        model.load_state_dict(saved_state_dict["model_state_dict"])
    else:
        model.load_state_dict(saved_state_dict)
    model.to(device)
    model.eval()
    return model


def crop_face_from_mat(image: Image.Image, mat_path: Path) -> tuple[Image.Image, tuple[int, int, int, int]]:
    pt2d = utils.get_pt2d_from_mat(str(mat_path))

    x_min = min(pt2d[0, :])
    y_min = min(pt2d[1, :])
    x_max = max(pt2d[0, :])
    y_max = max(pt2d[1, :])

    k = 0.20
    x_min -= 2 * k * abs(x_max - x_min)
    y_min -= 2 * k * abs(y_max - y_min)
    x_max += 2 * k * abs(x_max - x_min)
    y_max += 0.6 * k * abs(y_max - y_min)

    width, height = image.size
    box = (
        max(0, int(x_min)),
        max(0, int(y_min)),
        min(width, int(x_max)),
        min(height, int(y_max)),
    )
    return image.crop(box), box


def main():
    args = parse_args()
    snapshot = Path(args.snapshot)
    data_dir = Path(args.data_dir)
    output = Path(args.output)

    if not snapshot.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {snapshot}")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Missing AFLW2000 directory: {data_dir}")

    device = torch.device("cpu" if args.gpu < 0 or not torch.cuda.is_available() else f"cuda:{args.gpu}")
    sample = choose_sample(data_dir, args.sample)
    image_path = data_dir / f"{sample}.jpg"
    mat_path = data_dir / f"{sample}.mat"
    if not image_path.is_file() or not mat_path.is_file():
        raise FileNotFoundError(f"Missing sample pair: {image_path} / {mat_path}")

    transformations = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    model = load_model(snapshot, device)
    image = Image.open(image_path).convert("RGB")
    face, box = crop_face_from_mat(image, mat_path)
    tensor = transformations(face).unsqueeze(0).to(device)

    with torch.no_grad():
        R_pred = model(tensor)
        euler = utils.compute_euler_angles_from_rotation_matrices(R_pred) * 180 / np.pi

    pitch = float(euler[:, 0].cpu().item())
    yaw = float(euler[:, 1].cpu().item())
    roll = float(euler[:, 2].cpu().item())

    cv_image = cv2.imread(str(image_path))
    x_min, y_min, x_max, y_max = box
    bbox_width = max(80, abs(x_max - x_min))
    utils.plot_pose_cube(
        cv_image,
        yaw,
        pitch,
        roll,
        x_min + int(0.5 * (x_max - x_min)),
        y_min + int(0.5 * (y_max - y_min)),
        size=bbox_width,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), cv_image)
    print(f"Sample: {sample}")
    print(f"Device: {device}")
    print(f"Pitch: {pitch:.4f}, Yaw: {yaw:.4f}, Roll: {roll:.4f}")
    print(f"Saved demo visualization: {output}")


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
    main()
