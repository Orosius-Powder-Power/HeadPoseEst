#!/usr/bin/env python3
"""Evaluate 6DRepNet on AFLW2000 without requiring a GUI backend."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms


ROOT_DIR = Path(__file__).resolve().parents[1]
SIXD_DIR = ROOT_DIR / "sixdrepnet"
sys.path.insert(0, str(SIXD_DIR))

import datasets  # noqa: E402
import utils  # noqa: E402
from model import SixDRepNet  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate 6DRepNet on AFLW2000.")
    parser.add_argument("--snapshot", default=str(ROOT_DIR / "checkpoints" / "6DRepNet_300W_LP_AFLW2000.pth"))
    parser.add_argument("--data_dir", default=str(ROOT_DIR / "datasets" / "AFLW2000"))
    parser.add_argument("--filename_list", default=str(ROOT_DIR / "datasets" / "AFLW2000" / "files.txt"))
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device id. Use -1 for CPU.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    return parser.parse_args()


def ensure_aflw_file_list(data_dir: Path, filename_list: Path):
    if filename_list.is_file():
        return

    image_names = sorted(path.stem for path in data_dir.glob("*.jpg") if (data_dir / f"{path.stem}.mat").is_file())
    if not image_names:
        raise FileNotFoundError(f"No AFLW2000 .jpg/.mat pairs found in {data_dir}")
    filename_list.parent.mkdir(parents=True, exist_ok=True)
    filename_list.write_text("\n".join(image_names) + "\n", encoding="utf-8")
    print(f"Generated filename list: {filename_list} ({len(image_names)} samples)")


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


def wrapped_angle_error(gt_deg: torch.Tensor, pred_deg: torch.Tensor) -> torch.Tensor:
    return torch.min(
        torch.stack(
            (
                torch.abs(gt_deg - pred_deg),
                torch.abs(pred_deg + 360 - gt_deg),
                torch.abs(pred_deg - 360 - gt_deg),
                torch.abs(pred_deg + 180 - gt_deg),
                torch.abs(pred_deg - 180 - gt_deg),
            )
        ),
        0,
    )[0]


def main():
    args = parse_args()
    snapshot = Path(args.snapshot)
    data_dir = Path(args.data_dir)
    filename_list = Path(args.filename_list)

    if not snapshot.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {snapshot}")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Missing AFLW2000 directory: {data_dir}")
    ensure_aflw_file_list(data_dir, filename_list)

    device = torch.device("cpu" if args.gpu < 0 or not torch.cuda.is_available() else f"cuda:{args.gpu}")
    transformations = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    pose_dataset = datasets.getDataset("AFLW2000", str(data_dir), str(filename_list), transformations, train_mode=False)
    test_loader = torch.utils.data.DataLoader(
        dataset=pose_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    model = load_model(snapshot, device)

    total = 0
    yaw_error = torch.tensor(0.0)
    pitch_error = torch.tensor(0.0)
    roll_error = torch.tensor(0.0)

    with torch.no_grad():
        for images, _r_label, cont_labels, _name in test_loader:
            images = images.to(device)
            total += cont_labels.size(0)

            y_gt_deg = cont_labels[:, 0].float() * 180 / np.pi
            p_gt_deg = cont_labels[:, 1].float() * 180 / np.pi
            r_gt_deg = cont_labels[:, 2].float() * 180 / np.pi

            R_pred = model(images)
            euler = utils.compute_euler_angles_from_rotation_matrices(R_pred).cpu() * 180 / np.pi
            p_pred_deg = euler[:, 0]
            y_pred_deg = euler[:, 1]
            r_pred_deg = euler[:, 2]

            pitch_error += torch.sum(wrapped_angle_error(p_gt_deg, p_pred_deg))
            yaw_error += torch.sum(wrapped_angle_error(y_gt_deg, y_pred_deg))
            roll_error += torch.sum(wrapped_angle_error(r_gt_deg, r_pred_deg))

    yaw = float((yaw_error / total).item())
    pitch = float((pitch_error / total).item())
    roll = float((roll_error / total).item())
    mae = float(((yaw_error + pitch_error + roll_error) / (total * 3)).item())
    print(f"Device: {device}")
    print(f"Samples: {total}")
    print(f"Yaw: {yaw:.4f}, Pitch: {pitch:.4f}, Roll: {roll:.4f}, MAE: {mae:.4f}")
    print("Paper reference on AFLW2000: Yaw 3.63, Pitch 4.91, Roll 3.37, MAE 3.97")


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
    main()
