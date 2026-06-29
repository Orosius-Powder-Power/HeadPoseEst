#!/usr/bin/env python3
"""Evaluate TokenHPE on AFLW2000 with the author's metrics.

This helper keeps the original TokenHPE model, dataset, preprocessing, and
metric logic, while adding robust paths and checkpoint handling for this repo.
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms


TOKENHPE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOKENHPE_ROOT))

import datasets  # noqa: E402
import utils  # noqa: E402
from model import TokenHPE  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TokenHPE on AFLW2000.")
    parser.add_argument(
        "--data_dir",
        default=str(TOKENHPE_ROOT.parent / "6DRepNet/datasets/AFLW2000"),
        help="AFLW2000 directory containing jpg/mat files.",
    )
    parser.add_argument(
        "--filename_list",
        default=str(TOKENHPE_ROOT.parent / "6DRepNet/datasets/AFLW2000/files_filtered_99.txt"),
        help="AFLW2000 filename list. Use the 1969-sample filtered list for paper-style evaluation.",
    )
    parser.add_argument(
        "--model_path",
        default=str(TOKENHPE_ROOT / "checkpoints/TokenHPEv1-ViTB-224_224-lyr3.tar"),
        help="TokenHPE trained checkpoint .tar/.pth, or a directory containing archive/ from an extracted torch zip.",
    )
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--gpu", default=0, type=int, help="CUDA device id.")
    parser.add_argument(
        "--num_ori_tokens",
        default=9,
        type=int,
        choices=(9, 11),
        help="Number of TokenHPE orientation tokens: 9 for v1, 11 for v2.",
    )
    parser.add_argument(
        "--allow_repack",
        action="store_true",
        help="If model_path is an extracted checkpoint directory, repack it to a torch-loadable .tar zip.",
    )
    return parser.parse_args()


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def resolve_checkpoint(model_path: str, allow_repack: bool) -> Path:
    path = Path(model_path).expanduser().resolve()
    if path.is_file():
        return path

    if path.is_dir() and (path / "archive/data.pkl").exists():
        repacked = path.with_suffix(".tar")
        if repacked.exists():
            return repacked
        if not allow_repack:
            raise FileNotFoundError(
                f"Found extracted checkpoint directory: {path}\n"
                f"Please run with --allow_repack, or place the original .tar at {repacked}"
            )

        print(f"[TokenHPE eval] repacking extracted checkpoint to {repacked}")
        with zipfile.ZipFile(repacked, "w", compression=zipfile.ZIP_STORED) as zf:
            for item in sorted((path / "archive").rglob("*")):
                if not item.is_file() or item.name.endswith(":Zone.Identifier"):
                    continue
                zf.write(item, item.relative_to(path).as_posix())
        return repacked

    raise FileNotFoundError(f"Checkpoint not found or unsupported: {path}")


def load_model(model_path: Path, device: torch.device, num_ori_tokens: int) -> TokenHPE:
    model = TokenHPE(num_ori_tokens=num_ori_tokens, depth=3, heads=8, embedding="sine", dim=128)
    checkpoint = torch_load(model_path)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    filename_list = Path(args.filename_list).expanduser().resolve()
    checkpoint = resolve_checkpoint(args.model_path, args.allow_repack)

    if not data_dir.exists():
        raise FileNotFoundError(f"AFLW2000 data_dir not found: {data_dir}")
    if not filename_list.exists():
        raise FileNotFoundError(f"filename_list not found: {filename_list}")
    if not torch.cuda.is_available():
        raise RuntimeError("TokenHPE original utils expect CUDA tensors. Please run with a CUDA-enabled PyTorch.")

    device = torch.device(f"cuda:{args.gpu}")
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    print(f"[TokenHPE eval] root={TOKENHPE_ROOT}")
    print(f"[TokenHPE eval] data_dir={data_dir}")
    print(f"[TokenHPE eval] filename_list={filename_list}")
    print(f"[TokenHPE eval] checkpoint={checkpoint}")
    print(f"[TokenHPE eval] num_ori_tokens={args.num_ori_tokens}")
    print(f"[TokenHPE eval] device={device}")

    transformations = transforms.Compose(
        [
            transforms.Resize(250),
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
        pin_memory=True,
    )

    model = load_model(checkpoint, device, args.num_ori_tokens)
    print("[TokenHPE eval] model weight loaded")

    total = 0
    yaw_error = pitch_error = roll_error = 0.0
    v1_err = v2_err = v3_err = 0.0

    with torch.no_grad():
        for images, r_label, cont_labels, _name in test_loader:
            images = images.to(device, non_blocking=True)
            total += cont_labels.size(0)

            r_gt = r_label
            y_gt_deg = cont_labels[:, 0].float() * 180 / np.pi
            p_gt_deg = cont_labels[:, 1].float() * 180 / np.pi
            r_gt_deg = cont_labels[:, 2].float() * 180 / np.pi

            r_pred, _ori_9_d = model(images)
            euler = utils.compute_euler_angles_from_rotation_matrices(r_pred) * 180 / np.pi
            p_pred_deg = euler[:, 0].cpu()
            y_pred_deg = euler[:, 1].cpu()
            r_pred_deg = euler[:, 2].cpu()

            r_pred = r_pred.cpu()
            v1_err += torch.sum(torch.acos(torch.clamp(torch.sum(r_gt[:, 0] * r_pred[:, 0], 1), -1, 1)) * 180 / np.pi)
            v2_err += torch.sum(torch.acos(torch.clamp(torch.sum(r_gt[:, 1] * r_pred[:, 1], 1), -1, 1)) * 180 / np.pi)
            v3_err += torch.sum(torch.acos(torch.clamp(torch.sum(r_gt[:, 2] * r_pred[:, 2], 1), -1, 1)) * 180 / np.pi)

            pitch_error += torch.sum(
                torch.min(
                    torch.stack(
                        (
                            torch.abs(p_gt_deg - p_pred_deg),
                            torch.abs(p_pred_deg + 360 - p_gt_deg),
                            torch.abs(p_pred_deg - 360 - p_gt_deg),
                            torch.abs(p_pred_deg + 180 - p_gt_deg),
                            torch.abs(p_pred_deg - 180 - p_gt_deg),
                        )
                    ),
                    0,
                )[0]
            )
            yaw_error += torch.sum(
                torch.min(
                    torch.stack(
                        (
                            torch.abs(y_gt_deg - y_pred_deg),
                            torch.abs(y_pred_deg + 360 - y_gt_deg),
                            torch.abs(y_pred_deg - 360 - y_gt_deg),
                            torch.abs(y_pred_deg + 180 - y_gt_deg),
                            torch.abs(y_pred_deg - 180 - y_gt_deg),
                        )
                    ),
                    0,
                )[0]
            )
            roll_error += torch.sum(
                torch.min(
                    torch.stack(
                        (
                            torch.abs(r_gt_deg - r_pred_deg),
                            torch.abs(r_pred_deg + 360 - r_gt_deg),
                            torch.abs(r_pred_deg - 360 - r_gt_deg),
                            torch.abs(r_pred_deg + 180 - r_gt_deg),
                            torch.abs(r_pred_deg - 180 - r_gt_deg),
                        )
                    ),
                    0,
                )[0]
            )

    mae = (yaw_error + pitch_error + roll_error) / (total * 3)
    vmae = (v1_err + v2_err + v3_err) / (total * 3)
    print(f"Samples: {total}")
    print(
        "Yaw: %.4f, Pitch: %.4f, Roll: %.4f, MAE: %.4f"
        % (yaw_error / total, pitch_error / total, roll_error / total, mae)
    )
    print("Vec1: %.4f, Vec2: %.4f, Vec3: %.4f, VMAE: %.4f" % (v1_err / total, v2_err / total, v3_err / total, vmae))
    print("Paper reference on AFLW2000: MAE 4.81, VMAE 6.09")


if __name__ == "__main__":
    main()
