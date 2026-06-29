#!/usr/bin/env python3
"""Run 6DRepNet on a recorded video and save an annotated result video."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from face_detection import RetinaFace
from PIL import Image
from torchvision import transforms


ROOT_DIR = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT_DIR / "sixdrepnet"))

from model import SixDRepNet  # noqa: E402
import utils  # noqa: E402


TRANSFORM = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 6DRepNet on a recorded video.")
    parser.add_argument("--input", required=True, help="Input video path.")
    parser.add_argument("--output", required=True, help="Output annotated video path.")
    parser.add_argument("--csv", default="", help="Optional per-frame prediction CSV path.")
    parser.add_argument("--snapshot", required=True, help="6DRepNet checkpoint path.")
    parser.add_argument("--gpu", default=0, type=int, help="CUDA device id, or -1 for CPU model inference.")
    parser.add_argument("--score_thresh", default=0.95, type=float, help="RetinaFace confidence threshold.")
    parser.add_argument("--max_size", default=1280, type=int, help="Downscale long side for faster processing; 0 disables.")
    parser.add_argument("--fourcc", default="mp4v", help="Output video FOURCC, e.g. mp4v or XVID.")
    parser.add_argument("--progress_interval", default=30, type=int, help="Print progress every N frames.")
    return parser.parse_args()


def load_model(snapshot: Path, device: torch.device) -> SixDRepNet:
    model = SixDRepNet(
        backbone_name="RepVGG-B1g2",
        backbone_file="",
        deploy=True,
        pretrained=False,
    )
    state = torch.load(snapshot, map_location="cpu")
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    model.to(device)
    model.eval()
    return model


def resize_for_processing(frame, max_size: int):
    if max_size <= 0:
        return frame, 1.0
    h, w = frame.shape[:2]
    long_side = max(h, w)
    if long_side <= max_size:
        return frame, 1.0
    scale = max_size / float(long_side)
    resized = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    csv_path = Path(args.csv).expanduser().resolve() if args.csv else None
    snapshot = Path(args.snapshot).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    if not snapshot.exists():
        raise FileNotFoundError(f"Checkpoint not found: {snapshot}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path:
        csv_path.parent.mkdir(parents=True, exist_ok=True)

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        detector_gpu = args.gpu
    else:
        device = torch.device("cpu")
        detector_gpu = -1

    print(f"[6DRepNet video] input={input_path}")
    print(f"[6DRepNet video] output={output_path}")
    print(f"[6DRepNet video] csv={csv_path if csv_path else '(disabled)'}")
    print(f"[6DRepNet video] snapshot={snapshot}")
    print(f"[6DRepNet video] device={device}")

    model = load_model(snapshot, device)
    detector = RetinaFace(gpu_id=detector_gpu)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise OSError(f"Cannot open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    first_ok, first_frame = cap.read()
    if not first_ok or first_frame is None:
        raise OSError(f"Cannot read first frame from: {input_path}")
    proc_first, scale = resize_for_processing(first_frame, args.max_size)
    out_h, out_w = proc_first.shape[:2]

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*args.fourcc[:4]),
        fps,
        (out_w, out_h),
    )
    if not writer.isOpened():
        raise OSError(f"Cannot open output video writer: {output_path}")

    print(
        "[6DRepNet video] source=%dx%d %.3f fps, frames=%d, output=%dx%d, scale=%.4f"
        % (src_w, src_h, fps, frame_count, out_w, out_h, scale)
    )

    csv_file = None
    csv_writer = None
    if csv_path:
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["frame", "time_sec", "face_index", "score", "x_min", "y_min", "x_max", "y_max", "yaw", "pitch", "roll"])

    def process_frame(frame, frame_idx: int) -> int:
        frame, _ = resize_for_processing(frame, args.max_size)
        faces = detector(frame)
        used_faces = 0

        for face_idx, (box, _landmarks, score) in enumerate(faces):
            if score < args.score_thresh:
                continue

            x_min, y_min, x_max, y_max = [int(v) for v in box[:4]]
            bbox_width = abs(x_max - x_min)
            bbox_height = abs(y_max - y_min)
            if bbox_width <= 2 or bbox_height <= 2:
                continue

            x_min = max(0, x_min - int(0.2 * bbox_height))
            y_min = max(0, y_min - int(0.2 * bbox_width))
            x_max = min(frame.shape[1], x_max + int(0.2 * bbox_height))
            y_max = min(frame.shape[0], y_max + int(0.2 * bbox_width))
            if x_max <= x_min or y_max <= y_min:
                continue

            crop = frame[y_min:y_max, x_min:x_max]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(crop_rgb).convert("RGB")
            img = TRANSFORM(img)
            img = img[None, :].to(device)

            with torch.no_grad():
                r_pred = model(img)
                euler = utils.compute_euler_angles_from_rotation_matrices(r_pred) * 180 / np.pi

            pitch = float(euler[:, 0].cpu()[0])
            yaw = float(euler[:, 1].cpu()[0])
            roll = float(euler[:, 2].cpu()[0])

            utils.plot_pose_cube(
                frame,
                yaw,
                pitch,
                roll,
                x_min + int(0.5 * (x_max - x_min)),
                y_min + int(0.5 * (y_max - y_min)),
                size=bbox_width,
            )
            label = f"yaw {yaw:+.1f} pitch {pitch:+.1f} roll {roll:+.1f}"
            cv2.putText(frame, label, (x_min, max(20, y_min - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

            if csv_writer:
                csv_writer.writerow([frame_idx, frame_idx / fps, face_idx, float(score), x_min, y_min, x_max, y_max, yaw, pitch, roll])
            used_faces += 1

        writer.write(frame)
        return used_faces

    start = time.time()
    processed = 0
    detected_frames = 0

    detected = process_frame(proc_first if scale != 1.0 else first_frame, 0)
    processed += 1
    detected_frames += 1 if detected > 0 else 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        detected = process_frame(frame, processed)
        detected_frames += 1 if detected > 0 else 0
        processed += 1
        if args.progress_interval > 0 and processed % args.progress_interval == 0:
            print(f"[6DRepNet video] processed {processed}/{frame_count or '?'} frames, detected_frames={detected_frames}")

    elapsed = time.time() - start
    cap.release()
    writer.release()
    if csv_file:
        csv_file.close()

    print(f"[6DRepNet video] done: frames={processed}, detected_frames={detected_frames}, elapsed={elapsed:.2f}s")
    print(f"[6DRepNet video] saved video: {output_path}")
    if csv_path:
        print(f"[6DRepNet video] saved csv: {csv_path}")


if __name__ == "__main__":
    main()
