"""Run a descriptive group pose-agreement baseline on one video."""
import argparse
import csv
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

from pose import draw_poses, extract_poses
from synchronization import frame_disagreement

ROOT = Path(__file__).resolve().parent


def analyze(video, weights, output, max_frames=None):
    if not video.is_file() or not weights.is_file():
        raise ValueError("Input video and local detector weights must exist")
    cap = cv2.VideoCapture(str(video))
    writer = None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not cap.isOpened() or not np.isfinite(fps) or fps <= 0 or min(width, height) <= 0:
            raise ValueError("Cannot open video or read valid dimensions/FPS")
        detector = YOLO(str(weights))
        output.mkdir(parents=True, exist_ok=False)
        writer = cv2.VideoWriter(str(output / "overlay.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("Cannot open MP4 video writer")
        rows = []
        with mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1,
                                    min_detection_confidence=0.5) as estimator:
            with (output / "scores.csv").open("w", newline="") as handle:
                csv_writer = csv.writer(handle)
                csv_writer.writerow(["frame", "time_seconds", "detections", "poses",
                                     "valid_poses", "valid_pairs", "disagreement"])
                while max_frames is None or len(rows) < max_frames:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    boxes, poses = extract_poses(frame, detector, estimator)
                    score, valid_poses, pairs = frame_disagreement(poses)
                    index = len(rows)
                    time = index / fps
                    csv_writer.writerow([index, time, len(boxes), len(poses), valid_poses,
                                         pairs, score if np.isfinite(score) else ""])
                    rows.append((time, score))
                    draw_poses(frame, boxes, poses, mp.solutions.pose.POSE_CONNECTIONS)
                    label = f"Disagreement: {score:.3f}" if np.isfinite(score) else "Disagreement: unavailable"
                    cv2.putText(frame, f"{label} | valid pairs: {pairs}", (10, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                    writer.write(frame)
                    if len(rows) % 30 == 0:
                        print(f"Processed {len(rows)} frames", flush=True)
        if not rows:
            raise ValueError("Video contained no decodable frames")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    times, scores = np.asarray(rows).T
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, scores, linewidth=1)
    ax.set(xlabel="Time (seconds)", ylabel="Mean distance (torso lengths)",
           title="Same-time group pose disagreement (lower = closer poses)")
    ax.grid(alpha=0.25)
    if not np.isfinite(scores).any():
        ax.text(0.5, 0.5, "No frames with sufficient valid poses", transform=ax.transAxes,
                ha="center")
    fig.tight_layout()
    fig.savefig(output / "disagreement.png", dpi=150)
    plt.close(fig)
    print(f"Saved {len(rows)} frames to {output}. "
          f"Frames with a valid score: {np.isfinite(scores).sum()}/{len(rows)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", nargs="?", type=Path, default=ROOT / "data/raw/input_dance_video.mp4")
    parser.add_argument("--weights", type=Path, default=ROOT / "yolov8n.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--max-frames", type=int, help="Optional prefix for a smoke test")
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    try:
        analyze(args.video, args.weights, args.output, args.max_frames)
    except (ValueError, RuntimeError, FileExistsError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
