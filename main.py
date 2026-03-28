"""
Agricultural Weed Detection - Local Testing Environment

Supports three input modes for testing the crop/weed detection model:
  - webcam:  Process webcam frames and upload at 1 FPS
  - video:   Process video frames and upload at 1 FPS
  - images:  Process directory images and upload every 5 seconds

Usage:
  python main.py --source webcam
  python main.py --source video --input drone_footage.mp4
  python main.py --source images --input ./plant_images/
  python main.py --source webcam --model new.pt --conf 0.2

Environment:
  IMAGE_UPLOAD_URL - Endpoint to send processed frames
                     (default: http://localhost:3001/upload)
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import requests
from ultralytics import YOLO

# Preferred colors for known semantic classes
SEMANTIC_COLORS = {
    "crop": (0, 200, 0),  # green
    "weed": (0, 0, 255),  # red
}

# Fallback for any other class
DEFAULT_COLOR = (255, 165, 0)  # orange (BGR)

DEFAULT_UPLOAD_URL = os.getenv("IMAGE_UPLOAD_URL", "http://localhost:3001/upload")


def resolve_class_name(class_names, cls_id):
    """Return model class name for a class index, with safe fallback."""
    if isinstance(class_names, dict):
        return str(class_names.get(cls_id, f"class_{cls_id}"))

    if isinstance(class_names, (list, tuple)) and 0 <= cls_id < len(class_names):
        return str(class_names[cls_id])

    return f"class_{cls_id}"


def class_color(label, cls_id):
    """Pick a display color for a class label."""
    semantic = SEMANTIC_COLORS.get(label.strip().lower())
    if semantic is not None:
        return semantic

    # Deterministic fallback color per class id
    return (
        (37 * cls_id + 80) % 256,
        (17 * cls_id + 140) % 256,
        (29 * cls_id + 200) % 256,
    ) or DEFAULT_COLOR


def draw_detections(frame, results, class_names):
    """Draw bounding boxes with class labels and confidence scores."""
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            label = resolve_class_name(class_names, cls_id)
            color = class_color(label, cls_id)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label background + text
            text = f"{label} {conf:.2f}"
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                frame,
                text,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return frame


def draw_fps(frame, fps):
    """Draw FPS counter in the top-left corner."""
    text = f"FPS: {fps:.1f}"
    cv2.putText(
        frame,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_status(frame, text):
    """Draw a status bar at the bottom of the frame."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 35), (w, h), (40, 40, 40), -1)
    cv2.putText(
        frame,
        text,
        (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )


def upload_frame(frame, upload_url, source_tag):
    """Encode frame as JPEG and upload to backend."""
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        print(f"[{source_tag}] Failed to encode frame as JPEG.")
        return False

    try:
        response = requests.post(
            upload_url,
            data=encoded.tobytes(),
            headers={"Content-Type": "image/jpeg"},
            timeout=10,
        )
    except requests.RequestException as exc:
        print(f"[{source_tag}] Upload failed: {exc}")
        return False

    if response.ok:
        print(f"[{source_tag}] Uploaded frame ({response.status_code})")
        return True

    print(f"[{source_tag}] Upload error {response.status_code}: {response.text}")
    return False


def run_webcam(model, conf, upload_url):
    """Run webcam detection and upload at 1 FPS."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)

    print(f"Webcam opened. Uploading processed frames to: {upload_url}")
    print("Mode: webcam | Rate: 1 FPS")

    frame_num = 0
    while True:
        cycle_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from webcam.")
            time.sleep(1)
            continue

        frame_num += 1

        t_start = time.perf_counter()
        results = model(frame, conf=conf, verbose=False)
        t_end = time.perf_counter()

        fps = 1.0 / max(t_end - t_start, 1e-9)

        frame = draw_detections(frame, results, model.names)
        draw_fps(frame, fps)
        draw_status(frame, f"[WEBCAM] Frame {frame_num} | upload=1fps")

        upload_frame(frame, upload_url, f"WEBCAM #{frame_num}")

        elapsed = time.perf_counter() - cycle_start
        sleep_time = max(0.0, 1.0 - elapsed)
        time.sleep(sleep_time)

    cap.release()


def run_video(model, conf, video_path, upload_url):
    """Run detection on a video file and upload at 1 FPS."""
    if not Path(video_path).is_file():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    print(f"Video: {video_path} ({total_frames} frames, {video_fps:.1f} FPS source)")
    print(f"Uploading processed frames to: {upload_url}")
    print("Mode: video | Rate: 1 FPS | Loops when video ends")

    frame_num = 0

    while True:
        cycle_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_num = 0
            print("Reached end of video. Restarting from frame 1.")
            continue

        frame_num += 1

        t_start = time.perf_counter()
        results = model(frame, conf=conf, verbose=False)
        t_end = time.perf_counter()

        fps = 1.0 / max(t_end - t_start, 1e-9)

        frame = draw_detections(frame, results, model.names)
        draw_fps(frame, fps)
        draw_status(frame, f"[VIDEO] Frame {frame_num}/{total_frames} | upload=1fps")

        upload_frame(frame, upload_url, f"VIDEO #{frame_num}")

        elapsed = time.perf_counter() - cycle_start
        sleep_time = max(0.0, 1.0 - elapsed)
        time.sleep(sleep_time)

    cap.release()


def run_images(model, conf, image_dir, upload_url):
    """Run detection on a directory and upload one image every 5 seconds."""
    img_dir = Path(image_dir)
    if not img_dir.is_dir():
        print(f"Error: Directory not found: {image_dir}")
        sys.exit(1)

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    image_paths = sorted(
        [p for p in img_dir.iterdir() if p.suffix.lower() in extensions]
    )

    if not image_paths:
        print(f"Error: No images found in {image_dir}")
        sys.exit(1)

    print(f"Found {len(image_paths)} images in {image_dir}")
    print(f"Uploading processed images to: {upload_url}")
    print("Mode: images | Rate: 1 image every 5 seconds | Loops forever")

    idx = 0

    while True:
        cycle_start = time.perf_counter()

        img_path = image_paths[idx]
        frame = cv2.imread(str(img_path))

        if frame is None:
            print(f"Warning: Could not read {img_path}, skipping.")
            idx = (idx + 1) % len(image_paths)
            time.sleep(5)
            continue

        t_start = time.perf_counter()
        results = model(frame, conf=conf, verbose=False)
        t_end = time.perf_counter()

        inference_ms = (t_end - t_start) * 1000

        frame = draw_detections(frame, results, model.names)
        draw_status(
            frame,
            f"[IMAGES] {idx + 1}/{len(image_paths)} {img_path.name} | {inference_ms:.0f}ms | upload=5s",
        )

        upload_frame(frame, upload_url, f"IMAGES #{idx + 1} {img_path.name}")

        idx = (idx + 1) % len(image_paths)

        elapsed = time.perf_counter() - cycle_start
        sleep_time = max(0.0, 5.0 - elapsed)
        time.sleep(sleep_time)


def main():
    parser = argparse.ArgumentParser(
        description="Agricultural Weed Detection - Local Testing Environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        choices=["webcam", "video", "images"],
        help="Input source: webcam, video file, or image directory",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="best.pt",
        help="Path to YOLO model weights (default: trained crop/weed model)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.3,
        help="Confidence threshold for detections (default: 0.3)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to video file or image directory (required for video/images mode)",
    )
    parser.add_argument(
        "--upload-url",
        type=str,
        default=DEFAULT_UPLOAD_URL,
        help=(
            "Endpoint to upload processed frames "
            "(default from IMAGE_UPLOAD_URL env or http://localhost:3001/upload)"
        ),
    )

    args = parser.parse_args()

    # Validate --input is provided for video/images modes
    if args.source in ("video", "images") and args.input is None:
        parser.error(f"--input is required for --source {args.source}")

    # Validate model path
    if not Path(args.model).is_file():
        print(f"Error: Model file not found: {args.model}")
        print("Available models:")
        for p in Path(".").rglob("*.pt"):
            print(f"  {p}")
        sys.exit(1)

    # Load model
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    # Print class names from the loaded model
    if hasattr(model, "names"):
        print(f"Classes: {model.names}")
    print(f"Confidence threshold: {args.conf}")
    print(f"Upload URL: {args.upload_url}")
    print()

    # Dispatch to the appropriate mode
    if args.source == "webcam":
        run_webcam(model, args.conf, args.upload_url)
    elif args.source == "video":
        run_video(model, args.conf, args.input, args.upload_url)
    elif args.source == "images":
        run_images(model, args.conf, args.input, args.upload_url)


if __name__ == "__main__":
    main()
