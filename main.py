"""
Agricultural Weed Detection - Local Testing Environment

Supports three input modes for testing the crop/weed detection model:
  - webcam:  Live camera feed (simulates drone camera)
  - video:   Process a video file frame by frame
  - images:  Iterate through a directory of images

Usage:
  python main.py --source webcam
  python main.py --source video --input drone_footage.mp4
  python main.py --source images --input ./plant_images/
  python main.py --source webcam --model yolov8n.pt --conf 0.5

Controls:
  q     - Quit
  n     - Next image (images mode)
  p     - Previous image (images mode)
  SPACE - Pause/resume (video mode)
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

# Class index -> (color_bgr, label)
# Trained on plants.yaml: 0=crop, 1=weed
CLASS_COLORS = {
    0: ((0, 200, 0), "crop"),       # green
    1: ((0, 0, 255), "weed"),       # red
}

# Fallback for unknown class indices
DEFAULT_COLOR = (255, 165, 0)  # orange (BGR)


def draw_detections(frame, results):
    """Draw bounding boxes with class labels and confidence scores."""
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            color, label = CLASS_COLORS.get(cls_id, (DEFAULT_COLOR, f"class_{cls_id}"))

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label background + text
            text = f"{label} {conf:.2f}"
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, text, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def draw_fps(frame, fps):
    """Draw FPS counter in the top-left corner."""
    text = f"FPS: {fps:.1f}"
    cv2.putText(frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)


def draw_status(frame, text):
    """Draw a status bar at the bottom of the frame."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 35), (w, h), (40, 40, 40), -1)
    cv2.putText(frame, text, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)


def run_webcam(model, conf):
    """Run real-time detection on webcam feed."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)

    print("Webcam opened. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from webcam.")
            break

        t_start = time.perf_counter()
        results = model(frame, conf=conf, verbose=False)
        t_end = time.perf_counter()

        fps = 1.0 / max(t_end - t_start, 1e-9)

        frame = draw_detections(frame, results)
        draw_fps(frame, fps)
        draw_status(frame, "[WEBCAM] Press 'q' to quit")

        cv2.imshow("Weed Detection - Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_video(model, conf, video_path):
    """Run detection on a video file frame by frame."""
    if not Path(video_path).is_file():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_delay = int(1000 / video_fps)

    print(f"Video: {video_path} ({total_frames} frames, {video_fps:.1f} FPS)")
    print("Press 'q' to quit, SPACE to pause/resume.")

    paused = False
    frame_num = 0

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video.")
                break

            frame_num += 1

            t_start = time.perf_counter()
            results = model(frame, conf=conf, verbose=False)
            t_end = time.perf_counter()

            fps = 1.0 / max(t_end - t_start, 1e-9)

            frame = draw_detections(frame, results)
            draw_fps(frame, fps)
            draw_status(frame, f"[VIDEO] Frame {frame_num}/{total_frames} | SPACE=pause  q=quit")

            cv2.imshow("Weed Detection - Video", frame)

        key = cv2.waitKey(frame_delay if not paused else 50) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused
            if paused:
                print(f"Paused at frame {frame_num}/{total_frames}")
            else:
                print("Resumed.")

    cap.release()
    cv2.destroyAllWindows()


def run_images(model, conf, image_dir):
    """Run detection on a directory of images with navigation."""
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
    print("Press 'n' for next, 'p' for previous, 'q' to quit.")

    idx = 0
    cached_frame = None
    last_idx = -1

    while True:
        if idx != last_idx:
            img_path = image_paths[idx]
            frame = cv2.imread(str(img_path))

            if frame is None:
                print(f"Warning: Could not read {img_path}, skipping.")
                idx = min(idx + 1, len(image_paths) - 1)
                continue

            t_start = time.perf_counter()
            results = model(frame, conf=conf, verbose=False)
            t_end = time.perf_counter()

            inference_ms = (t_end - t_start) * 1000

            frame = draw_detections(frame, results)
            draw_status(
                frame,
                f"[{idx + 1}/{len(image_paths)}] {img_path.name} | "
                f"{inference_ms:.0f}ms | n=next  p=prev  q=quit"
            )

            cached_frame = frame
            last_idx = idx

        cv2.imshow("Weed Detection - Images", cached_frame)

        key = cv2.waitKey(50) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("n"):
            if idx < len(image_paths) - 1:
                idx += 1
            else:
                print("Already at last image.")
        elif key == ord("p"):
            if idx > 0:
                idx -= 1
            else:
                print("Already at first image.")

    cv2.destroyAllWindows()


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
        default="runs/detect/train5/weights/best.pt",
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
    print()

    # Dispatch to the appropriate mode
    if args.source == "webcam":
        run_webcam(model, args.conf)
    elif args.source == "video":
        run_video(model, args.conf, args.input)
    elif args.source == "images":
        run_images(model, args.conf, args.input)


if __name__ == "__main__":
    main()
