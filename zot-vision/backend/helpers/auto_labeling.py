"""
Auto Image Labeler (CLIP + YOLOv8x)
=====================================
Detects:
  - "hazard"  : fire or smoke detected (CLIP zero-shot)
  - "person"  : person detected (YOLOv8x COCO-pretrained)
  - "both"    : hazard + person
  - "null"    : nothing detected — safe scene

Install:
    pip install ultralytics opencv-python-headless tqdm open-clip-torch pillow

Usage — single image:
    python auto_labeling.py --image_path path/to/image.jpg

Usage — batch (directory):
    python auto_labeling.py --image_dir datasets/images --output datasets/results

    --image_path and --image_dir are mutually exclusive; one is required.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import open_clip
import torch
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────

YOLO_CONFIDENCE = 0.50           # minimum confidence for person detection
YOLO_MIN_BOX_AREA = 0.005        # minimum bbox area as fraction of image (filters tiny spurious boxes)

CLIP_MODEL_NAME = "ViT-L-14"
CLIP_PRETRAINED = "openai"

HAZARD_PROMPTS = [
    # fire
    "a photo of fire",
    "a photo of flames burning",
    "a photo of a building engulfed in fire",
    "a photo of a wildfire with orange flames",
    "a photo of an active fire emergency with visible flames",
    "a photo of burning structures",
    "a photo of a house on fire",
    "a photo of a car on fire",
    "a photo of flames and smoke in the sky",
    # smoke
    "a photo of smoke",
    "a photo of smoke in the air",
    "a photo of white smoke rising",
    "a photo of gray smoke haze",
    "a photo of thick black smoke",
    "a photo of smoke filling the air",
    "a photo of dense smoke billowing from a fire",
    "a photo of a smoky scene",
    "a photo of smoke coming from a building",
]
SAFE_PROMPTS = [
    "a normal photo with no fire or smoke",
    "a photo of a safe indoor scene",
    "a photo of a landscape with a clear blue sky",
    "a photo of a room with no danger",
    "a photo of everyday objects with no emergency",
    "a photo of people going about their day safely",
    "a photo of a sunset with orange sky but no fire",
    "a photo of fog or mist with no fire",
    "a photo of clouds in the sky",
    "a photo of steam or vapor with no fire",
    "a photo of a city street with no emergency",
    "a photo of trees and nature with no fire",
]

# Primary: mean hazard must exceed mean safe by this margin.
# Secondary: if any single hazard prompt similarity exceeds this absolute threshold,
#            also flag as hazard (catches subtle smoke even when mean is low).
CLIP_HAZARD_MARGIN = 0.03
CLIP_HAZARD_MAX_THRESHOLD = 0.25


# ─────────────────────────────────────────────────────────────────
# DETECTION FUNCTIONS
# ─────────────────────────────────────────────────────────────────

def detect_person(model, img_path):
    """Detect people using YOLOv8x with minimum bounding box area filter."""
    results = model(img_path, verbose=False, conf=YOLO_CONFIDENCE)
    for result in results:
        img_h, img_w = result.orig_shape
        img_area = img_h * img_w
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            if model.names[cls_id] == "person":
                # Filter out tiny spurious detections
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                box_area = (x2 - x1) * (y2 - y1)
                if box_area / img_area >= YOLO_MIN_BOX_AREA:
                    return True
    return False


def detect_hazard_clip(clip_model, preprocess, tokenizer, device, img_path):
    """Detect fire/smoke using CLIP zero-shot classification.

    Two-condition detection:
    1. Mean hazard similarity exceeds mean safe similarity by CLIP_HAZARD_MARGIN
       (catches clear fire/smoke scenes).
    2. Any single hazard prompt similarity exceeds CLIP_HAZARD_MAX_THRESHOLD
       (catches subtle smoke that scores high on one specific prompt but dilutes
       the mean).
    """
    image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)

    all_prompts = HAZARD_PROMPTS + SAFE_PROMPTS
    text = tokenizer(all_prompts).to(device)

    autocast_device = "cpu" if device.type == "mps" else device.type
    with torch.no_grad(), torch.amp.autocast(device_type=autocast_device):
        image_features = clip_model.encode_image(image)
        text_features = clip_model.encode_text(text)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        similarity = (image_features @ text_features.T)[0]

    n_hazard = len(HAZARD_PROMPTS)
    hazard_sims = similarity[:n_hazard]
    safe_sims = similarity[n_hazard:]

    hazard_mean = hazard_sims.mean().item()
    safe_mean = safe_sims.mean().item()
    hazard_max = hazard_sims.max().item()

    # Flag if mean margin exceeds threshold OR any single prompt is very confident
    return (hazard_mean > safe_mean + CLIP_HAZARD_MARGIN) or (hazard_max > CLIP_HAZARD_MAX_THRESHOLD)


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def _label_one(img_path, yolo_model, clip_model, preprocess, tokenizer, device) -> str:
    """Return the label string for a single image path."""
    has_person = detect_person(yolo_model, str(img_path))
    has_hazard = detect_hazard_clip(clip_model, preprocess, tokenizer, device, img_path)
    if has_hazard and has_person:
        return "both"
    if has_hazard:
        return "hazard"
    if has_person:
        return "person"
    return "null"


def _load_models(device):
    """Load YOLOv8x and CLIP once; return (yolo, clip_model, preprocess, tokenizer)."""
    print("[INFO] Loading YOLOv8x (COCO-pretrained)...")
    yolo_model = YOLO("yolov8x.pt")

    print(f"[INFO] Loading CLIP ({CLIP_MODEL_NAME}) on {device}...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
    )
    clip_model = clip_model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
    return yolo_model, clip_model, preprocess, tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Auto-label images as: hazard / person / both / null",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Single image\n"
            "  python auto_labeling.py --image_path datasets/images/42.jpg\n\n"
            "  # Whole directory → writes datasets/results/labels.txt\n"
            "  python auto_labeling.py --image_dir datasets/images --output datasets/results\n"
        ),
    )

    # ── Input: exactly one of --image_path or --image_dir ──
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--image_path",
        metavar="FILE",
        help="Path to a single image file to label (.jpg / .png / .webp …)",
    )
    src.add_argument(
        "--image_dir",
        metavar="DIR",
        help="Path to a directory of images to label in batch",
    )

    parser.add_argument(
        "--output",
        metavar="DIR",
        default="./results",
        help="Output directory for labels.txt (batch mode only, default: ./results)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing labels.txt instead of overwriting (batch mode only)",
    )
    args = parser.parse_args()

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    yolo_model, clip_model, preprocess, tokenizer = _load_models(device)

    # ── Single-image mode ──
    if args.image_path:
        img_path = Path(args.image_path).resolve()
        if not img_path.is_file():
            raise SystemExit(f"[ERROR] Image not found: {img_path}")
        label = _label_one(img_path, yolo_model, clip_model, preprocess, tokenizer, device)
        print(f"\nImage : {img_path}")
        print(f"Label : {label}")
        return

    # ── Batch mode ──
    image_dir = Path(args.image_dir).resolve()
    if not image_dir.is_dir():
        raise SystemExit(f"[ERROR] Directory not found: {image_dir}")

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        [
            p for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
            for p in image_dir.rglob(ext)
        ],
        key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem,
    )
    if not image_paths:
        raise SystemExit(f"[ERROR] No images found in {image_dir}")

    print(f"[INFO] Found {len(image_paths)} images in {image_dir}")
    print("[INFO] Labels: hazard, person, both, null")
    print("-" * 50)

    stats = {"hazard": 0, "person": 0, "both": 0, "null": 0}
    label_lines = []

    for img_path in tqdm(image_paths, desc="Labeling", unit="img"):
        label = _label_one(img_path, yolo_model, clip_model, preprocess, tokenizer, device)
        stats[label] += 1
        label_lines.append(label)

    labels_path = output_dir / "labels.txt"
    mode = "a" if args.append else "w"
    with open(labels_path, mode) as f:
        for line in label_lines:
            f.write(line + "\n")

    total = len(image_paths)
    print("\n" + "-" * 50)
    print(f"Done! Total: {total}")
    print(f"  hazard : {stats['hazard']}")
    print(f"  person : {stats['person']}")
    print(f"  both   : {stats['both']}")
    print(f"  null   : {stats['null']}")
    print(f"Labels file → {labels_path}")


if __name__ == "__main__":
    main()