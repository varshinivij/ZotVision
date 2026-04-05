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

Usage:
    python auto_labeling.py --image_dir datasets/images --output datasets/results
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

def main():
    parser = argparse.ArgumentParser(
        description="Auto-label images: hazard / person / both / null"
    )
    parser.add_argument("--image_dir", required=True, help="Folder containing images")
    parser.add_argument("--output", default="./results", help="Output folder")
    args = parser.parse_args()

    image_dir = Path(args.image_dir).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([
        p for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
        for p in image_dir.rglob(ext)
    ], key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)

    if not image_paths:
        raise SystemExit(f"[ERROR] No images found in {image_dir}")

    print(f"[INFO] Found {len(image_paths)} images in {image_dir}")

    # Load YOLOv8x
    print("[INFO] Loading YOLOv8x (COCO-pretrained)...")
    yolo_model = YOLO("yolov8x.pt")

    # Load CLIP
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Loading CLIP ({CLIP_MODEL_NAME}) on {device}...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED)
    clip_model = clip_model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)

    print("[INFO] Labels: hazard, person, both, null")
    print("-" * 50)

    stats = {"hazard": 0, "person": 0, "both": 0, "null": 0}
    label_lines = []

    for img_path in tqdm(image_paths, desc="Labeling", unit="img"):
        has_person = detect_person(yolo_model, str(img_path))
        has_hazard = detect_hazard_clip(clip_model, preprocess, tokenizer, device, img_path)

        if has_hazard and has_person:
            label = "both"
        elif has_hazard:
            label = "hazard"
        elif has_person:
            label = "person"
        else:
            label = "null"

        stats[label] += 1
        label_lines.append(label)

    # Write labels (filename + label for traceability)
    labels_path = output_dir / "labels.txt"
    with open(labels_path, "w") as f:
        for line in label_lines:
            f.write(line + "\n")

    # Summary
    total = len(image_paths)
    print("\n" + "-" * 50)
    print(f"Done! Total: {total}")
    print(f"  hazard: {stats['hazard']}")
    print(f"  person: {stats['person']}")
    print(f"  both:   {stats['both']}")
    print(f"  null:   {stats['null']}")
    print(f"Labels file → {labels_path}")


if __name__ == "__main__":
    main()