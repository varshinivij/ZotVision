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

YOLO_CONFIDENCE = 0.25

CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"

HAZARD_PROMPTS = [
    "a photo of fire",
    "a photo of flames burning",
    "a photo of smoke",
    "a photo of a building on fire",
    "a photo of a wildfire",
]
SAFE_PROMPTS = [
    "a normal photo with no fire or smoke",
    "a photo of a safe indoor scene",
    "a photo of a landscape with clear sky",
]

CLIP_HAZARD_THRESHOLD = 0.70  # probability above this = hazard


# ─────────────────────────────────────────────────────────────────
# DETECTION FUNCTIONS
# ─────────────────────────────────────────────────────────────────

def detect_person(model, img_path):
    """Detect people using YOLOv8x."""
    results = model(img_path, verbose=False, conf=YOLO_CONFIDENCE)
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            if model.names[cls_id] == "person":
                return True
    return False


def detect_hazard_clip(clip_model, preprocess, tokenizer, device, img_path):
    """Detect fire/smoke using CLIP zero-shot classification."""
    image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)

    all_prompts = HAZARD_PROMPTS + SAFE_PROMPTS
    text = tokenizer(all_prompts).to(device)

    with torch.no_grad(), torch.amp.autocast(device_type=device.type if device.type != "mps" else "cpu"):
        image_features = clip_model.encode_image(image)
        text_features = clip_model.encode_text(text)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (image_features @ text_features.T).softmax(dim=-1)[0]

    hazard_score = similarity[: len(HAZARD_PROMPTS)].sum().item()
    return hazard_score > CLIP_HAZARD_THRESHOLD


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
    ])

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

    # Write labels
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