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

import numpy as np
import open_clip
import torch
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────

YOLO_CONFIDENCE   = 0.50    # minimum YOLO confidence for person
YOLO_MIN_BOX_AREA = 0.005   # min bbox as fraction of image area

CLIP_MODEL_NAME = "ViT-L-14"
CLIP_PRETRAINED = "openai"

# Shorter, unambiguous prompts work better with CLIP
HAZARD_PROMPTS = [
    # fire — visually unambiguous
    "visible fire with burning flames",
    "active fire emergency with flames",
    "building on fire with flames and smoke",
    "wildfire burning with orange flames",
    "car or vehicle on fire",
    # smoke — heavy and clearly fire-related
    "heavy black smoke rising from a fire",
    "thick smoke billowing from a burning building",
    "dense smoke cloud from fire",
    "smoke and fire filling the air",
    "large smoke plume from emergency fire",
]

SAFE_PROMPTS = [
    "normal outdoor scene with no fire or smoke",
    "safe indoor environment no emergency",
    "clear blue sky with no smoke",
    "orange sunset sky without fire",
    "morning fog or mist without fire",
    "steam from cooking or factory no fire",
    "dust cloud with no fire",
    "clouds in the sky no smoke",
    "street scene no emergency",
    "people in a normal safe environment",
    "forest or nature with no fire",
    "industrial smoke from a chimney not a fire",
]

# ── Tiered hazard detection thresholds ──
# HIGH:   mean_margin > HIGH_MARGIN                         → hazard (clear evidence)
# MEDIUM: mean_margin > MED_MARGIN  AND  max > MIN_MAX      → hazard (two signals required)
# else:   not hazard
HIGH_MARGIN = 0.015  # clear fire/smoke: mean margin alone sufficient
MED_MARGIN  = 0.005  # borderline: require BOTH margin AND high max
MIN_MAX     = 0.23   # single-prompt floor for medium tier


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
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                box_area = (x2 - x1) * (y2 - y1)
                if box_area / img_area >= YOLO_MIN_BOX_AREA:
                    return True
    return False


def precompute_text_features(clip_model, tokenizer, device):
    """Precompute and cache normalized text features for all prompts (run once)."""
    all_prompts = HAZARD_PROMPTS + SAFE_PROMPTS
    text = tokenizer(all_prompts).to(device)
    autocast_device = "cpu" if device.type == "mps" else device.type
    with torch.no_grad(), torch.amp.autocast(device_type=autocast_device):
        text_features = clip_model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features


def detect_hazard_clip(clip_model, preprocess, device, text_features, img_path):
    """Tiered hazard detection:
    - HIGH confidence : mean_margin > HIGH_MARGIN              → hazard
    - MEDIUM confidence: mean_margin > MED_MARGIN AND max > MIN_MAX → hazard
    - otherwise        : not hazard
    """
    image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)

    autocast_device = "cpu" if device.type == "mps" else device.type
    with torch.no_grad(), torch.amp.autocast(device_type=autocast_device):
        image_features = clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ text_features.T)[0]

    n_hazard    = len(HAZARD_PROMPTS)
    hazard_sims = similarity[:n_hazard]
    safe_sims   = similarity[n_hazard:]

    hazard_mean = hazard_sims.mean().item()
    safe_mean   = safe_sims.mean().item()
    hazard_max  = hazard_sims.max().item()
    margin      = hazard_mean - safe_mean

    high_conf   = margin > HIGH_MARGIN
    medium_conf = (margin > MED_MARGIN) and (hazard_max > MIN_MAX)

    return high_conf or medium_conf


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Auto-label images: hazard / person / both / null"
    )
    parser.add_argument("--image_dir", required=True, help="Folder containing images")
    parser.add_argument("--output", default="./results", help="Output folder")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing labels.txt instead of overwriting")
    args = parser.parse_args()

    image_dir  = Path(args.image_dir).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([
        p for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
        for p in image_dir.rglob(ext)
    ], key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)

    if not image_paths:
        raise SystemExit(f"[ERROR] No images found in {image_dir}")

    print(f"[INFO] Found {len(image_paths)} images in {image_dir}")

    device = torch.device(
        "mps"  if torch.backends.mps.is_available()  else
        "cuda" if torch.cuda.is_available()           else
        "cpu"
    )
    print(f"[INFO] Using device: {device}")

    print("[INFO] Loading YOLOv8x (COCO-pretrained)...")
    yolo_model = YOLO("yolov8x.pt")
    yolo_model.to(device)

    print(f"[INFO] Loading CLIP ({CLIP_MODEL_NAME}) on {device}...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
    )
    clip_model = clip_model.to(device).eval()
    tokenizer  = open_clip.get_tokenizer(CLIP_MODEL_NAME)

    print("[INFO] Precomputing text features...")
    text_features = precompute_text_features(clip_model, tokenizer, device)

    print(f"[INFO] Thresholds — HIGH_MARGIN={HIGH_MARGIN}  MED_MARGIN={MED_MARGIN}  MIN_MAX={MIN_MAX}")
    print("-" * 50)

    stats = {"hazard": 0, "person": 0, "both": 0, "null": 0}
    label_lines = []

    for img_path in tqdm(image_paths, desc="Labeling", unit="img"):
        has_person = detect_person(yolo_model, str(img_path))
        has_hazard = detect_hazard_clip(clip_model, preprocess, device, text_features, img_path)

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

    labels_path = output_dir / "labels.txt"
    mode = "a" if args.append else "w"
    with open(labels_path, mode) as f:
        for line in label_lines:
            f.write(line + "\n")

    total = len(image_paths)
    print("\n" + "-" * 50)
    print(f"Done! Total: {total}")
    print(f"  hazard: {stats['hazard']} ({stats['hazard']/total*100:.1f}%)")
    print(f"  person: {stats['person']} ({stats['person']/total*100:.1f}%)")
    print(f"  both:   {stats['both']}   ({stats['both']/total*100:.1f}%)")
    print(f"  null:   {stats['null']}   ({stats['null']/total*100:.1f}%)")
    print(f"Labels file → {labels_path}")


if __name__ == "__main__":
    main()
