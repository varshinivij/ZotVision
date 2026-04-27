import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from transformer import CNNViTHybrid  # noqa: F401

ID_TO_LABEL = {0: "null", 1: "hazard", 2: "person", 3: "both"}
IMG_SIZE = 224

_INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model(pkl_path, device):
    raw = torch.load(pkl_path, map_location=device, weights_only=False)
    if isinstance(raw, dict):
        model = CNNViTHybrid()
        model.load_state_dict(raw)
    else:
        model = raw
    return model.to(device).eval()


def enhance_image(img_bgr, level=1):
    """CLAHE on the L channel + gamma correction. Level 2 adds a brightness bump."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clip_limit = 2.0 if level == 1 else 4.0 #higher: more contrast boost
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8)) #contrast on brightness alg
    out = cv2.merge([clahe.apply(l), a, b])
    out = cv2.cvtColor(out, cv2.COLOR_LAB2BGR)
    gamma = 1.3 if level == 1 else 1.6
    lut = np.array([(i / 255.0) ** (1.0 / gamma) * 255 for i in range(256)], dtype=np.uint8)
    out = cv2.LUT(out, lut)
    if level == 2:
        out = cv2.convertScaleAbs(out, alpha=1.0, beta=30)
    return out


def _run_inference(model, img_bgr, device):
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))  # PIL image converted from OpenCV BGR to RGB
    tensor = _INFER_TRANSFORM(pil).unsqueeze(0).to(device)  # resize, normalize, convert to tensor for model input
    with torch.no_grad():
        logits = model(tensor)
    probs = F.softmax(logits, dim=1).squeeze(0)  # probabilities per class
    pred_idx = probs.argmax().item()
    confidence = probs[pred_idx].item()
    return (
        ID_TO_LABEL[pred_idx],
        round(confidence, 4),
        round(1.0 - confidence, 4),
        {ID_TO_LABEL[i]: round(probs[i].item(), 4) for i in range(4)},
    )


def predict_low_visibility(model, image_path, uncertainty_threshold=0.4, device=None, timeout=10.0):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # Holds the highest-confidence result seen so far across completed passes.
    best_so_far = []

    def _record(label, confidence, uncertainty, all_probs, level):
        entry = {
            "result": label,
            "confidence": confidence,
            "uncertainty": uncertainty,
            "enhancement_level": level,
            "all_probs": all_probs,
            "error": False,
        }
        if not best_so_far or confidence > best_so_far[0]["confidence"]:
            best_so_far[:] = [entry]
        return entry

    def _pipeline():
        # Pass 0 — raw image, no enhancement
        entry = _record(*_run_inference(model, img, device), level=0)
        if entry["uncertainty"] < uncertainty_threshold:
            return entry

        # Pass 1 — mild CLAHE + gamma 1.3; only runs if uncertainty >= threshold
        entry = _record(*_run_inference(model, enhance_image(img, level=1), device), level=1)
        if entry["uncertainty"] < uncertainty_threshold:
            return entry

        # Pass 2 — aggressive CLAHE + gamma 1.6 + brightness boost; only runs if still >= threshold
        return _record(*_run_inference(model, enhance_image(img, level=2), device), level=2)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_pipeline)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            if best_so_far:
                best_so_far[0]["error"] = True
                return best_so_far[0]
            return {
                "result": None,
                "confidence": 0.0,
                "uncertainty": 1.0,
                "enhancement_level": None,
                "all_probs": {},
                "error": True,
            }


def main(model_path, image_path, uncertainty_threshold=0.4, timeout=10.0, device=None):
    device = torch.device(device) if isinstance(device, str) else (
        device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = load_model(model_path, device)
    output = predict_low_visibility(model, image_path, uncertainty_threshold, device, timeout)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
