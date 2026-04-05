"""
Label Cleaning Pipeline (Confident Learning)
=============================================
Trains a ResNet-18 with 5-fold cross-validation to get out-of-fold
predicted probabilities, then uses Cleanlab to identify likely
mislabeled samples.

Outputs:
    flagged_labels.csv — ranked list of suspected mislabeled images
                         with current label, predicted label, and
                         confidence scores.

Usage:
    python clean_labels.py
"""

import os
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from sklearn.model_selection import StratifiedKFold
from PIL import Image
from cleanlab.filter import find_label_issues

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent
IMAGE_DIR       = BASE_DIR / "images"
LABELS_FILE     = BASE_DIR / "results" / "labels.txt"
IMAGE_ORDER     = BASE_DIR / "results" / "image_order.txt"
OUTPUT_CSV      = BASE_DIR / "flagged_labels.csv"

LABEL_MAP   = {"person": 0, "null": 1, "hazard": 2, "both": 3}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

IMG_SIZE    = 224
BATCH_SIZE  = 32
NUM_FOLDS   = 5
NUM_EPOCHS  = 8       # enough to learn signal, not enough to memorize noise
LR          = 1e-3
DEVICE      = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

# ──────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────
class FlatImageDataset(Dataset):
    """Loads images from a flat directory with a separate labels file.
    Uses image_order.txt to correctly map labels to filenames
    (auto_labeling.py sorts alphabetically, not numerically)."""

    def __init__(self, image_dir: Path, labels_file: Path, image_order_file: Path, transform=None):
        self.transform = transform
        self.samples = []

        # Read labels (one per line)
        with open(labels_file) as f:
            labels = [line.strip() for line in f if line.strip()]

        # Read image ordering — maps line index to actual filename
        with open(image_order_file) as f:
            filenames = [line.strip() for line in f if line.strip()]

        for idx, (fname, label_str) in enumerate(zip(filenames, labels)):
            img_path = image_dir / fname
            if img_path.exists() and label_str in LABEL_MAP:
                self.samples.append((str(img_path), LABEL_MAP[label_str]))

        print(f"[INFO] Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


# ──────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────
def build_model(num_classes: int):
    """Pretrained ResNet-18 with a new classification head."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Freeze early layers for faster training
    for name, param in model.named_parameters():
        if "layer3" not in name and "layer4" not in name and "fc" not in name:
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)


# ──────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────
def train_fold(model, train_loader, val_indices, dataset, num_epochs):
    """Train on one fold, return out-of-fold predicted probabilities."""
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        avg_loss = running_loss / len(train_loader.dataset)
        print(f"    Epoch {epoch+1}/{num_epochs} — loss: {avg_loss:.4f}")

    # Get out-of-fold predictions
    model.eval()
    val_dataset = Subset(dataset, val_indices)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_probs = []
    with torch.no_grad():
        for imgs, _ in val_loader:
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)

    return np.concatenate(all_probs, axis=0)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Config: {NUM_FOLDS}-fold CV, {NUM_EPOCHS} epochs/fold, lr={LR}")

    # Load full dataset (eval transforms for consistent predictions)
    dataset = FlatImageDataset(IMAGE_DIR, LABELS_FILE, IMAGE_ORDER, transform=get_transforms(train=False))
    labels = np.array([s[1] for s in dataset.samples])

    # Determine number of classes actually present
    unique_classes = np.unique(labels)
    num_classes = len(unique_classes)
    print(f"[INFO] Classes present: {[ID_TO_LABEL[c] for c in unique_classes]} ({num_classes} classes)")

    # Also create a training-augmented version of the dataset
    train_dataset = FlatImageDataset(IMAGE_DIR, LABELS_FILE, IMAGE_ORDER, transform=get_transforms(train=True))

    # Out-of-fold predicted probabilities (N x num_classes)
    # Cleanlab needs probabilities for ALL label map classes, not just present ones
    total_label_classes = max(LABEL_MAP.values()) + 1
    oof_probs = np.zeros((len(dataset), total_label_classes))

    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(dataset)), labels)):
        print(f"\n[FOLD {fold+1}/{NUM_FOLDS}]")
        print(f"  Train: {len(train_idx)} | Val: {len(val_idx)}")

        model = build_model(total_label_classes)
        train_subset = Subset(train_dataset, train_idx)
        train_loader = DataLoader(
            train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
        )

        fold_probs = train_fold(model, train_loader, val_idx, dataset, NUM_EPOCHS)
        oof_probs[val_idx] = fold_probs

        # Quick val accuracy
        val_preds = fold_probs.argmax(axis=1)
        val_labels = labels[val_idx]
        acc = (val_preds == val_labels).mean()
        print(f"  Fold {fold+1} val accuracy: {acc:.4f}")

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Cleanlab: find label issues ──
    print("\n" + "=" * 50)
    print("[CLEANLAB] Finding label issues...")

    issue_indices = find_label_issues(
        labels=labels,
        pred_probs=oof_probs,
        return_indices_ranked_by="self_confidence",
    )

    print(f"[CLEANLAB] Found {len(issue_indices)} suspected mislabeled samples")

    # ── Build output CSV ──
    rows = []
    for rank, idx in enumerate(issue_indices, 1):
        img_path = dataset.samples[idx][0]
        img_fname = os.path.basename(img_path)
        current_label = ID_TO_LABEL[labels[idx]]
        predicted_class = oof_probs[idx].argmax()
        predicted_label = ID_TO_LABEL[predicted_class]
        confidence_current = oof_probs[idx][labels[idx]]
        confidence_predicted = oof_probs[idx][predicted_class]

        rows.append({
            "rank": rank,
            "image": img_fname,
            "current_label": current_label,
            "predicted_label": predicted_label,
            "confidence_current": round(float(confidence_current), 4),
            "confidence_predicted": round(float(confidence_predicted), 4),
            "image_path": img_path,
        })

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[OUTPUT] Flagged {len(rows)} images → {OUTPUT_CSV}")

    # ── Summary stats ──
    if rows:
        print("\n── Top 20 most suspicious samples ──")
        print(f"{'Rank':<6} {'Image':<12} {'Current':<10} {'Predicted':<10} {'Conf(cur)':<12} {'Conf(pred)':<12}")
        print("-" * 62)
        for r in rows[:20]:
            print(
                f"{r['rank']:<6} {r['image']:<12} {r['current_label']:<10} "
                f"{r['predicted_label']:<10} {r['confidence_current']:<12} "
                f"{r['confidence_predicted']:<12}"
            )

        # Breakdown
        mismatches = sum(1 for r in rows if r["current_label"] != r["predicted_label"])
        print(f"\n── Summary ──")
        print(f"Total flagged:        {len(rows)}")
        print(f"Label mismatches:     {mismatches} (model disagrees with label)")
        print(f"Low-confidence same:  {len(rows) - mismatches} (model agrees but uncertain)")
        print(f"Review queue:         ~{len(rows)} images ({len(rows)*100//len(dataset)}% of dataset)")


if __name__ == "__main__":
    main()
