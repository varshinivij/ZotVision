"""
CNN-ViT Hybrid Model: EfficientNet + Google ViT
Output: 4-class classification → ['none', 'hazard', 'person', 'both']
CLS token is extracted from the ViT's last hidden state for classification.
"""

import os
import random
import itertools
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTModel, ViTConfig
from efficientnet_pytorch import EfficientNet
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
KAGGLE_DATASET = "varshvijay05/firefighter-hazard-dataset"
DATASET_DIR    = os.path.join(os.path.dirname(__file__), "..", "datasets")
IMAGES_DIR     = os.path.join(DATASET_DIR, "images")
LABELS_FILE    = os.path.join(DATASET_DIR, "results", "labels.txt")


def ensure_dataset():
    """Download dataset from Kaggle if not already present locally."""
    if os.path.isdir(IMAGES_DIR) and os.path.isfile(LABELS_FILE):
        return
    print(f"[INFO] Dataset not found locally. Downloading from Kaggle: {KAGGLE_DATASET}")
    import kagglehub
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    # Copy downloaded files into expected locations
    import shutil
    src_images = os.path.join(path, "images")
    src_results = os.path.join(path, "results")
    if os.path.isdir(src_images):
        os.makedirs(IMAGES_DIR, exist_ok=True)
        for f in os.listdir(src_images):
            shutil.copy2(os.path.join(src_images, f), IMAGES_DIR)
    if os.path.isdir(src_results):
        os.makedirs(os.path.dirname(LABELS_FILE), exist_ok=True)
        for f in os.listdir(src_results):
            shutil.copy2(os.path.join(src_results, f), os.path.dirname(LABELS_FILE))
    print(f"[INFO] Dataset downloaded to {DATASET_DIR}")
NUM_CLASSES   = 4
LABEL_MAP     = {"null": 0, "hazard": 1, "person": 2, "both": 3}
ID_TO_LABEL   = {0: "null", 1: "hazard", 2: "person", 3: "both"}

BATCH_SIZE   = 16
NUM_EPOCHS   = 20
LR           = 3e-4
IMG_SIZE     = 224
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────
class CustomDataset(Dataset):
    """
    Reads labels.txt (filename,label per line) and resolves images from IMAGES_DIR.
    No need to move images into class subdirectories.
    """
    def __init__(self, samples: list, transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def load_samples(labels_file: str = LABELS_FILE, images_dir: str = IMAGES_DIR):
    """Parse labels.txt (one label per line) → list of (image_path, label_idx).
    Image at line i maps to {i+1}.jpg."""
    samples = []
    with open(labels_file) as f:
        for i, line in enumerate(f):
            label_name = line.strip()
            if not label_name:
                continue
            if label_name not in LABEL_MAP:
                print(f"[WARN] Unknown label '{label_name}' at line {i+1}, skipping")
                continue
            img_path = os.path.join(images_dir, f"{i+1}.jpg")
            if not os.path.exists(img_path):
                print(f"[WARN] Missing image {img_path}, skipping")
                continue
            samples.append((img_path, LABEL_MAP[label_name]))
    return samples


def get_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2),
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
class CNNViTHybrid(nn.Module):
    """
    Architecture:
      1. EfficientNet-B4 backbone  →  extracts spatial feature map (C×H×W)
      2. Patch projection          →  flattens spatial tokens  (N_patches × D)
      3. Prepend learnable CLS token
      4. Add positional embeddings
      5. Google ViT transformer encoder  →  contextualises all tokens
      6. Extract final CLS token         →  (B, D)
      7. MLP classifier head             →  4 logits
    """

    def __init__(
        self,
        efficientnet_variant: str = "efficientnet-b4",
        vit_hidden_size: int = 768,
        vit_num_layers: int = 6,
        vit_num_heads: int = 12,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.3,
    ):
        super().__init__()

        # ── 1. EfficientNet backbone (remove classifier + pooling) ──
        self.cnn = EfficientNet.from_pretrained(efficientnet_variant)
        cnn_out_channels = self.cnn._conv_head.out_channels  # 1792 for B4

        # Remove EfficientNet's own pooling & FC so we get a feature map
        self.cnn._avg_pooling  = nn.Identity()
        self.cnn._dropout      = nn.Identity()
        self.cnn._fc           = nn.Identity()

        # Freeze early B4 blocks (0-27) — only fine-tune last 4 blocks + conv_head
        for name, param in self.cnn.named_parameters():
            if not any(k in name for k in ['_blocks.28', '_blocks.29', '_blocks.30', '_blocks.31', '_conv_head']):
                param.requires_grad = False

        # ── 2. Project CNN feature map channels → ViT hidden dim ──
        self.patch_proj = nn.Conv2d(cnn_out_channels, vit_hidden_size, kernel_size=1)

        # ── 3. Learnable CLS token ──
        self.cls_token = nn.Parameter(torch.zeros(1, 1, vit_hidden_size))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ── 4. Positional embedding ──
        # Pre-computed for EfficientNet-B4 @ 224x224 → 7x7 = 49 patches (same as B0).
        # Registered as nn.Parameter so it's saved in state_dict and moves with .to(device).
        expected_patches = 49
        self.pos_embed = nn.Parameter(torch.zeros(1, expected_patches + 1, vit_hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.vit_hidden_size = vit_hidden_size

        # ── 5. Google ViT transformer encoder ──
        vit_config = ViTConfig(
            hidden_size=vit_hidden_size,
            num_hidden_layers=vit_num_layers,
            num_attention_heads=vit_num_heads,
            intermediate_size=vit_hidden_size * 4,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            # We feed our own patch tokens, so image_size / patch_size don't matter here
            image_size=IMG_SIZE,
            patch_size=16,
            num_channels=3,
        )
        self.vit_encoder = ViTModel(vit_config)

        # ── 6. Classification head ──
        self.classifier = nn.Sequential(
            nn.LayerNorm(vit_hidden_size),
            nn.Linear(vit_hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)

        # ── CNN feature extraction ──
        feat = self.cnn.extract_features(x)          # (B, C_cnn, H', W')
        feat = self.patch_proj(feat)                  # (B, D, H', W')

        N    = feat.shape[2] * feat.shape[3]          # number of patch tokens

        # Flatten spatial dims → sequence of patch tokens
        feat = feat.flatten(2).transpose(1, 2)        # (B, N, D)

        # ── Prepend CLS token ──
        cls_tokens = self.cls_token.expand(B, -1, -1) # (B, 1, D)
        tokens     = torch.cat([cls_tokens, feat], dim=1)  # (B, N+1, D)

        # ── Positional embedding ──
        if self.pos_embed.size(1) != N + 1:
            pos = torch.nn.functional.interpolate(
                self.pos_embed.transpose(1, 2), size=N + 1, mode='linear', align_corners=False
            ).transpose(1, 2)
        else:
            pos = self.pos_embed
        tokens = tokens + pos                         # (B, N+1, D)

        # ── ViT transformer encoder ──
        # Call encoder blocks directly (bypasses ViTEmbeddings which requires pixel_values)
        vit_out     = self.vit_encoder.encoder(tokens)
        last_hidden = self.vit_encoder.layernorm(vit_out.last_hidden_state)  # (B, N+1, D)

        # ── Extract CLS token (position 0) ──
        cls_out   = last_hidden[:, 0, :]              # (B, D)

        # ── Classify ──
        logits    = self.classifier(cls_out)          # (B, 4)
        return logits


# ──────────────────────────────────────────────
# TRAINING UTILITIES
# ──────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * imgs.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_per_class(model, loader, device):
    """Collect all predictions and labels for detailed metrics."""
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


def plot_confusion_heatmap(preds, labels, class_names, save_path):
    """Save a confusion matrix heatmap to disk."""
    cm = confusion_matrix(labels, preds, labels=range(len(class_names)))
    # Normalize per row (per true class) for per-class accuracy
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title("Confusion Matrix (counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    # Normalized (per-class accuracy on diagonal)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Oranges",
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title("Confusion Matrix (normalized)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Heatmap saved → {save_path}")


def print_per_class_accuracy(preds, labels, class_names):
    """Print per-class accuracy and full classification report."""
    report = classification_report(labels, preds, target_names=class_names, digits=4)
    print("\n" + report)


# ──────────────────────────────────────────────
# INFERENCE HELPER
# ──────────────────────────────────────────────
@torch.no_grad()
def predict(model, image_path: str, device=DEVICE) -> str:
    """Run inference on a single image; returns label string."""
    transform = get_transforms(train=False)
    img    = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    model.eval()
    logits = model(tensor)
    pred   = logits.argmax(dim=1).item()
    return ID_TO_LABEL[pred]


# ──────────────────────────────────────────────
# HYPERPARAMETER CONFIGS
# ──────────────────────────────────────────────
HYPERPARAM_GRID = [
    {"lr": 3e-4, "dropout": 0.1, "weight_decay": 1e-3, "batch_size": 16, "num_epochs": 20},
    {"lr": 1e-4, "dropout": 0.2, "weight_decay": 1e-4, "batch_size": 16, "num_epochs": 20},
    {"lr": 5e-4, "dropout": 0.3, "weight_decay": 1e-3, "batch_size": 32, "num_epochs": 20},
    {"lr": 3e-4, "dropout": 0.2, "weight_decay": 1e-2, "batch_size": 8,  "num_epochs": 20},
    {"lr": 1e-3, "dropout": 0.1, "weight_decay": 1e-4, "batch_size": 16, "num_epochs": 15},
]


def train_config(config, train_samples, val_samples, run_id, results_dir):
    """Train a single hyperparameter configuration and return results."""
    print(f"\n{'='*60}")
    print(f"Run {run_id}: {config}")
    print(f"{'='*60}")

    train_ds = CustomDataset(train_samples, get_transforms(train=True))
    val_ds   = CustomDataset(val_samples,   get_transforms(train=False))

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=config["batch_size"], shuffle=False, num_workers=4, pin_memory=True)

    model = CNNViTHybrid(
        efficientnet_variant="efficientnet-b4",
        vit_hidden_size=768,
        vit_num_layers=6,
        vit_num_heads=12,
        num_classes=NUM_CLASSES,
        dropout=config["dropout"],
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["num_epochs"])
    class_weights = torch.tensor([1.0, 1.2, 1.0, 2.5]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, config["num_epochs"] + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:03d}/{config['num_epochs']} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(results_dir, f"model_run{run_id}.pth"))

    # Per-class evaluation on best model
    model.load_state_dict(torch.load(os.path.join(results_dir, f"model_run{run_id}.pth"), weights_only=True))
    class_names = [ID_TO_LABEL[i] for i in range(NUM_CLASSES)]
    preds, labels = evaluate_per_class(model, val_loader, DEVICE)
    print_per_class_accuracy(preds, labels, class_names)
    plot_confusion_heatmap(preds, labels, class_names,
                           os.path.join(results_dir, f"heatmap_run{run_id}.png"))

    return {"run_id": run_id, "config": config, "best_val_acc": best_val_acc, "history": history}


def plot_run_comparison(all_results, results_dir):
    """Plot val accuracy curves for all runs and a summary bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Loss/acc curves per run
    for r in all_results:
        label = f"run{r['run_id']} (acc={r['best_val_acc']:.3f})"
        axes[0].plot(r["history"]["val_acc"], label=label)
    axes[0].set_title("Validation Accuracy per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(fontsize=8)

    # Bar chart of best val acc
    run_labels = [f"run{r['run_id']}" for r in all_results]
    accs = [r["best_val_acc"] for r in all_results]
    bars = axes[1].bar(run_labels, accs, color=sns.color_palette("viridis", len(all_results)))
    axes[1].set_title("Best Validation Accuracy per Config")
    axes[1].set_ylabel("Accuracy")
    for bar, acc in zip(bars, accs):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{acc:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(results_dir, "run_comparison.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nComparison plot saved → {save_path}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    print(f"Using device: {DEVICE}")
    ensure_dataset()

    results_dir = os.path.join(DATASET_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    # ── Datasets (80/20 split from labels.txt, same split for all runs) ──
    all_samples = load_samples()
    random.shuffle(all_samples)
    split = int(0.8 * len(all_samples))
    train_samples = all_samples[:split]
    val_samples   = all_samples[split:]

    print(f"Train samples: {len(train_samples)} | Val samples: {len(val_samples)}")

    # ── Run all hyperparameter configs ──
    all_results = []
    for i, config in enumerate(HYPERPARAM_GRID):
        result = train_config(config, train_samples, val_samples, run_id=i, results_dir=results_dir)
        all_results.append(result)

    # ── Summary ──
    plot_run_comparison(all_results, results_dir)

    best = max(all_results, key=lambda r: r["best_val_acc"])
    print(f"\n{'='*60}")
    print(f"BEST CONFIG: run{best['run_id']} — val_acc={best['best_val_acc']:.4f}")
    print(f"  {best['config']}")
    print(f"{'='*60}")

    # Copy best model as the main weights file
    best_src = os.path.join(results_dir, f"model_run{best['run_id']}.pth")
    best_dst = os.path.join(results_dir, "model_weights.pth")
    torch.save(torch.load(best_src, weights_only=True), best_dst)
    print(f"Best model saved → {best_dst}")

    # Save all results as JSON for reference
    summary = []
    for r in all_results:
        summary.append({"run_id": r["run_id"], "config": r["config"], "best_val_acc": r["best_val_acc"]})
    with open(os.path.join(results_dir, "hyperparam_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved → {os.path.join(results_dir, 'hyperparam_results.json')}")


if __name__ == "__main__":
    main()