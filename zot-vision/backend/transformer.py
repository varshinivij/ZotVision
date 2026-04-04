"""
CNN-ViT Hybrid Model: EfficientNet + Google ViT
Output: 4-class classification → ['human', 'none', 'both', 'hazard']
CLS token is extracted from the ViT's last hidden state for classification.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTModel, ViTConfig
from efficientnet_pytorch import EfficientNet
from PIL import Image
import numpy as np

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
DATASET_PATH = "/dummy/path/to/dataset"          # ← replace with real path
NUM_CLASSES  = 4
LABEL_MAP    = {"human": 0, "none": 1, "both": 2, "hazard": 3}
ID_TO_LABEL  = {v: k for k, v in LABEL_MAP.items()}

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
    Expects directory structure:
        /dummy/path/to/dataset/
            human/   *.jpg
            none/    *.jpg
            both/    *.jpg
            hazard/  *.jpg
    """
    def __init__(self, root_dir: str, transform=None):
        self.samples   = []
        self.transform = transform

        for label_name, label_idx in LABEL_MAP.items():
            class_dir = os.path.join(root_dir, label_name)
            if not os.path.isdir(class_dir):
                print(f"[WARN] Missing class folder: {class_dir}")
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(class_dir, fname), label_idx))

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
        dropout: float = 0.1,
    ):
        super().__init__()

        # ── 1. EfficientNet backbone (remove classifier + pooling) ──
        self.cnn = EfficientNet.from_pretrained(efficientnet_variant)
        cnn_out_channels = self.cnn._conv_head.out_channels  # 1792 for B4

        # Remove EfficientNet's own pooling & FC so we get a feature map
        self.cnn._avg_pooling  = nn.Identity()
        self.cnn._dropout      = nn.Identity()
        self.cnn._fc           = nn.Identity()

        # ── 2. Project CNN feature map channels → ViT hidden dim ──
        self.patch_proj = nn.Conv2d(cnn_out_channels, vit_hidden_size, kernel_size=1)

        # ── 3. Learnable CLS token ──
        self.cls_token = nn.Parameter(torch.zeros(1, 1, vit_hidden_size))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ── 4. Positional embedding (built dynamically on first forward) ──
        #     We'll create it lazily so any spatial resolution works.
        self.pos_embed = None
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

    def _build_pos_embed(self, num_patches: int):
        """Lazily build / update positional embeddings."""
        total_tokens = num_patches + 1  # +1 for CLS
        pos = nn.Parameter(
            torch.zeros(1, total_tokens, self.vit_hidden_size, device=self.cls_token.device)
        )
        nn.init.trunc_normal_(pos, std=0.02)
        return pos

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)

        # ── CNN feature extraction ──
        feat = self.cnn.extract_features(x)          # (B, C_cnn, H', W')
        feat = self.patch_proj(feat)                  # (B, D, H', W')

        H, W = feat.shape[2], feat.shape[3]
        N    = H * W                                  # number of patch tokens

        # Flatten spatial dims → sequence of patch tokens
        feat = feat.flatten(2).transpose(1, 2)        # (B, N, D)

        # ── Prepend CLS token ──
        cls_tokens = self.cls_token.expand(B, -1, -1) # (B, 1, D)
        tokens     = torch.cat([cls_tokens, feat], dim=1)  # (B, N+1, D)

        # ── Positional embedding ──
        if self.pos_embed is None or self.pos_embed.size(1) != N + 1:
            self.pos_embed = self._build_pos_embed(N)
        tokens = tokens + self.pos_embed              # (B, N+1, D)

        # ── ViT transformer encoder ──
        # We pass pre-computed patch tokens directly using inputs_embeds
        vit_out   = self.vit_encoder(inputs_embeds=tokens)
        last_hidden = vit_out.last_hidden_state       # (B, N+1, D)

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
# MAIN
# ──────────────────────────────────────────────
def main():
    print(f"Using device: {DEVICE}")

    # ── Datasets ──
    train_ds = CustomDataset(os.path.join(DATASET_PATH, "train"), get_transforms(train=True))
    val_ds   = CustomDataset(os.path.join(DATASET_PATH, "val"),   get_transforms(train=False))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    # ── Model ──
    model = CNNViTHybrid(
        efficientnet_variant="efficientnet-b4",
        vit_hidden_size=768,
        vit_num_layers=6,
        vit_num_heads=12,
        num_classes=NUM_CLASSES,
        dropout=0.1,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    # ── Optimizer & scheduler ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ── Training loop ──
    best_val_acc = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        print(
            f"Epoch {epoch:03d}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_cnn_vit.pth")
            print(f"  ✓ Saved best model (val_acc={best_val_acc:.4f})")

    print(f"\nTraining complete. Best Val Acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()