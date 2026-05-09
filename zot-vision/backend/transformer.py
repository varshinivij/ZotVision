import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from PIL import Image
from torchvision import transforms
from transformers import ViTModel, ViTConfig

NUM_CLASSES = 3
IMG_SIZE = 224
ID_TO_LABEL = {0: "null", 1: "hazard", 2: "person"}

_INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class CNNViTHybrid(nn.Module):
    """EfficientNet-B4 feature extractor → patch projection → CLS token
    → positional embedding → Google ViT encoder → MLP classifier"""

    def __init__(self, efficientnet_variant="efficientnet-b4", vit_hidden_size=768,
                 vit_num_layers=6, vit_num_heads=12, num_classes=3, dropout=0.3):
        super().__init__()
        self.cnn = EfficientNet.from_pretrained(efficientnet_variant)
        cnn_ch = self.cnn._conv_head.out_channels
        self.cnn._avg_pooling = self.cnn._dropout = self.cnn._fc = nn.Identity()
        for name, p in self.cnn.named_parameters():
            if not any(k in name for k in ["_blocks.28", "_blocks.29", "_blocks.30", "_blocks.31", "_conv_head"]):
                p.requires_grad = False

        self.patch_proj = nn.Conv2d(cnn_ch, vit_hidden_size, kernel_size=1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, vit_hidden_size))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pos_embed = nn.Parameter(torch.zeros(1, 50, vit_hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.vit_hidden_size = vit_hidden_size

        self.vit_encoder = ViTModel(ViTConfig(
            hidden_size=vit_hidden_size, num_hidden_layers=vit_num_layers,
            num_attention_heads=vit_num_heads, intermediate_size=vit_hidden_size * 4,
            hidden_dropout_prob=dropout, attention_probs_dropout_prob=dropout,
            image_size=IMG_SIZE, patch_size=16, num_channels=3,
        ))
        self.classifier = nn.Sequential(
            nn.LayerNorm(vit_hidden_size), nn.Linear(vit_hidden_size, 256),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(256, num_classes),
        )

    def forward(self, x):
        B = x.size(0)
        feat = self.patch_proj(self.cnn.extract_features(x))
        N = feat.shape[2] * feat.shape[3]
        feat = feat.flatten(2).transpose(1, 2)
        tokens = torch.cat([self.cls_token.expand(B, -1, -1), feat], dim=1)
        if self.pos_embed.size(1) != N + 1:
            pos = nn.functional.interpolate(
                self.pos_embed.transpose(1, 2), size=N + 1, mode="linear", align_corners=False
            ).transpose(1, 2)
        else:
            pos = self.pos_embed
        tokens = tokens + pos
        out = self.vit_encoder.encoder(tokens)
        cls = self.vit_encoder.layernorm(out.last_hidden_state)[:, 0, :]
        return self.classifier(cls)


@torch.no_grad()
def predict(model, image_path, device=None):
    if device is None:
        device = next(model.parameters()).device
    img = _INFER_TRANSFORM(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    model.eval()
    return ID_TO_LABEL[model(img).argmax(1).item()]
