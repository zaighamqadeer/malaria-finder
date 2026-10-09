"""MobileNetV3-Small backbone definition for TinyMalariaNet.

The model is a thin wrapper around torchvision's MobileNetV3-Small that:
  * targets 128x128 single-centre-cell crops,
  * emits a single logit (binary: Parasitized vs. Uninfected),
  * is QAT-friendly for the INT8 export path in ``src/quantize.py``.

Note: MobileNetV3-Small ends in a 576-wide GAP feature vector. At 128x128
input the spatial map is 4x4 (downsampled by 32 from 128/4=32 -> 4), so GAP
output is deterministic regardless of input size, which keeps ONNX/torchscript
traces stable across batch sizes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision.models import MobileNetV3, mobilenet_v3_small


@dataclass
class ModelConfig:
    """Hyperparameters / shape metadata shared across train + export."""

    image_size: int = 128
    num_classes: int = 2
    width_mult: float = 1.0
    dropout: float = 0.2
    pretrained: bool = True


class TinyMalariaNet(nn.Module):
    """MobileNetV3-Small binary classifier over single RBC crops."""

    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ModelConfig()
        try:
            base: MobileNetV3 = mobilenet_v3_small(
                weights="DEFAULT" if self.cfg.pretrained else None,
                width_mult=self.cfg.width_mult,
                dropout=self.cfg.dropout,
            )
        except TypeError:  # torchvision < 0.13 API fallback
            base = mobilenet_v3_small(pretrained=self.cfg.pretrained)
            base.width_mult = self.cfg.width_mult

        self.features = base.features
        self.avgpool = base.avgpool
        feature_dim = base.classifier[0].in_features

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=self.cfg.dropout, inplace=True),
            nn.Linear(1024, 1),  # single logit -> binary
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits of shape (B,). Apply sigmoid/BCE-with-logits."""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x).squeeze(1)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Sigmoid output in [0, 1]; >0.5 means Parasitized."""
        self.eval()
        return torch.sigmoid(self.forward(x))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(cfg: ModelConfig | None = None, pretrained: bool = True) -> TinyMalariaNet:
    cfg = cfg or ModelConfig()
    cfg.pretrained = pretrained
    return TinyMalariaNet(cfg)


if __name__ == "__main__":
    m = build_model(pretrained=False)
    out = m(torch.randn(2, 3, 128, 128))
    print("output shape:", tuple(out.shape))
    print("parameters: {:,}".format(m.num_parameters()))
