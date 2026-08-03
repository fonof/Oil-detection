"""
U-Net для сегментации нефтяных пятен на SAR (segmentation_models_pytorch).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import log_info


def set_batchnorm_eval(module: nn.Module) -> None:
    """BatchNorm в eval во время train — фиксирует ImageNet running stats."""
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def create_model(
    encoder_name: str = "resnet34",
    device: str | torch.device = "cuda",
    decoder_dropout: float = 0.2,
) -> nn.Module:
    import segmentation_models_pytorch as smp

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
        decoder_dropout=decoder_dropout,
    ).to(device)
    log_info(f"U-Net ({encoder_name}, BatchNorm frozen in train) -> {device}")
    return model


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        pt = torch.exp(-bce)
        alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        return (alpha_t * (1.0 - pt) ** self.gamma * bce).mean()


class BCEDiceLoss(nn.Module):
    """BCE (уверенность) + Dice (форма)."""

    def __init__(self, bce_weight: float = 0.8, dice_weight: float = 0.2) -> None:
        super().__init__()
        import segmentation_models_pytorch as smp

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(pred, target) + self.dice_weight * self.dice(
            pred, target
        )


def get_loss(bce_weight: float = 0.8, dice_weight: float = 0.2) -> nn.Module:
    return BCEDiceLoss(bce_weight=bce_weight, dice_weight=dice_weight)


def get_metrics(threshold: float = 0.5) -> list:
    from segmentation_models_pytorch.utils.metrics import Fscore, IoU

    # activation='sigmoid' обязателен: без него порог 0.5 применяется к logits!
    return [
        IoU(threshold=threshold, activation="sigmoid"),
        Fscore(threshold=threshold, activation="sigmoid"),
    ]
