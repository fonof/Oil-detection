"""
Обучение U-Net на Inria / building detection patches.

Features:
- Gradient accumulation, AMP
- TensorBoard logging
- Val snapshot каждые 5 эпох
- Early stopping по best Val IoU
- Предупреждения overfitting / низкий IoU
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import create_dataloaders, get_normalize_stats
from src.model import create_model, get_loss, set_batchnorm_eval
from src.utils import ensure_dir, log_error, log_info, log_ok, log_warn


class SegmentationMetricTracker:
    """Global IoU/F1 over all pixels (стабильнее batch-level SMP метрик)."""

    def __init__(self) -> None:
        self.loss_sum = 0.0
        self.count = 0
        self.intersection = 0.0
        self.pred_sum = 0.0
        self.target_sum = 0.0

    def update(
        self,
        loss: torch.Tensor,
        logits: torch.Tensor,
        masks: torch.Tensor,
        threshold: float = 0.5,
    ) -> None:
        bs = logits.size(0)
        self.loss_sum += loss.item() * bs
        self.count += bs
        with torch.no_grad():
            preds = (torch.sigmoid(logits.float()) > threshold).float()
            self.intersection += (preds * masks).sum().item()
            self.pred_sum += preds.sum().item()
            self.target_sum += masks.sum().item()

    def averages(self) -> tuple[float, float, float]:
        if self.count == 0:
            return 0.0, 0.0, 0.0
        loss = self.loss_sum / self.count
        union = self.pred_sum + self.target_sum - self.intersection
        iou = self.intersection / (union + 1e-7)
        tp = self.intersection
        fp = self.pred_sum - self.intersection
        fn = self.target_sum - self.intersection
        f1 = (2 * tp) / (2 * tp + fp + fn + 1e-7)
        return loss, iou, f1


def _denormalize_image(tensor: torch.Tensor, dataset_name: str) -> np.ndarray:
    """Tensor (3,H,W) -> uint8 RGB для визуализации."""
    mean, std = get_normalize_stats(dataset_name)
    img = tensor.cpu().float().numpy().transpose(1, 2, 0)
    for c in range(3):
        img[:, :, c] = img[:, :, c] * std[c] + mean[c]
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


@torch.no_grad()
def save_val_predictions(
    model: torch.nn.Module,
    val_loader,
    device: torch.device,
    epoch: int,
    output_root: Path,
    dataset_name: str = "inria",
    n_samples: int = 5,
) -> None:
    """Сохраняет image | GT | prediction для n_samples примеров."""
    model.eval()
    out_dir = output_root / f"val_epoch_{epoch:03d}"
    ensure_dir(out_dir, create=True)

    saved = 0
    for images, masks in val_loader:
        images = images.to(device)
        masks = masks.to(device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(images)
            preds = (torch.sigmoid(logits) > 0.5).float()

        for i in range(images.size(0)):
            if saved >= n_samples:
                return

            img_rgb = _denormalize_image(images[i], dataset_name)
            gt = (masks[i, 0].cpu().numpy() > 0.5).astype(np.uint8) * 255
            pred = (preds[i, 0].cpu().numpy() > 0.5).astype(np.uint8) * 255

            fig, axes = plt.subplots(1, 3, figsize=(9, 3))
            axes[0].imshow(img_rgb)
            axes[0].set_title("Image")
            axes[0].axis("off")
            axes[1].imshow(gt, cmap="gray")
            axes[1].set_title("Ground Truth")
            axes[1].axis("off")
            axes[2].imshow(pred, cmap="gray")
            axes[2].set_title("Prediction")
            axes[2].axis("off")
            plt.tight_layout()
            fig.savefig(out_dir / f"sample_{saved:02d}.png", dpi=120, bbox_inches="tight")
            plt.close(fig)
            saved += 1


def validate_epoch(
    model: torch.nn.Module,
    val_loader,
    device: torch.device,
    criterion: torch.nn.Module,
    use_amp: bool,
    debug: bool = False,
) -> tuple[float, list[float], float]:
    """
    Валидация одной эпохи.

    Обязательно: model.eval() + inference_mode (без dropout / grad).
    После возврата модель снова в train mode.
    """
    model.eval()
    if debug:
        log_info(f"  [debug] validate: model.training={model.training}")

    tracker = SegmentationMetricTracker()
    conf_sum = 0.0
    conf_count = 0

    with torch.inference_mode():
        for images, masks in tqdm(val_loader, desc="Val", leave=False):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(images)
                loss = criterion(out, masks)
            tracker.update(loss, out, masks)
            bs = out.size(0)
            conf_sum += torch.sigmoid(out.float()).mean().item() * bs
            conf_count += bs

    va_loss, val_iou, val_f1 = tracker.averages()
    val_confidence = conf_sum / max(conf_count, 1)
    model.train()
    set_batchnorm_eval(model)
    return va_loss, [val_iou, val_f1], val_confidence


def train(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int,
    lr: float,
    patience: int,
    save_path: Path,
    accumulation_steps: int = 1,
    dataset_name: str = "inria",
    debug: bool = False,
    log_dir: Path | None = None,
    snapshot_dir: Path | None = None,
    bce_weight: float = 0.8,
    dice_weight: float = 0.2,
) -> float:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # T_max = число эпох (не батчей). step() — только раз в конце эпохи.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    criterion = get_loss(bce_weight=bce_weight, dice_weight=dice_weight)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    accum = max(1, accumulation_steps)

    writer = SummaryWriter(str(log_dir or PROJECT_ROOT / "runs" / "building_detection"))
    snapshot_dir = snapshot_dir or PROJECT_ROOT / "output"

    best_iou = 0.0
    no_improve = 0
    prev_train_iou = 0.0
    prev_val_iou = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        set_batchnorm_eval(model)
        tr = SegmentationMetricTracker()
        train_conf_sum = 0.0
        train_conf_count = 0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (images, masks) in enumerate(
            tqdm(train_loader, desc=f"Train E{epoch}", leave=False)
        ):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(images)
                loss = criterion(out, masks) / accum

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accum == 0:
                scaler.step(optimizer)  # GradScaler, не LR scheduler
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            tr.update(loss * accum, out.detach(), masks)
            with torch.no_grad():
                bs = out.size(0)
                train_conf_sum += torch.sigmoid(out.float()).mean().item() * bs
                train_conf_count += bs

            if debug and batch_idx == 0 and epoch == 1:
                with torch.no_grad():
                    prob = torch.sigmoid(out.float())
                    log_info(
                        f"  [debug] batch0 mean_sigmoid={prob.mean():.4f}, "
                        f"gt_coverage={masks.mean():.4f}"
                    )

        if len(train_loader) % accum != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        va_loss, va_m, val_confidence = validate_epoch(
            model,
            val_loader,
            device,
            criterion,
            use_amp,
            debug=debug,
        )

        tr_loss, train_iou, train_f1 = tr.averages()
        val_iou, val_f1 = va_m[0], va_m[1]
        train_confidence = train_conf_sum / max(train_conf_count, 1)
        lr_before = optimizer.param_groups[0]["lr"]
        # LR scheduler — строго раз в эпоху, после train + val
        scheduler.step()
        lr_after = optimizer.param_groups[0]["lr"]

        writer.add_scalar("Loss/train", tr_loss, epoch)
        writer.add_scalar("Loss/val", va_loss, epoch)
        writer.add_scalar("IoU/train", train_iou, epoch)
        writer.add_scalar("IoU/val", val_iou, epoch)
        writer.add_scalar("F1/train", train_f1, epoch)
        writer.add_scalar("F1/val", val_f1, epoch)
        writer.add_scalar("LR", lr_before, epoch)
        writer.add_scalar("Confidence/train", train_confidence, epoch)
        writer.add_scalar("Confidence/val", val_confidence, epoch)

        if val_iou > best_iou:
            best_iou = val_iou
            no_improve = 0
            torch.save(model.state_dict(), save_path)
            log_ok(f"Saved best IoU={val_iou:.4f} -> {save_path}")
        else:
            no_improve += 1

        log_info(
            f"Epoch {epoch}/{epochs} | LR={lr_before:.6f}->{lr_after:.6f} | "
            f"train loss={tr_loss:.4f} IoU={train_iou:.4f} conf={train_confidence:.3f} | "
            f"val loss={va_loss:.4f} IoU={val_iou:.4f} F1={val_f1:.4f} conf={val_confidence:.3f} | "
            f"best={best_iou:.4f} patience={no_improve}/{patience}"
        )

        if epoch <= 10 and lr_before < 1e-5:
            log_warn(f"LR={lr_before:.6f} < 1e-5 на эпохе {epoch} — scheduler слишком агрессивный")

        if val_iou < 0.1:
            log_warn(f"Val IoU={val_iou:.4f} < 0.1 — проверьте данные/маски")

        if epoch > 1 and train_iou > prev_train_iou and val_iou < prev_val_iou:
            log_warn(
                f"Overfitting: train IoU {prev_train_iou:.4f}->{train_iou:.4f}, "
                f"val IoU {prev_val_iou:.4f}->{val_iou:.4f}"
            )

        prev_train_iou, prev_val_iou = train_iou, val_iou

        if epoch % 5 == 0:
            save_val_predictions(
                model, val_loader, device, epoch, snapshot_dir, dataset_name
            )
            model.train()
            log_info(f"Val snapshots -> {snapshot_dir / f'val_epoch_{epoch:03d}'}")

        if no_improve >= patience:
            log_warn(f"Early stopping (patience={patience}, best IoU={best_iou:.4f})")
            break

    writer.close()
    return best_iou


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--images_dir", default="data/inria_subset_10k/images")
    p.add_argument("--masks_dir", default="data/inria_subset_10k/labels")
    p.add_argument("--val_images_dir", default=None, help="Опционально: отдельный val images")
    p.add_argument("--val_masks_dir", default=None, help="Опционально: отдельный val masks")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--accumulation_steps", type=int, default=1)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--save_path", default="models/inria_10k_fixed.pth")
    p.add_argument("--dataset_name", default="inria")
    p.add_argument("--log_dir", default="runs/building_detection")
    p.add_argument("--bce_weight", type=float, default=0.8)
    p.add_argument("--dice_weight", type=float, default=0.2)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    images_dir = PROJECT_ROOT / args.images_dir
    masks_dir = PROJECT_ROOT / args.masks_dir
    save_path = PROJECT_ROOT / args.save_path
    ensure_dir(save_path.parent, create=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_info(f"Device: {device}")
    log_info(
        f"Config: batch={args.batch_size}, accum={args.accumulation_steps}, "
        f"lr={args.lr}, epochs={args.epochs}"
    )

    val_images = PROJECT_ROOT / args.val_images_dir if args.val_images_dir else None
    val_masks = PROJECT_ROOT / args.val_masks_dir if args.val_masks_dir else None

    train_loader, val_loader = create_dataloaders(
        images_dir=images_dir,
        masks_dir=masks_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        dataset_name=args.dataset_name,
        val_images_dir=val_images,
        val_masks_dir=val_masks,
    )

    model = create_model(device=device)
    best = train(
        model,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        save_path=save_path,
        accumulation_steps=args.accumulation_steps,
        dataset_name=args.dataset_name,
        debug=args.debug,
        log_dir=PROJECT_ROOT / args.log_dir,
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
    )
    log_ok(f"Done. Best Val IoU: {best:.4f}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        log_error(str(exc))
        sys.exit(1)
