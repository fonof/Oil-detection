"""
Dataset для детекции зданий.

Поддерживаемые датасеты:
- Inria Aerial Image Labeling (PNG патчи 256x256 после prepare_inria.py)
- Massachusetts / SpaceNet (PNG/JPG, стандартные пары по stem)
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

from src.utils import ensure_dir, log_error, log_info, log_warn

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
NUM_WORKERS = 2

# --- Нормализация по датасетам ---

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_inria_stats() -> tuple[list[float], list[float]]:
    """
    Статистики нормализации для аэрофотоснимков Inria.

    Более точные значения для RGB aerial imagery (не ImageNet).
    """
    mean = [0.35, 0.35, 0.35]
    std = [0.15, 0.15, 0.15]
    return mean, std


def get_massachusetts_stats() -> tuple[list[float], list[float]]:
    """Нормализация для Massachusetts Buildings (aerial RGB)."""
    mean = [0.36, 0.36, 0.36]
    std = [0.18, 0.18, 0.18]
    return mean, std


def get_oil_spill_stats() -> tuple[list[float], list[float]]:
    """Нормализация для optical / SAR-like oil spill RGB кадров."""
    mean = [0.30, 0.35, 0.38]
    std = [0.18, 0.18, 0.18]
    return mean, std


def get_oil_spill_sar_stats() -> tuple[list[float], list[float]]:
    """Sentinel-1 grayscale (R=G=B), nabilsherif oil-spill."""
    mean = [0.45, 0.45, 0.45]
    std = [0.20, 0.20, 0.20]
    return mean, std


def get_normalize_stats(dataset_name: str = "imagenet") -> tuple[list[float], list[float]]:
    """
    Возвращает mean/std для A.Normalize по имени датасета.

    Args:
        dataset_name: 'inria', 'massachusetts', 'oil_spill', 'oil_spill_sar', 'imagenet'
    """
    name = dataset_name.lower()
    if name in ("inria", "inria_patches"):
        return get_inria_stats()
    if name in ("massachusetts", "mass"):
        return get_massachusetts_stats()
    if name in ("oil_spill_sar", "sar_oil", "nabilsherif"):
        return get_oil_spill_sar_stats()
    if name in ("oil_spill", "oil", "oilspill"):
        return get_oil_spill_stats()
    return IMAGENET_MEAN, IMAGENET_STD


# --- Загрузка файлов ---


def _load_rgb_image(path: Path) -> np.ndarray:
    """RGB float32 [H, W, 3] в [0, 1]."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"OpenCV не смог прочитать: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image.astype(np.float32) / 255.0


def _load_binary_mask(path: Path) -> np.ndarray:
    """Бинарная маска float32 [H, W] — 0.0 / 1.0."""
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"OpenCV не смог прочитать маску: {path}")

    if mask.ndim == 3:
        b, g, r = cv2.split(mask)
        red = (r > 128) & (r > b) & (r > g)
        white = (r > 200) & (g > 200) & (b > 200)
        return np.logical_or(red, white).astype(np.float32)

    return (mask > 128).astype(np.float32)


def _collect_pairs(images_dir: Path, masks_dir: Path) -> tuple[list[Path], list[Path]]:
    """Сопоставление image/mask по stem (00000.png <-> 00000.png / Oil (1).jpg <-> Oil (1).png)."""
    image_paths = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"Нет изображений в {images_dir}")

    mask_by_stem = {
        p.stem.lower(): p
        for p in masks_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    }

    paired_images: list[Path] = []
    paired_masks: list[Path] = []
    missing = 0
    for img_path in image_paths:
        mask_path = mask_by_stem.get(img_path.stem.lower())
        if mask_path is None:
            missing += 1
            continue
        paired_images.append(img_path)
        paired_masks.append(mask_path)

    if not paired_images:
        raise FileNotFoundError(f"Нет пар image/mask в {images_dir} / {masks_dir}")
    if missing:
        log_warn(f"Пропущено без маски: {missing}")

    log_info(f"Найдено пар: {len(paired_images)}")
    return paired_images, paired_masks


class BuildingDataset(Dataset):
    """PyTorch Dataset: RGB image + binary building mask."""

    def __init__(
        self,
        images_dir: str | Path,
        masks_dir: str | Path,
        img_size: int = 256,
        transform: A.Compose | None = None,
        image_paths: list[Path] | None = None,
        mask_paths: list[Path] | None = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.img_size = img_size
        self.transform = transform

        if image_paths is not None and mask_paths is not None:
            self.image_paths = list(image_paths)
            self.mask_paths = list(mask_paths)
        else:
            ensure_dir(self.images_dir)
            ensure_dir(self.masks_dir)
            self.image_paths, self.mask_paths = _collect_pairs(
                self.images_dir, self.masks_dir
            )

        log_info(f"Dataset: {len(self.image_paths)} пар")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = _load_rgb_image(self.image_paths[idx])
        mask = _load_binary_mask(self.mask_paths[idx])

        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image = aug["image"]
            mask = aug["mask"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            mask = torch.from_numpy(mask).unsqueeze(0).float()

        if isinstance(mask, torch.Tensor):
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            mask = mask.float()
        else:
            mask = torch.from_numpy(np.asarray(mask, dtype=np.float32))
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)

        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()

        return image.float(), mask.float()


def get_transforms(
    is_train: bool = True,
    img_size: int = 256,
    dataset_name: str = "inria",
    strong_augment: bool = True,
) -> A.Compose:
    """
    Albumentations-пайплайн.

    Train (strong_augment=True): базовые аугментации без extreme distortions.
    """
    mean, std = get_normalize_stats(dataset_name)
    mask_targets = {"mask": "mask"}

    if is_train and strong_augment:
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Affine(
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                    scale=(0.9, 1.1),
                    rotate=(-15, 15),
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=0.5,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.3,
                ),
                A.Normalize(mean=mean, std=std, max_pixel_value=1.0),
                ToTensorV2(),
            ],
            additional_targets=mask_targets,
            is_check_shapes=False,
        )

    if is_train:
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Affine(
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                    scale=(0.9, 1.1),
                    rotate=(-15, 15),
                    p=0.5,
                ),
                A.RandomBrightnessContrast(p=0.3),
                A.Normalize(mean=mean, std=std, max_pixel_value=1.0),
                ToTensorV2(),
            ],
            additional_targets=mask_targets,
        )

    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=mean, std=std, max_pixel_value=1.0),
            ToTensorV2(),
        ],
        additional_targets=mask_targets,
    )


def create_dataloaders(
    images_dir: str | Path,
    masks_dir: str | Path,
    batch_size: int = 4,
    train_ratio: float = 0.8,
    img_size: int = 256,
    dataset_name: str = "inria",
    val_images_dir: str | Path | None = None,
    val_masks_dir: str | Path | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Train/val: либо готовый split (val_*), либо 80/20 из одной папки."""
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)

    if val_images_dir is not None and val_masks_dir is not None:
        train_images, train_masks = _collect_pairs(images_dir, masks_dir)
        val_images, val_masks = _collect_pairs(Path(val_images_dir), Path(val_masks_dir))
        train_ds = BuildingDataset(
            images_dir,
            masks_dir,
            img_size=img_size,
            transform=get_transforms(True, img_size, dataset_name),
            image_paths=train_images,
            mask_paths=train_masks,
        )
        val_ds = BuildingDataset(
            Path(val_images_dir),
            Path(val_masks_dir),
            img_size=img_size,
            transform=get_transforms(False, img_size, dataset_name),
            image_paths=val_images,
            mask_paths=val_masks,
        )
    else:
        all_images, all_masks = _collect_pairs(images_dir, masks_dir)
        n = len(all_images)
        rng = torch.Generator().manual_seed(42)
        perm = torch.randperm(n, generator=rng).tolist()
        train_size = max(1, int(n * train_ratio))
        if n - train_size == 0:
            train_size = n - 1

        train_ds = BuildingDataset(
            images_dir,
            masks_dir,
            img_size=img_size,
            transform=get_transforms(True, img_size, dataset_name),
            image_paths=[all_images[i] for i in perm[:train_size]],
            mask_paths=[all_masks[i] for i in perm[:train_size]],
        )
        val_ds = BuildingDataset(
            images_dir,
            masks_dir,
            img_size=img_size,
            transform=get_transforms(False, img_size, dataset_name),
            image_paths=[all_images[i] for i in perm[train_size:]],
            mask_paths=[all_masks[i] for i in perm[train_size:]],
        )

    log_info(f"Split: train={len(train_ds)}, val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def validate_inria_patches(
    output_dir: str | Path,
    save_path: str | Path = "output/inria_validation.png",
    n_samples: int = 10,
    max_check: int = 2000,
) -> dict[str, float]:
    """
    Проверка патчей Inria после prepare_inria.py.

    - Валидация размеров и бинарности (случайная выборка max_check)
    - Статистика покрытия зданий
    - Визуализация n_samples случайных патчей

    Returns:
        dict с mean/min/max coverage
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    save_path = Path(save_path)

    image_paths, mask_paths = _collect_pairs(images_dir, labels_dir)
    total = len(image_paths)
    check_n = min(max_check, total)
    check_indices = (
        random.sample(range(total), check_n) if check_n < total else list(range(total))
    )

    coverages: list[float] = []
    errors: list[str] = []

    for idx in check_indices:
        img_p, mask_p = image_paths[idx], mask_paths[idx]
        img = _load_rgb_image(img_p)
        mask = _load_binary_mask(mask_p)

        if img.shape[:2] != mask.shape[:2]:
            errors.append(f"Размер: {img_p.name}")
        unique = np.unique(mask)
        if not set(unique.tolist()).issubset({0.0, 1.0}):
            errors.append(f"Не бинарная маска: {mask_p.name}")

        coverages.append(float(mask.mean()))

    stats = {
        "count": total,
        "checked": check_n,
        "mean_coverage": float(np.mean(coverages)) if coverages else 0.0,
        "min_coverage": float(np.min(coverages)) if coverages else 0.0,
        "max_coverage": float(np.max(coverages)) if coverages else 0.0,
        "errors": len(errors),
    }

    print("=" * 60)
    print("INRIA PATCHES VALIDATION")
    print("=" * 60)
    print(f"  Патчей всего:    {stats['count']}")
    print(f"  Проверено:       {stats['checked']} (sample)")
    print(f"  Coverage mean:   {stats['mean_coverage']:.2%}")
    print(f"  Coverage min:    {stats['min_coverage']:.2%}")
    print(f"  Coverage max:    {stats['max_coverage']:.2%}")
    if errors:
        print(f"  ⚠️  Ошибок: {len(errors)}")
        for e in errors[:5]:
            print(f"     - {e}")
    else:
        print("  ✅ Все патчи корректны")

    # Визуализация
    n = min(n_samples, len(image_paths))
    if n == 0:
        log_warn("Нет патчей для визуализации")
        return stats

    indices = random.sample(range(len(image_paths)), n)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = np.array([axes])

    for row, idx in enumerate(indices):
        img = (_load_rgb_image(image_paths[idx]) * 255).astype(np.uint8)
        mask = (_load_binary_mask(mask_paths[idx]) * 255).astype(np.uint8)
        overlay = img.copy()
        overlay[mask > 128, 0] = 255

        axes[row, 0].imshow(img)
        axes[row, 0].set_title(f"Image {image_paths[idx].stem}")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(mask, cmap="gray")
        cov = float(mask.mean())
        axes[row, 1].set_title(f"Mask ({cov:.1%})")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title("Overlay")
        axes[row, 2].axis("off")

    plt.tight_layout()
    ensure_dir(save_path.parent, create=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log_info(f"Визуализация: {save_path}")

    return stats


if __name__ == "__main__":
    import argparse

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--patches_dir", default="data/inria_patches")
    parser.add_argument("--output", default="output/inria_validation.png")
    args = parser.parse_args()

    validate_inria_patches(
        PROJECT_ROOT / args.patches_dir,
        PROJECT_ROOT / args.output,
    )
