"""
Подготовка nabilsherif/oil-spill (Sentinel-1 SAR + masks) для U-Net.

Cyan / near-cyan в labels -> бинарная маска нефти.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "data" / "oil_spill" / "kaggle" / "nabilsherif" / "oil-spill"
DST = PROJECT_ROOT / "data" / "oil_spill_sar_unet"

# В labels нефть = cyan (RGB 0,255,255) -> BGR (255,255,0)
# Также встречаются близкие оттенки
SIZE = 256


def oil_from_label(lab_bgr: np.ndarray) -> np.ndarray:
    """BGR label -> binary oil mask. Cyan (BGR 255,255,0) = oil (M4D scheme)."""
    if lab_bgr.ndim == 2:
        return (lab_bgr > 0).astype(np.uint8) * 255

    b, g, r = cv2.split(lab_bgr)
    # oil = cyan; look-alike=red, land=green, ship=brown — игнорируем
    cyan = (b > 200) & (g > 200) & (r < 60)
    return cyan.astype(np.uint8) * 255


def process_split(split: str) -> int:
    img_dir = SRC / split / "images"
    lab_dir = SRC / split / "labels"
    out_img = DST / split / "images"
    out_mask = DST / split / "masks"
    out_img.mkdir(parents=True, exist_ok=True)
    out_mask.mkdir(parents=True, exist_ok=True)

    count = 0
    oil_px = 0
    total_px = 0
    empty = 0

    for img_path in sorted(img_dir.glob("*.jpg")):
        lab_path = lab_dir / f"{img_path.stem}.png"
        if not lab_path.exists():
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        lab = cv2.imread(str(lab_path), cv2.IMREAD_COLOR)
        if img is None or lab is None:
            continue

        binary = oil_from_label(lab)
        oil_here = int((binary > 0).sum())
        if oil_here == 0:
            empty += 1

        oil_px += oil_here
        total_px += binary.size

        img_r = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        mask_r = cv2.resize(binary, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)

        name = f"{img_path.stem}.png"
        cv2.imwrite(str(out_img / name), img_r)
        cv2.imwrite(str(out_mask / name), mask_r)
        count += 1

    cov = oil_px / max(total_px, 1)
    print(f"{split}: {count} pairs, empty={empty}, oil coverage ~{cov:.2%}")
    return count


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not SRC.exists():
        print(f"Не найден: {SRC}")
        print("Скачайте: kaggle datasets download -d nabilsherif/oil-spill")
        return 1

    if DST.exists():
        shutil.rmtree(DST)

    total = 0
    for split in ("train", "test"):
        if (SRC / split).exists():
            total += process_split(split)

    # val = часть train (или копируем test как val для отчёта — лучше split train)
    # create_dataloaders умеет брать val отдельно — используем test как val
    # для портфолио: symlink/copy test -> val
    val_img = DST / "val" / "images"
    val_mask = DST / "val" / "masks"
    if (DST / "test").exists() and not val_img.exists():
        shutil.copytree(DST / "test" / "images", val_img)
        shutil.copytree(DST / "test" / "masks", val_mask)
        print("val: скопирован test")

    print(f"Готово: {DST} ({total} пар)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
