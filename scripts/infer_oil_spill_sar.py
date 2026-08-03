"""
Инференс U-Net (SAR oil spill) на Sentinel-1 GRD / нормализованном GeoTIFF.

Скользящее окно 256x256 -> маска -> overlay для портфолио.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import get_normalize_stats
from src.model import create_model
from src.utils import ensure_dir, log_info, log_ok


def load_sar_gray(path: Path, max_side: int = 4096) -> np.ndarray:
    """Загружает растр как float32 [0,1] grayscale, опционально даунсэмпл."""
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        nodata = src.nodata

    if nodata is not None:
        data = np.where(data == nodata, np.nan, data)

    valid = np.isfinite(data)
    if not valid.any():
        raise RuntimeError(f"Пустой растр: {path}")

    # если уже 0..1 — ок; если dB — растянем по процентилям
    vals = data[valid]
    vmin, vmax = np.percentile(vals, [2, 98])
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    norm[~valid] = 0.0

    h, w = norm.shape
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        norm = cv2.resize(norm, (new_w, new_h), interpolation=cv2.INTER_AREA)
        log_info(f"Даунсэмпл {h}x{w} -> {new_h}x{new_w}")

    return norm.astype(np.float32)


def predict_tiles(
    model: torch.nn.Module,
    gray: np.ndarray,
    device: torch.device,
    tile: int = 256,
    stride: int = 128,
    dataset_name: str = "oil_spill_sar",
    threshold: float = 0.5,
) -> np.ndarray:
    """Скользящее окно, усреднение вероятностей."""
    mean, std = get_normalize_stats(dataset_name)
    h, w = gray.shape
    prob = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)

    ys = list(range(0, max(1, h - tile + 1), stride))
    xs = list(range(0, max(1, w - tile + 1), stride))
    if ys[-1] != h - tile and h >= tile:
        ys.append(h - tile)
    if xs[-1] != w - tile and w >= tile:
        xs.append(w - tile)
    if h < tile:
        ys = [0]
    if w < tile:
        xs = [0]

    model.eval()
    with torch.inference_mode():
        for y0 in ys:
            for x0 in xs:
                y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
                patch = gray[y0:y1, x0:x1]
                ph, pw = patch.shape
                canvas = np.zeros((tile, tile), dtype=np.float32)
                canvas[:ph, :pw] = patch

                rgb = np.stack([canvas, canvas, canvas], axis=0)  # 3,H,W
                for c in range(3):
                    rgb[c] = (rgb[c] - mean[c]) / std[c]

                tensor = torch.from_numpy(rgb).unsqueeze(0).float().to(device)
                logits = model(tensor)
                p = torch.sigmoid(logits)[0, 0].cpu().numpy()
                prob[y0:y1, x0:x1] += p[:ph, :pw]
                weight[y0:y1, x0:x1] += 1.0

    weight = np.maximum(weight, 1e-6)
    prob /= weight
    return (prob >= threshold).astype(np.uint8), prob


def postprocess_mask(
    gray: np.ndarray,
    mask: np.ndarray,
    land_percentile: float = 65.0,
    min_area: int = 80,
) -> np.ndarray:
    """Убираем яркую сушу и мелкий шум (look-alike cleanup)."""
    out = mask.astype(np.uint8).copy()
    if land_percentile > 0:
        thr = float(np.percentile(gray, land_percentile))
        out[gray >= thr] = 0
    if min_area > 0:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
        cleaned = np.zeros_like(out)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                cleaned[labels == i] = 1
        out = cleaned
    return out


def make_portfolio_figure(
    gray: np.ndarray,
    mask: np.ndarray,
    prob: np.ndarray,
    out_path: Path,
    title: str,
    area_km2: float,
) -> None:
    rgb = np.stack([gray, gray, gray], axis=-1)
    overlay = rgb.copy()
    overlay[mask > 0] = [1.0, 0.15, 0.1]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("Sentinel-1 SAR")
    axes[0].axis("off")

    axes[1].imshow(prob, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title("U-Net probability")
    axes[1].axis("off")

    axes[2].imshow(np.clip(0.65 * rgb + 0.35 * overlay, 0, 1))
    axes[2].set_title(f"Candidates (~{area_km2:.2f} km²)")
    axes[2].axis("off")

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        default=str(
            PROJECT_ROOT
            / "S1D_IW_GRDH_1SDV_20260707T052138_20260707T052203_003563_006544_3681.SAFE"
            / "output"
            / "sentinel1_normalized_v2.tif"
        ),
    )
    p.add_argument("--model", default="models/oil_spill_sar_unet.pth")
    p.add_argument("--output_dir", default="output/oil_spill_sar_portfolio")
    p.add_argument("--max_side", type=int, default=3072)
    p.add_argument("--tile", type=int, default=256)
    p.add_argument("--stride", type=int, default=128)
    p.add_argument("--threshold", type=float, default=0.55)
    p.add_argument("--pixel_m", type=float, default=10.0)
    p.add_argument("--dataset_name", default="oil_spill_sar")
    p.add_argument("--land_percentile", type=float, default=65.0)
    p.add_argument("--min_area", type=int, default=80)
    args = p.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        # fallback: sigma0 db
        alt = input_path.parent / "sentinel1_sigma0_db.tif"
        if alt.exists():
            input_path = alt
            log_info(f"Использую {input_path}")
        else:
            print(f"Нет файла: {args.input}")
            return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = PROJECT_ROOT / args.model
    model = create_model(device=device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    log_info(f"Модель: {model_path}")

    gray = load_sar_gray(input_path, max_side=args.max_side)
    log_info(f"SAR shape={gray.shape}")

    mask_raw, prob = predict_tiles(
        model,
        gray,
        device,
        tile=args.tile,
        stride=args.stride,
        dataset_name=args.dataset_name,
        threshold=args.threshold,
    )
    mask = postprocess_mask(
        gray,
        mask_raw,
        land_percentile=args.land_percentile,
        min_area=args.min_area,
    )

    h0, w0 = gray.shape
    full_side = 16806  # typical SAFE height
    px_m = args.pixel_m * (full_side / max(h0, w0)) if max(h0, w0) < full_side else args.pixel_m
    area_km2 = float(mask.sum()) * (px_m**2) / 1e6

    out_dir = ensure_dir(PROJECT_ROOT / args.output_dir, create=True)
    cv2.imwrite(str(out_dir / "sar_preview.png"), (gray * 255).astype(np.uint8))
    cv2.imwrite(str(out_dir / "oil_mask_unet_raw.png"), (mask_raw * 255).astype(np.uint8))
    cv2.imwrite(str(out_dir / "oil_mask_unet.png"), (mask * 255).astype(np.uint8))
    np.save(out_dir / "oil_prob.npy", prob)

    make_portfolio_figure(
        gray,
        mask,
        prob,
        out_dir / "portfolio_sar_unet.png",
        title="Oil spill candidates — U-Net on Sentinel-1 (nabilsherif-trained)",
        area_km2=area_km2,
    )

    # side-by-side with classical if exists
    classical = input_path.parent / "oil_detection_final.png"
    if classical.exists():
        classical_img = cv2.imread(str(classical))
        if classical_img is not None:
            cv2.imwrite(str(out_dir / "classical_threshold.png"), classical_img)

    stats = out_dir / "stats.txt"
    stats.write_text(
        f"input={input_path}\n"
        f"model={model_path}\n"
        f"shape={gray.shape}\n"
        f"threshold={args.threshold}\n"
        f"land_percentile={args.land_percentile}\n"
        f"min_area={args.min_area}\n"
        f"oil_pixels_raw={int(mask_raw.sum())}\n"
        f"oil_pixels={int(mask.sum())}\n"
        f"oil_area_km2_approx={area_km2:.4f}\n"
        f"pixel_m_approx={px_m:.2f}\n"
        f"note=dark channels/deltas often look like oil in SAR; treat as candidates\n",
        encoding="utf-8",
    )

    log_ok(f"Портфолио: {out_dir / 'portfolio_sar_unet.png'}")
    log_ok(f"Площадь кандидатов (оценка): {area_km2:.2f} км²")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
