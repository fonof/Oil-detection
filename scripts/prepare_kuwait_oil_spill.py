"""
Калибровка Kuwait 2017 SAFE (VV) + crop Al Khiran AOI via geolocation grid.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from scipy import ndimage

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAFE = PROJECT_ROOT / "data" / "kuwait_2017_oil" / (
    "S1A_IW_GRDH_1SDV_20170810T024714_20170810T024738_017855_01DEF7_F48C.SAFE"
)
OUT = SAFE / "output"
AOI = (48.325, 28.414, 48.754, 28.743)  # west, south, east, north
STD_FACTOR = 1.0
MIN_OBJECT_PX = 50


def load_sigma_lut(calib_xml: Path):
    root = ET.parse(calib_xml).getroot()
    vectors = root.findall(".//calibrationVector")
    lines, pixels, sigma = [], [], []
    for vec in vectors:
        line = int(vec.findtext("line"))
        pix = np.fromstring(vec.findtext("pixel") or "", sep=" ", dtype=np.float64)
        sig = np.fromstring(vec.findtext("sigmaNought") or "", sep=" ", dtype=np.float64)
        if pix.size == 0 or sig.size == 0:
            continue
        n = min(pix.size, sig.size)
        lines.append(line)
        pixels.append(pix[:n])
        sigma.append(sig[:n])
    return np.asarray(lines, dtype=np.float64), pixels, sigma


def build_sigma_lut_window(row_off, col_off, height, width, lines, pixels, sigma):
    x_win = np.arange(col_off, col_off + width, dtype=np.float64)
    row_luts = np.empty((len(lines), width), dtype=np.float32)
    for i, (pix, sig) in enumerate(zip(pixels, sigma)):
        row_luts[i] = np.interp(x_win, pix, sig).astype(np.float32)
    y_win = np.arange(row_off, row_off + height, dtype=np.float64)
    idx = np.clip(np.searchsorted(lines, y_win, side="right") - 1, 0, len(lines) - 2)
    y0, y1 = lines[idx], lines[idx + 1]
    t = ((y_win - y0) / np.maximum(y1 - y0, 1e-6)).astype(np.float32)
    lut = np.empty((height, width), dtype=np.float32)
    for start in range(0, height, 512):
        stop = min(height, start + 512)
        i0 = idx[start:stop]
        tw = t[start:stop][:, None]
        lut[start:stop] = row_luts[i0] * (1.0 - tw) + row_luts[i0 + 1] * tw
    return lut


def calibrate_to_sigma0_db(dn, a_lut):
    a = np.maximum(a_lut, 1e-6)
    return (10.0 * np.log10((dn / a) ** 2 + 1e-12)).astype(np.float32)


def geogrid_points(ann_xml: Path):
    root = ET.parse(ann_xml).getroot()
    pts = root.findall(".//{*}geolocationGridPoint") or root.findall(
        ".//geolocationGridPoint"
    )
    lines, pixels, lats, lons = [], [], [], []
    for p in pts:

        def t(name, node=p):
            el = node.find(name)
            if el is None:
                el = node.find(f"{{*}}{name}")
            return el.text if el is not None else None

        lines.append(float(t("line")))
        pixels.append(float(t("pixel")))
        lats.append(float(t("latitude")))
        lons.append(float(t("longitude")))
    return map(np.asarray, (lines, pixels, lats, lons))


def lonlat_to_rowcol(lon, lat, lines, pixels, lats, lons):
    d = (lats - lat) ** 2 + (lons - lon) ** 2
    i = int(np.argmin(d))
    return int(lines[i]), int(pixels[i])


def aoi_window_from_geogrid(ann_xml: Path, aoi, full_h, full_w, margin=300):
    west, south, east, north = aoi
    lines, pixels, lats, lons = geogrid_points(ann_xml)
    corners = [
        lonlat_to_rowcol(lon, lat, lines, pixels, lats, lons)
        for lon, lat in (
            (west, south),
            (west, north),
            (east, south),
            (east, north),
        )
    ]
    rows = [r for r, _ in corners]
    cols = [c for _, c in corners]
    r0 = max(0, min(rows) - margin)
    r1 = min(full_h, max(rows) + margin)
    c0 = max(0, min(cols) - margin)
    c1 = min(full_w, max(cols) + margin)
    return r0, r1, c0, c1


def downsample(arr, max_side=2048):
    h, w = arr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale >= 1.0:
        return arr
    ys = np.linspace(0, h - 1, max(1, int(h * scale))).astype(np.int32)
    xs = np.linspace(0, w - 1, max(1, int(w * scale))).astype(np.int32)
    return arr[ys][:, xs]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    meas = SAFE / "measurement"
    calib = SAFE / "annotation" / "calibration"
    ann_dir = SAFE / "annotation"

    pol_files = list(meas.glob("*vv*.tiff")) or list(meas.glob("*vh*.tiff"))
    if not pol_files:
        print("No measurement tiff")
        return 1
    path = pol_files[0]
    pol = "vv" if "vv" in path.name.lower() else "vh"
    print(f"Using {path.name} ({pol})")

    ann_xml = list(ann_dir.glob(f"*-{pol}-*.xml")) or list(ann_dir.glob(f"*{pol}*.xml"))
    if not ann_xml:
        print("No annotation XML")
        return 1

    with rasterio.open(path) as src:
        full_h, full_w = src.height, src.width
        r0, r1, c0, c1 = aoi_window_from_geogrid(ann_xml[0], AOI, full_h, full_w)
        print(f"AOI window rows {r0}:{r1} cols {c0}:{c1}")
        window = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)
        data = src.read(1, window=window).astype(np.float32)
        profile = src.profile.copy()
        transform = rasterio.transform.from_origin(float(c0), float(r0), 1.0, 1.0)
        row_off, col_off = r0, c0

    print(f"crop={data.shape} offset=({row_off},{col_off})")

    calib_files = list(calib.glob(f"calibration-*{pol}*.xml"))
    if not calib_files:
        print("No calib XML")
        return 1
    lines, pixels, sigma = load_sigma_lut(calib_files[0])
    a_lut = build_sigma_lut_window(
        row_off, col_off, data.shape[0], data.shape[1], lines, pixels, sigma
    )
    db = calibrate_to_sigma0_db(data, a_lut)
    del data, a_lut

    valid = db > -50.0
    p1 = float(np.percentile(db[valid], 1))
    p99 = float(np.percentile(db[valid], 99))
    clipped = np.clip(db, p1, p99)
    clipped[~valid] = p1
    norm = (clipped - p1) / (p99 - p1 + 1e-10)
    norm[~valid] = 0.0

    profile.update(
        height=norm.shape[0],
        width=norm.shape[1],
        transform=transform,
        dtype="float32",
        count=1,
        compress="lzw",
        nodata=-9999.0,
        crs=None,
    )
    out_norm = OUT / "sentinel1_normalized_v2.tif"
    with rasterio.open(out_norm, "w", **profile) as dst:
        dst.write(norm.astype(np.float32), 1)
    with rasterio.open(OUT / "sentinel1_sigma0_db.tif", "w", **profile) as dst:
        dst.write(clipped.astype(np.float32), 1)
    print(f"Saved {out_norm}")

    sea = valid & (clipped <= np.percentile(clipped[valid], 95))
    thr = float(clipped[sea].mean() - STD_FACTOR * clipped[sea].std())
    oil = valid & (clipped < thr)
    oil[clipped > np.percentile(clipped[valid], 95)] = False
    oil = ndimage.binary_opening(oil, structure=np.ones((3, 3)))
    oil = ndimage.binary_closing(oil, structure=np.ones((5, 5)))
    labeled, _ = ndimage.label(oil)
    sizes = np.bincount(labeled.ravel())
    remove = sizes < MIN_OBJECT_PX
    remove[0] = False
    oil[remove[labeled]] = False
    area = float(oil.sum()) * 100.0 / 1e6
    print(f"Classical oil ~{area:.1f} km2 (thr={thr:.1f} dB)")

    mask_prof = profile.copy()
    mask_prof.update(dtype="uint8", nodata=255)
    with rasterio.open(OUT / "oil_mask_cleaned.tif", "w", **mask_prof) as dst:
        dst.write((oil.astype(np.uint8) * 255), 1)

    viz = downsample(norm)
    viz_oil = downsample(oil.astype(np.float32))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(viz, cmap="gray")
    axes[0].set_title("Kuwait 2017-08-10 VV (AOI)")
    axes[0].axis("off")
    rgb = np.stack([viz, viz, viz], -1)
    rgb[viz_oil > 0.5] = (1, 0, 0)
    axes[1].imshow(np.clip(rgb, 0, 1))
    axes[1].set_title(f"Classical ~{area:.1f} km²")
    axes[1].axis("off")
    fig.savefig(OUT / "oil_detection_final.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Viz -> {OUT / 'oil_detection_final.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
