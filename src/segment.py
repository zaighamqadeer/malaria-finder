"""Stage 1: OpenCV Red Blood Cell (RBC) segmenter.

Isolates candidate RBC crops (128x128) from raw full-field microscope captures
or smartphone eyepiece photos. Raw phone FOV frames are messy: uneven
illumination, dust/scratches, plasma haze, out-of-focus blobs and clumped
cells. The pipeline therefore does:

    grayscale -> illumination correction (background division)
      -> center-weighted band-pass (removes background plasma + dust specks)
      -> adaptive threshold -> morphological cleanup
      -> distance transform + peak detection -> watershed (splits touching RBCs)
      -> shape/size/circularity filtering (discards WBCs, debris, overlap clumps)
      -> square padded crops exported at the classifier input size

Depends only on OpenCV + NumPy so it stays dependency-light on device.
"""

from __future__ import annotations

# Support BOTH documented entry points: `python src/segment.py --test-image ...`
# (design doc section 3) and `python -m src.segment`.
if __package__ in (None, ""):  # pragma: no cover - exercised via the shell
    import os
    import sys as _sys

    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "src"  # noqa: A001

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class SegmentConfig:
    # Illumination / background
    bg_kernel: int = 51           # large-gaussian background estimate (odd)
    resize_max_side: int = 1600   # cap for processing, crops are re-read at full res

    # Cell band-pass (remove tiny specks + huge blobs)
    min_area_frac: float = 0.00015  # fraction of image area
    max_area_frac: float = 0.02
    min_radius_frac: float = 0.002
    max_radius_frac: float = 0.06

    # Thresholding
    adaptive_block: int = 31      # must be odd
    adaptive_c: float = 4.0
    otsu_blend: float = 0.5

    # Watershed splitting of touching cells
    split_clumps: bool = True
    peak_rel_thresh: float = 0.35  # distance-transform peak threshold
    min_split_distance: int = 6

    # Shape filters
    min_circularity: float = 0.35
    max_aspect_ratio: float = 2.6
    max_solidity_gap: float = 0.55  # holes inside blob vs cell body

    # Output
    crop_size: int = 128
    pad_frac: float = 0.18
    min_crop_contrast: float = 12.0
    max_candidates: int = 4000


@dataclass
class Candidate:
    path: str
    x: int
    y: int
    w: int
    h: int
    cx: float
    cy: float
    area: float
    circularity: float
    mean_gray: float
    contrast: float
    confidence: float = 1.0


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _read_image(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    img = cv2.imread(str(path), flags)
    if img is None:
        raise FileNotFoundError(f"Could not decode image: {path}")
    return img


def _resize_to_max_side(img: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return img, scale


def correct_illumination(gray: np.ndarray, kernel: int) -> np.ndarray:
    """Divide out the low-frequency staining/lighting background.

    Phone eyepiece captures have strong vignetting; dividing by a heavily
    blurred version of the image flattens it without erasing cell boundaries.
    """
    k = kernel if kernel % 2 == 1 else kernel + 1
    bg = cv2.GaussianBlur(gray, (k, k), 0)
    bg = np.maximum(bg.astype(np.float32), 1.0)
    flat = gray.astype(np.float32) / bg * float(np.mean(gray))
    return np.clip(flat, 0, 255).astype(np.uint8)


def _bandpass_mask(gray: np.ndarray, cfg: SegmentConfig, h: int, w: int) -> np.ndarray:
    """Adaptive threshold with Otsu blend; pixels inside cells are foreground."""
    block = cfg.adaptive_block if cfg.adaptive_block % 2 == 1 else cfg.adaptive_block + 1
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block,
        cfg.adaptive_c,
    )
    otsu_val, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    blended = cv2.addWeighted(adaptive, 1.0 - cfg.otsu_blend, otsu, cfg.otsu_blend, 0.0)
    _, blended = cv2.threshold(blended, 127, 255, cv2.THRESH_BINARY)

    # Remove 1px specks (dust, stain speckle) and close hairline gaps.
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(blended, cv2.MORPH_OPEN, open_k, iterations=1)
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k, iterations=2)
    return mask


def _shape_stats(cnt: np.ndarray) -> tuple[float, float]:
    area = cv2.contourArea(cnt)
    perim = cv2.arcLength(cnt, True)
    circularity = float(4.0 * math.pi * area / (perim * perim)) if perim > 0 else 0.0
    (_, _), (rw, rh), _ = cv2.minAreaRect(cnt)
    aspect = float(max(rw, rh) / max(1.0, min(rw, rh)))
    return circularity, aspect


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #
def segment_cells(
    image_path: str | Path,
    cfg: SegmentConfig | None = None,
    output_dir: str | Path | None = None,
    save_debug: bool = False,
) -> list[Candidate]:
    """Extract candidate RBC crops from a single raw FOV image.

    Returns a list of :class:`Candidate`. Crops are written to ``output_dir``
    (as ``cell_<i>.png``) together with ``candidates.json`` when provided.
    """
    cfg = cfg or SegmentConfig()
    orig = _read_image(image_path)
    if orig.ndim == 2:
        orig = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)

    # Work on a capped-size copy for contour math; crop from full-res later.
    work, scale = _resize_to_max_side(orig, cfg.resize_max_side)
    h, w = work.shape[:2]
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)

    gray = correct_illumination(gray, cfg.bg_kernel)
    mask = _bandpass_mask(gray, cfg, h, w)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total_area = float(h * w)
    min_area = max(20.0, cfg.min_area_frac * total_area)
    max_area = cfg.max_area_frac * total_area
    min_radius = cfg.min_radius_frac * max(h, w)
    max_radius = cfg.max_radius_frac * max(h, w)

    blobs: list[np.ndarray] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        circ, aspect = _shape_stats(cnt)
        if circ < cfg.min_circularity * 0.6 or aspect > cfg.max_aspect_ratio * 1.5:
            continue  # obvious debris / elongated artifact
        blobs.append(cnt)

    if cfg.split_clumps:
        blobs = _split_blobs_with_watershed(gray, mask, blobs, cfg, min_area)

    candidates: list[Candidate] = []
    for cnt in blobs:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        circ, aspect = _shape_stats(cnt)
        if circ < cfg.min_circularity or aspect > cfg.max_aspect_ratio:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        cx, cy = x + bw / 2.0, y + bh / 2.0
        # map back to original coordinates
        ox, oy = int(round(x / scale)), int(round(y / scale))
        ow, oh = int(round(bw / scale)), int(round(bh / scale))

        # quality signal from the underlying grayscale
        pad = max(1, int(round(cfg.pad_frac * max(ow, oh))))
        x0, y0 = max(0, ox - pad), max(0, oy - pad)
        x1, y1 = min(orig.shape[1], ox + ow + pad), min(orig.shape[0], oy + oh + pad)
        patch = orig[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        gray_full = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        contrast = float(gray_full.std())

        conf = min(1.0, (circ / max(0.40, cfg.min_circularity)) * min(1.0, contrast / 30.0))
        candidates.append(
            Candidate(
                path="",  # filled when saving
                x=ox, y=oy, w=ow, h=oh,
                cx=cx / scale, cy=cy / scale,
                area=float(area / (scale * scale)),
                circularity=float(circ),
                mean_gray=float(gray_full.mean()),
                contrast=contrast,
                confidence=float(max(0.0, conf)),
            )
        )

    candidates.sort(key=lambda c: (-c.area, c.confidence * -1))
    candidates = candidates[: cfg.max_candidates]

    if output_dir is not None:
        _save_candidates(orig, candidates, Path(output_dir), cfg)

    if save_debug:
        _save_debug(work, mask, blobs, Path(output_dir or "."))

    return candidates


def _split_blobs_with_watershed(
    gray: np.ndarray,
    mask: np.ndarray,
    contours: list[np.ndarray],
    cfg: SegmentConfig,
    min_area: float,
) -> list[np.ndarray]:
    """Watershed on the distance transform to separate touching RBCs."""
    # Build a clean binary mask of accepted regions only.
    canvas = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(canvas, contours, -1, 255, thickness=cv2.FILLED)

    # Background = 0, unknown = 1, foreground = 2 for distance transform.
    dist = cv2.distanceTransform(canvas, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return contours

    peak_thresh = cfg.peak_rel_thresh * float(dist.max())
    _, sure_fg = cv2.threshold(dist, peak_thresh, 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Merge nearby peaks so a single never splits into ghost cells.
    if cfg.min_split_distance > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.min_split_distance,) * 2)
        sure_fg = cv2.morphologyEx(sure_fg, cv2.MORPH_CLOSE, k)

    n_markers, markers = cv2.connectedComponents(sure_fg)
    if n_markers <= 1:
        return contours  # nothing to split

    markers = markers + 1
    unknown = cv2.subtract(canvas, sure_fg)
    markers[unknown == 255] = 0
    markers = cv2.watershed(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), markers)

    out: list[np.ndarray] = []
    for label in range(2, n_markers + 1):
        region = np.uint8(markers == label) * 255
        sub, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sub:
            a = cv2.contourArea(c)
            if a >= min_area:
                out.append(c)

    if len(out) <= len(contours):
        return contours
    return out


def _save_candidates(
    orig: np.ndarray, candidates: list[Candidate], out_dir: Path, cfg: SegmentConfig
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    size = cfg.crop_size
    for i, cand in enumerate(candidates):
        pad = max(2, int(round(cfg.pad_frac * max(cand.w, cand.h))))
        x0, y0 = max(0, cand.x - pad), max(0, cand.y - pad)
        x1, y1 = min(orig.shape[1], cand.x + cand.w + pad), min(orig.shape[0], cand.y + cand.h + pad)
        patch = orig[y0:y1, x0:x1]
        if patch.size == 0:
            continue

        # Centre-crop to square then letterbox-free resize to the target size.
        ph, pw = patch.shape[:2]
        side = max(ph, pw)
        canvas = np.full((side, side, 3), int(np.median(patch)), dtype=np.uint8)
        ox, oy = (side - pw) // 2, (side - ph) // 2
        canvas[oy:oy + ph, ox:ox + pw] = patch
        crop = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)

        fname = f"cell_{i:04d}.png"
        cv2.imwrite(str(out_dir / fname), crop)
        cand.path = str(out_dir / fname)

    payload = {
        "num_candidates": len(candidates),
        "crop_size": size,
        "candidates": [asdict(c) for c in candidates],
    }
    (out_dir / "candidates.json").write_text(json.dumps(payload, indent=2))


def _save_debug(
    work: np.ndarray, mask: np.ndarray, contours: list[np.ndarray], out_dir: Path
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "debug_mask.png"), mask)
    overlay = work.copy()
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)
    cv2.imwrite(str(out_dir / "debug_overlay.png"), overlay)


# --------------------------------------------------------------------------- #
# Batch / CLI
# --------------------------------------------------------------------------- #
def segment_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    pattern: str = "*",
    cfg: SegmentConfig | None = None,
) -> dict[str, Any]:
    """Segment every image under ``input_dir`` into ``output_dir/<stem>/``."""
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    images = sorted(p for p in input_dir.rglob(pattern) if p.suffix.lower() in
                    {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
    if not images:
        raise FileNotFoundError(f"No images found under {input_dir}")

    cfg = cfg or SegmentConfig()
    summary: dict[str, Any] = {"per_image": {}, "total": 0}
    for img in images:
        dest = output_dir / img.stem
        cands = segment_cells(img, cfg=cfg, output_dir=dest, save_debug=True)
        summary["per_image"][str(img)] = len(cands)
        summary["total"] += len(cands)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OpenCV RBC segmenter for TinyMalariaNet")
    p.add_argument("--test-image", type=str, default=None, help="Single raw FOV image to segment")
    p.add_argument("--input-dir", type=str, default=None, help="Batch mode: directory of raw frames")
    p.add_argument("--output-dir", type=str, default="data/segmented", help="Where crops + json go")
    p.add_argument("--crop-size", type=int, default=128)
    p.add_argument("--no-split", action="store_true", help="Disable watershed clump splitting")
    p.add_argument("--debug", action="store_true", help="Write mask/overlay images")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = SegmentConfig(crop_size=args.crop_size, split_clumps=not args.no_split)

    if args.test_image:
        if not Path(args.test_image).exists():
            print(f"[ERROR] image not found: {args.test_image}")
            return 2
        cands = segment_cells(
            args.test_image, cfg=cfg, output_dir=args.output_dir, save_debug=args.debug
        )
        print(f"[segment] {args.test_image}")
        print(f"[segment] candidates: {len(cands)}")
        for c in cands[:10]:
            print(
                f"  cell at ({c.x},{c.y}) {c.w}x{c.h} area={c.area:.0f} "
                f"circ={c.circularity:.2f} conf={c.confidence:.2f}"
            )
        print(f"[segment] crops written to: {args.output_dir}")
        return 0

    if args.input_dir:
        summary = segment_directory(args.input_dir, args.output_dir, cfg=cfg)
        print(f"[segment] {len(summary['per_image'])} images, {summary['total']} candidates total")
        for k, v in list(summary["per_image"].items())[:20]:
            print(f"  {Path(k).name}: {v}")
        return 0

    print("Nothing to do: pass --test-image or --input-dir")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
