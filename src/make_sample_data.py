"""Synthesise mock RBC crops for the offline codespace dry-run.

Real datasets (NIH / BBBC041 / Makerere) are large downloads that must not
block a smoke test, so this module generates small synthetic crops with
plausible microscopy statistics: pale pink biconcave discs on a light plasma
background, with Giemsa-stained chromatin rings/blobs inside Parasitized cells.

These images are for EXERCISING THE PIPELINE ONLY. They carry no diagnostic
signal and must never be used to claim model accuracy.

Usage
-----
    python -m src.make_sample_data --out data/sample --total 200
"""

from __future__ import annotations

# Support BOTH documented entry points: `python src/make_sample_data.py ...` and
# `python -m src.make_sample_data`.
if __package__ in (None, ""):  # pragma: no cover - exercised via the shell
    import os
    import sys as _sys

    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "src"  # noqa: A001

import argparse
from pathlib import Path
import json
from typing import Any

import cv2
import numpy as np

from .dataset import patient_id_from_stem


def _draw_rbc(
    rng: np.random.Generator,
    size: int,
    parasitized: bool,
    ring_stage: bool = True,
) -> tuple[np.ndarray, float, float]:
    """Draw one synthetic RBC crop.

    Returns ``(bgr_image, radius_x, radius_y)`` so callers can record the true
    cell footprint for IoU scoring (the crop is larger than the cell itself).
    """
    # Plasma background: warm off-white with low-frequency blotches.
    base = rng.normal(238, 6, size=(size, size, 1))
    bg = np.repeat(base, 3, axis=2).astype(np.float32)
    bg[:, :, 0] *= rng.uniform(0.96, 1.04)   # B
    bg[:, :, 2] *= rng.uniform(0.96, 1.04)   # R

    cx, cy = size / 2.0 + rng.normal(0, 1.0, size=2)
    radius = rng.uniform(0.26, 0.34) * size

    # Cell body mask with a slightly elliptical outline.
    rx, ry = radius * rng.uniform(0.92, 1.08), radius * rng.uniform(0.92, 1.08)
    yy, xx = np.mgrid[0:size, 0:size]
    dist = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2

    # RBC takes eosin (pink); add a brighter biconcave centre (pallor).
    body = dist <= 1.0
    pallor = dist <= 0.45
    cell_colour = np.array([150.0, 180.0, 225.0])  # BGR pink
    pallor_colour = np.array([185.0, 215.0, 240.0])

    out = bg.copy()
    out[body] = cell_colour
    out[pallor] = pallor_colour

    if parasitized:
        # Chromatin takes Giemsa (deep purple-blue).
        n_rings = int(rng.integers(1, 3))
        for _ in range(n_rings):
            r = radius * rng.uniform(0.22, 0.34)
            ox = cx + rng.normal(0, radius * 0.26)
            oy = cy + rng.normal(0, radius * 0.26)
            if ring_stage and rng.random() < 0.7:
                cv2.circle(out, (int(ox), int(oy)), int(r), (110.0, 60.0, 130.0),
                           max(1, int(r * 0.42)), cv2.LINE_AA)
            else:
                blob_pts = []
                for k in range(14):
                    a = 2 * np.pi * k / 14.0
                    rr = r * rng.uniform(0.75, 1.25)
                    blob_pts.append([ox + rr * np.cos(a), oy + rr * np.sin(a)])
                cv2.fillPoly(out, [np.array(blob_pts, dtype=np.int32)],
                             (95.0, 45.0, 120.0), cv2.LINE_AA)

    # Focus falloff + mild sensor noise, mimicking a real smear capture.
    if rng.random() < 0.35:
        k = int(rng.integers(1, 3)) * 2 + 1
        out = cv2.GaussianBlur(out, (k, k), 0)
    out += rng.normal(0, 5.0, size=out.shape).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8), rx, ry


def ensure_sample_dataset(
    root: str | Path,
    total: int = 200,
    image_size: int = 128,
    slides: int = 8,
    force: bool = False,
) -> list[dict]:
    """Create ``<root>/sample/{Parasitized,Uninfected}`` crops and return records."""
    root = Path(root)
    sample_root = root / "sample"
    if sample_root.exists() and not force and any(sample_root.rglob("*.png")):
        records = _build_records(sample_root)
        print(f"[sample] existing mock crops found at {sample_root} ({len(records)} images)")
        return records

    rng = np.random.default_rng(20241008)
    per_slide = max(2, total // max(1, slides))
    records: list[dict] = []

    for cls_name, label in (("Parasitized", 1), ("Uninfected", 0)):
        dest = sample_root / cls_name
        dest.mkdir(parents=True, exist_ok=True)
        n = 0
        for slide in range(slides):
            patient_id = f"slide_{slide:03d}"
            # Each synthetic slide is predominantly one class with a small
            # minority of the other, mirroring real smear composition.
            for i in range(per_slide):
                this_label = label
                if rng.random() < 0.06:  # mixed slides
                    this_label = 1 - label
                img, _, _ = _draw_rbc(rng, image_size, parasitized=bool(this_label))
                path = dest / f"{patient_id}_cell_{i:04d}.png"
                cv2.imwrite(str(path), img)
                records.append(
                    {
                        "path": str(path),
                        "label": int(this_label),
                        "patient_id": patient_id,
                        "domain": "nih",
                        "source": "synthetic_mock",
                        "license": "synthetic-no-license",
                    }
                )
                n += 1
        print(f"[sample] wrote {n} {cls_name} crops")

    return records


def _build_records(sample_root: Path) -> list[dict]:
    records = []
    for cls_dir in sorted(p for p in sample_root.iterdir() if p.is_dir()):
        label = 1 if "para" in cls_dir.name.lower() else 0
        for img in sorted(cls_dir.glob("*.png")):
            records.append(
                {
                    "path": str(img),
                    "label": label,
                    "patient_id": patient_id_from_stem(img.stem),
                    "domain": "nih",
                    "source": "synthetic_mock",
                    "license": "synthetic-no-license",
                }
            )
    return records


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate synthetic mock RBC crops")
    p.add_argument("--out", type=str, default="data/sample")
    p.add_argument("--total", type=int, default=200)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--slides", type=int, default=8)
    p.add_argument("--force", action="store_true")
    p.add_argument("--mode", type=str, default="crops", choices=["crops", "fov"],
                   help="crops = isolated 128px crops; fov = full raw field-of-view frames")
    p.add_argument("--frames", type=int, default=20,
                   help="number of raw frames to synthesise in fov mode")
    p.add_argument("--fov-size", type=int, default=900)
    p.add_argument("--parasitized-frac", type=float, default=0.12,
                   help="fraction of parasitized cells in fov mode")
    return p


def _build_fov(rng: np.random.Generator, size: int, cells: int, parasitized_frac: float,
               cell_px: int = 118) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Compose a raw, messy full-field-of-view frame.

    Deliberately includes the artefacts a phone-on-eyepiece capture has:
    vignetting, uneven illumination, a few dust specks, slight blur and a glare
    blob, so the segmenter is exercised on realistic input rather than a
    sterile patchwork.

    Returns ``(bgr_image, boxes)`` where boxes are ground-truth cell records
    used to score segmentation recall/IoU.
    """
    # Uneven plasma background with low-frequency blotches.
    bg = np.full((size, size, 3), 236.0, dtype=np.float32)
    blotch = rng.random((8, 8, 3)).astype(np.float32) * 22.0
    bg += cv2.resize(blotch, (size, size), interpolation=cv2.INTER_CUBIC)

    # Poisson-ish cell layout with mild overlap.
    placed: list[tuple[int, int]] = []
    boxes: list[dict[str, Any]] = []
    attempts = 0
    while len(placed) < cells and attempts < cells * 40:
        attempts += 1
        x = int(rng.integers(cell_px // 2, size - cell_px // 2))
        y = int(rng.integers(cell_px // 2, size - cell_px // 2))
        if any((x - px) ** 2 + (y - py) ** 2 < (cell_px * 0.62) ** 2 for px, py in placed):
            continue  # too much overlap -> would create an unresolvable clump
        placed.append((x, y))

    for i, (x, y) in enumerate(placed):
        para = rng.random() < parasitized_frac
        cell, cell_rx, cell_ry = _draw_rbc(rng, cell_px, parasitized=para)
        x0, y0 = max(0, x - cell_px // 2), max(0, y - cell_px // 2)
        x1, y1 = min(size, x0 + cell_px), min(size, y0 + cell_px)
        h, w = y1 - y0, x1 - x0
        if h <= 0 or w <= 0:
            continue
        patch = cell[:h, :w]
        # Soft circular mask so cells blend into plasma instead of pasting squares.
        yy, xx = np.mgrid[0:h, 0:w]
        c = ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / (min(h, w) / 2) ** 2
        alpha = np.clip(1.0 - c, 0.0, 1.0)[..., None]
        bg[y0:y1, x0:x1] = bg[y0:y1, x0:x1] * (1 - alpha) + patch * alpha
        # Ground truth is the CELL FOOTPRINT, not the paste rectangle: the
        # patch is padded around the disc, so using the patch bbox would make
        # perfectly-correct detections score an IoU of only ~0.30.
        bw, bh = int(round(cell_rx * 2)), int(round(cell_ry * 2))
        boxes.append(
            {
                "x": int(round(x - cell_rx)),
                "y": int(round(y - cell_ry)),
                "w": bw,
                "h": bh,
                "parasitized": bool(para),
            }
        )

    # Vignetting.
    yy, xx = np.mgrid[0:size, 0:size]
    r2 = ((xx - size / 2) ** 2 + (yy - size / 2) ** 2) / (size / 2) ** 2
    vig = 1.0 - 0.28 * np.clip(r2, 0, 1)
    bg *= vig[..., None]

    # Lens glare blob.
    if rng.random() < 0.5:
        gx, gy = int(rng.integers(0, size)), int(rng.integers(0, size))
        gr = int(rng.uniform(0.08, 0.16) * size)
        glare = np.zeros((size, size), np.float32)
        cv2.circle(glare, (gx, gy), gr, 1.0, -1)
        glare = cv2.GaussianBlur(glare, (0, 0), gr * 0.5)
        bg += (glare * 130.0)[..., None]

    # Dust specks / debris.
    for _ in range(int(rng.integers(6, 20))):
        dx, dy = int(rng.integers(0, size)), int(rng.integers(0, size))
        cv2.circle(bg, (dx, dy), int(rng.integers(1, 4)),
                   (float(rng.uniform(120, 170)),) * 3, -1, cv2.LINE_AA)

    if rng.random() < 0.4:
        bg = cv2.GaussianBlur(bg, (3, 3), 0)

    bg += rng.normal(0, 6.0, size=bg.shape)
    return np.clip(bg, 0, 255).astype(np.uint8), boxes


def ensure_phone_test_set(root: str | Path, frames: int = 20, size: int = 900,
                          force: bool = False) -> list[Path]:
    """Create raw synthetic FOV frames + ground-truth boxes under ``<root>/phone_test/``."""
    root = Path(root)
    out = root / "phone_test"
    existing = sorted(out.glob("*.jpg")) if out.exists() else []
    if existing and not force and (out / "ground_truth.json").exists():
        print(f"[fov] {len(existing)} existing raw frames in {out}")
        return existing

    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(4242)
    written: list[Path] = []
    truth: dict[str, Any] = {"images": {}, "synthetic": True}
    for i in range(frames):
        n_cells = int(rng.integers(28, 46))
        frac = float(np.clip(rng.normal(0.12, 0.06), 0.0, 0.35))
        img, boxes = _build_fov(rng, size, n_cells, frac)
        path = out / f"sample_slide_{i:03d}.jpg"
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        truth["images"][str(path)] = {"boxes": boxes, "size": size}
        written.append(path)
    (out / "ground_truth.json").write_text(json.dumps(truth, indent=2))
    print(f"[fov] wrote {len(written)} raw FOV frames to {out}")
    return written


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.mode == "fov":
        ensure_phone_test_set(args.out, frames=args.frames, size=args.fov_size,
                              force=args.force)
        return 0
    recs = ensure_sample_dataset(
        args.out, total=args.total, image_size=args.image_size,
        slides=args.slides, force=args.force,
    )
    print(f"[sample] {len(recs)} records ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())