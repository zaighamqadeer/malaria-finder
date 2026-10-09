"""Parse the Lacuna Malaria Dataset annotations into classifier crops.

Verified against the published dataset while writing this module:

* The annotation released with the dataset is ``Labels-CSV.csv`` with columns
  ``Image_name, xmin, ymin, width, height, Class`` where the coordinates are
  **absolute pixels** (origin top-left).
* Class names observed across the published archives::

      Parasitized cell   9,637 boxes
      Artifact           1,703 boxes
      Trophozoite          534 boxes
      WBC                  402 boxes
      Gametocyte             12 boxes

  (counts are for the Uganda thin-smear archive; the parser counts at runtime
  rather than trusting these numbers.)
* **The CSV contains exact duplicate rows** — 4,796 of 12,288 rows in that same
  archive are byte-identical repeats. They are de-duplicated before cropping,
  which would otherwise have doubled every positive sample.
* A parallel ``Labels-YOLO/<stem>.txt`` tree exists, but the archive ships no
  ``classes.txt`` (so the numeric ids cannot be resolved to names) and the box
  geometry does not agree with the CSV. ``Labels-CSV.csv`` is therefore treated
  as the authoritative source; use ``--yolo`` to parse the YOLO tree instead,
  supplying ``--classes`` for the id -> name mapping.

Mapping to the binary target
---------------------------
    Parasitized cell, Trophozoite, Gametocyte -> 1 (Parasitized)
    Artifact, WBC                              -> 0 (Uninfected, hard negatives)
    un-annotated background                    -> 0 (healthy RBCs, opt-in)

``Trophozoite`` and ``Gametocyte`` are malaria parasite life-cycle stages found
*inside* an RBC, so a cell carrying one is parasitized. ``Artifact`` and ``WBC``
have no parasites: they are deliberately kept as **hard negatives** because
debris and white blood cells are the classic sources of false alarms in a
cell-level malaria classifier.

Usage
-----
    python src/parse_lacuna.py --src data/lacuna --out data/processed/lacuna_crops
    python src/parse_lacuna.py --src data/lacuna --negatives-from-background 0.5
    python src/parse_lacuna.py --src data/lacuna --max-images 40 --dry-run
"""

from __future__ import annotations

# Support BOTH documented entry points: `python src/parse_lacuna.py ...` and
# `python -m src.parse_lacuna`.
if __package__ in (None, ""):  # pragma: no cover - exercised via the shell
    import os
    import sys as _sys

    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "src"  # noqa: A001

import argparse
import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from .dataset import patient_id_from_stem

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_POSITIVE = ("Parasitized cell", "Trophozoite", "Gametocyte")
DEFAULT_NEGATIVE = ("Artifact", "WBC")
DEFAULT_CROP_SIZE = 128


# --------------------------------------------------------------------------- #
# Label model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Box:
    """A bounding box in absolute pixel coordinates (origin top-left)."""

    x: int
    y: int
    w: int
    h: int
    source_class: str = ""

    def clipped(self, width: int, height: int) -> "Box":
        x0, y0 = max(0, self.x), max(0, self.y)
        x1, y1 = min(width, self.x + self.w), min(height, self.y + self.h)
        if x1 <= x0 or y1 <= y0:
            return Box(0, 0, 0, 0, self.source_class)
        return Box(x0, y0, x1 - x0, y1 - y0, self.source_class)

    @property
    def area(self) -> int:
        return max(0, self.w) * max(0, self.h)

    def overlaps(self, other: "Box", margin: int = 0) -> bool:
        return not (self.x - margin > other.x + other.w
                    or other.x - margin > self.x + self.w
                    or self.y - margin > other.y + other.h
                    or other.y - margin > self.y + self.h)


def _norm_key(value: str) -> str:
    return "".join(str(value).lower().split())


def build_class_map(positive: Sequence[str], negative: Sequence[str]) -> dict[str, int]:
    """Normalised class name -> binary label."""
    out: dict[str, int] = {}
    for name in positive:
        out[_norm_key(name)] = 1
    for name in negative:
        out[_norm_key(name)] = 0
    return out


# --------------------------------------------------------------------------- #
# Annotation readers
# --------------------------------------------------------------------------- #
def read_labels_csv(path: Path) -> list[dict[str, Any]]:
    """Read ``Labels-CSV.csv`` and drop exact duplicate rows.

    De-duplication is required: the published CSVs repeat a large fraction of
    their rows verbatim, which would otherwise duplicate entire classes of
    crop.
    """
    seen: set[tuple] = set()
    rows: list[dict[str, Any]] = []
    duplicates = 0
    with path.open(newline="", encoding="utf-8-sig") as fh:
        for raw in csv.DictReader(fh):
            key = (raw.get("Image_name"), raw.get("xmin"), raw.get("ymin"),
                   raw.get("width"), raw.get("height"), raw.get("Class"))
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            rows.append(
                {
                    "image": raw["Image_name"].strip(),
                    "x": int(round(float(raw["xmin"]))),
                    "y": int(round(float(raw["ymin"]))),
                    "w": int(round(float(raw["width"]))),
                    "h": int(round(float(raw["height"]))),
                    "class": raw["Class"].strip(),
                }
            )
    if duplicates:
        print(f"[parse] {path.name}: dropped {duplicates:,} duplicate rows "
              f"({100*duplicates/max(1,len(rows)+duplicates):.1f}%)")
    return rows


def read_labels_yolo(path: Path, classes: Sequence[str]) -> list[dict[str, Any]]:
    """Read a YOLO-format label file (needs an explicit id -> name mapping)."""
    out: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cid = int(parts[0])
        if cid >= len(classes):
            continue
        out.append(
            {
                "image": path.stem + ".jpg",
                "cx": float(parts[1]),
                "cy": float(parts[2]),
                "w_norm": float(parts[3]),
                "h_norm": float(parts[4]),
                "class": classes[cid],
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Crop extraction
# --------------------------------------------------------------------------- #
def square_crop(image: np.ndarray, box: Box, size: int,
                pad_frac: float = 0.18) -> np.ndarray | None:
    """Square-padded crop around ``box`` resized to ``size x size``.

    The box marks the *cell*, so the crop is padded by ``pad_frac`` on the long
    side to give the classifier context without pulling in neighbouring cells.
    """
    side = int(round(max(box.w, box.h) * (1.0 + pad_frac)))
    if side <= 0:
        return None
    cx, cy = box.x + box.w / 2.0, box.y + box.h / 2.0
    # Desired window in image coordinates (may fall outside the frame).
    wx0, wy0 = int(round(cx - side / 2.0)), int(round(cy - side / 2.0))
    wx1, wy1 = wx0 + side, wy0 + side
    h, w = image.shape[:2]
    # Intersection with the image, clamped. Clamping must happen BEFORE slicing:
    # a raw negative index in NumPy counts from the end of the axis and would
    # silently read the wrong (or an empty) region.
    ox0, oy0 = max(0, wx0), max(0, wy0)
    ox1, oy1 = min(w, wx1), min(h, wy1)
    if ox1 <= ox0 or oy1 <= oy0:
        return None
    fill = int(np.median(image)) if image.size else 0
    canvas = np.full((side, side, 3), fill, dtype=np.uint8)
    canvas[oy0 - wy0: oy0 - wy0 + (oy1 - oy0),
           ox0 - wx0: ox0 - wx0 + (ox1 - ox0)] = image[oy0:oy1, ox0:ox1]
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA) \
        if (side, side) != (size, size) else canvas


def background_tiles(image: np.ndarray, width: int, height: int, size: int,
                     occupied: Sequence[Box], n: int,
                     rng: random.Random) -> list[Box]:
    """Sample ``n`` ``size``-sized tiles that contain no annotated box.

    These are the "healthy RBC" negatives: the Lacuna release does not annotate
    uninfected red blood cells, so the only way to obtain them is to sample
    regions of the field that carry no annotation. A tile is rejected if it
    overlaps any annotated box by more than half its area.
    """
    boxes: list[Box] = []
    if width <= size or height <= size:
        return boxes
    threshold = 0.5 * size * size
    for _ in range(n * 25):
        if len(boxes) >= n:
            break
        x = rng.randint(0, width - size)
        y = rng.randint(0, height - size)
        tile = Box(x, y, size, size, "background")
        bad = False
        for other in occupied:
            if not tile.overlaps(other):
                continue
            inter_w = min(x + size, other.x + other.w) - max(x, other.x)
            inter_h = min(y + size, other.y + other.h) - max(y, other.y)
            if max(0, inter_w) * max(0, inter_h) > threshold:
                bad = True
                break
        if not bad:
            boxes.append(tile)
    return boxes


def write_crop(crop: np.ndarray, out_root: Path, label: int, patient_id: str,
               index: int, source: str) -> Path:
    class_dir = {0: "Uninfected", 1: "Parasitized"}[label]
    dest_dir = out_root / class_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    # patient_id is kept in the filename so downstream tools (and
    # src/dataset.py:patient_id_from_stem) recover the field-of-view grouping.
    name = f"{patient_id}_lacuna_{index:06d}.png"
    path = dest_dir / name
    cv2.imwrite(str(path), crop)
    return path


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
@dataclass
class ParseConfig:
    src: Path
    out: Path
    crop_size: int = DEFAULT_CROP_SIZE
    positive: tuple[str, ...] = tuple(DEFAULT_POSITIVE)
    negative: tuple[str, ...] = tuple(DEFAULT_NEGATIVE)
    pad_frac: float = 0.18
    max_images: int | None = None
    negatives_from_background: float = 0.0
    yolo: bool = False
    yolo_classes: tuple[str, ...] = ()
    min_box_px: int = 4
    max_box_frac: float = 0.25
    dry_run: bool = False
    manifest_name: str = "manifest.json"
    seed: int = 1337
    manifest_csv: bool = False


def discover_archives(src: Path) -> tuple[dict[str, Path], dict[str, list[dict]]]:
    """Find every image/annotation pair under ``src``.

    Returns ``(image_paths, annotations_by_image)``.
    """
    images = {p.name: p for p in src.rglob("images/*")
              if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    if not images:
        # some releases keep images at the archive root
        images = {p.name: p for p in src.rglob("*")
                  if p.suffix.lower() in {".jpg", ".jpeg"} and
                  "images/" not in str(p).replace("\\", "/")}
    annotations: dict[str, list[dict]] = defaultdict(list)
    for csv_path in sorted(src.rglob("Labels-CSV.csv")):
        for row in read_labels_csv(csv_path):
            annotations[row["image"]].append(row)
    return images, annotations


def parse_dataset(cfg: ParseConfig) -> dict[str, Any]:
    src = Path(cfg.src)
    if not src.exists():
        raise FileNotFoundError(
            f"{src} not found. Download the dataset first:\n"
            f"  python src/download_lacuna.py --files Thin_Uganda.rar --extract "
            f"--extract-to {cfg.src}"
        )

    class_map = build_class_map(cfg.positive, cfg.negative)
    unknown_classes: dict[str, int] = defaultdict(int)
    images, annotations = discover_archives(src)
    if not images:
        raise FileNotFoundError(f"No images found under {src}/images")

    rng = random.Random(cfg.seed)
    image_names = sorted(images)
    if cfg.max_images:
        rng.shuffle(image_names)
        image_names = image_names[: cfg.max_images]

    out_root = Path(cfg.out)
    if not cfg.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "images_parsed": 0,
        "images_without_annotations": 0,
        "boxes_seen": 0,
        "boxes_kept": 0,
        "crops_written": 0,
        "duplicates_skipped": 0,
        "background_tiles": 0,
        "class_counts": defaultdict(int),
        "label_counts": {"0": 0, "1": 0},
    }
    seen_spatial: set[tuple[str, int, int, int, int]] = set()

    for idx, name in enumerate(image_names):
        img_path = images[name]
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"[parse] WARNING could not decode {img_path}, skipping")
            continue
        height, width = image.shape[:2]
        # Each smartphone capture is a distinct field of view -> its own
        # "patient". Crops from one field share illumination, focus and staging,
        # so grouping here is what keeps the train/val split honest.
        patient_id = patient_id_from_stem(Path(name).stem)
        boxes_raw = annotations.get(name, [])
        if not boxes_raw:
            stats["images_without_annotations"] += 1
        boxes: list[Box] = []
        for row in boxes_raw:
            stats["boxes_seen"] += 1
            cls = row["class"]
            key = _norm_key(cls)
            if key not in class_map:
                unknown_classes[cls] += 1
                continue
            if cfg.yolo:
                cx, cy = row["cx"] * width, row["cy"] * height
                w, h = row["w_norm"] * width, row["h_norm"] * height
                box = Box(int(cx - w / 2), int(cy - h / 2), int(w), int(h), cls)
            else:
                box = Box(row["x"], row["y"], row["w"], row["h"], cls)
            box = box.clipped(width, height)
            if box.w < cfg.min_box_px or box.h < cfg.min_box_px:
                continue
            if box.area > cfg.max_box_frac * width * height:
                continue  # annotation the size of a third of the frame: not a cell
            signature = (name, box.x, box.y, box.w, box.h)
            if signature in seen_spatial:
                stats["duplicates_skipped"] += 1
                continue
            seen_spatial.add(signature)
            stats["boxes_kept"] += 1
            stats["class_counts"][cls] += 1
            boxes.append(box)

        crops_for_image: list[tuple[np.ndarray, int, str]] = []
        for box in boxes:
            crop = square_crop(image, box, cfg.crop_size, cfg.pad_frac)
            if crop is not None:
                crops_for_image.append((crop, class_map[_norm_key(box.source_class)],
                                        box.source_class))

        if cfg.negatives_from_background > 0 and crops_for_image:
            want = int(round(cfg.negatives_from_background * len(crops_for_image)))
            for tile in background_tiles(image, width, height, cfg.crop_size,
                                         boxes, want, rng):
                crop = square_crop(image, tile, cfg.crop_size, cfg.pad_frac)
                if crop is not None:
                    crops_for_image.append((crop, 0, "background"))
                    stats["background_tiles"] += 1

        for n, (crop, label, source_class) in enumerate(crops_for_image):
            stats["label_counts"][str(label)] += 1
            if cfg.dry_run:
                continue
            path = write_crop(crop, out_root, label, patient_id, n, source_class)
            records.append(
                {
                    "path": str(path.relative_to(REPO_ROOT)) if
                    path.is_relative_to(REPO_ROOT) else str(path),
                    "label": int(label),
                    "patient_id": patient_id,
                    "domain": "phone",
                    "source": "Lacuna_Malaria_Datasets",
                    "source_class": source_class,
                    "source_image": name,
                    "license": "CC BY 4.0",
                }
            )
            stats["crops_written"] += 1

        stats["images_parsed"] += 1
        if stats["images_parsed"] % 100 == 0:
            print(f"[parse] {stats['images_parsed']}/{len(image_names)} fields, "
                  f"{stats['crops_written']} crops so far")

    # ---- manifest -------------------------------------------------------- #
    manifest = {
        "dataset": "lacuna_phone",
        "domain": "phone",
        "license": "CC BY 4.0",
        "source": "Lacuna Malaria Datasets (Harvard Dataverse, "
                  "doi:10.7910/DVN/VEADSE, Makerere AI Lab)",
        "positive_classes": list(cfg.positive),
        "negative_classes": list(cfg.negative),
        "crop_size": cfg.crop_size,
        "negatives_from_background": cfg.negatives_from_background,
        "num_crops": len(records),
        "num_fields": len({r["patient_id"] for r in records}),
        "label_counts": {
            "0": sum(1 for r in records if int(r["label"]) == 0),
            "1": sum(1 for r in records if int(r["label"]) == 1),
        },
        "source_class_counts": {
            "Parasitized": stats["class_counts"].get("Parasitized cell", 0),
            "Trophozoite": stats["class_counts"].get("Trophozoite", 0),
            "Gametocyte": stats["class_counts"].get("Gametocyte", 0),
            "Artifact": stats["class_counts"].get("Artifact", 0),
            "WBC": stats["class_counts"].get("WBC", 0),
            "background": stats["background_tiles"],
        },
        "images_parsed": stats["images_parsed"],
        "images_without_annotations": stats["images_without_annotations"],
        "unmapped_classes": dict(sorted(unknown_classes.items())),
        "images": records,
    }

    _print_summary(manifest, cfg, unknown_classes)

    if not cfg.dry_run:
        manifest_path = Path(cfg.out) / cfg.manifest_name
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"[parse] manifest -> {manifest_path}")
        if cfg.manifest_csv:
            csv_path = Path(cfg.out) / "manifest.csv"
            with csv_path.open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(records[0].keys()) if
                                        records else ["path"])
                writer.writeheader()
                writer.writerows(records)
            print(f"[parse] manifest -> {csv_path}")

    return manifest


def _print_summary(manifest: dict[str, Any], cfg: ParseConfig,
                   unknown: dict[str, int]) -> None:
    print()
    print("[parse] ---------------- summary ----------------")
    print(f"  fields parsed          : {manifest['images_parsed']}")
    print(f"  fields w/o annotations : {manifest['images_without_annotations']}")
    print(f"  crops                   : {manifest['num_crops']}")
    print(f"  distinct fields (slides): {manifest['num_fields']}")
    print(f"  Uninfected (0)          : {manifest['label_counts']['0']}")
    print(f"  Parasitized (1)         : {manifest['label_counts']['1']}")
    print("  by source class:")
    for cls, n in manifest["source_class_counts"].items():
        print(f"    {cls:15s} {n:>8,}")
    if unknown:
        print(f"  unmapped classes (skipped): {dict(unknown)}")
    if manifest["num_crops"] and manifest["label_counts"]["1"] == 0:
        print("  WARNING: no positive crops - check --positive against the "
              "class names in Labels-CSV.csv")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Turn the Lacuna Malaria Dataset annotations into "
                    "128x128 classifier crops for ImageFolder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--src", default="data/lacuna",
                   help="directory holding images/ and Labels-CSV.csv")
    p.add_argument("--out", default="data/processed/lacuna_crops")
    p.add_argument("--crop-size", type=int, default=DEFAULT_CROP_SIZE)
    p.add_argument("--positive", default=",".join(DEFAULT_POSITIVE),
                   help="classes mapped to label 1")
    p.add_argument("--negative", default=",".join(DEFAULT_NEGATIVE),
                   help="classes mapped to label 0")
    p.add_argument("--pad-frac", type=float, default=0.18,
                   help="context padding around each cell, as a fraction of the box")
    p.add_argument("--max-images", type=int, default=None,
                   help="only parse N fields (useful for a smoke test)")
    p.add_argument("--negatives-from-background", type=float, default=0.0,
                   help="ratio of extra negative crops sampled from "
                        "un-annotated parts of the field (0 = off)")
    p.add_argument("--yolo", action="store_true",
                   help="parse Labels-YOLO/*.txt instead of Labels-CSV.csv")
    p.add_argument("--yolo-classes", default=None,
                   help="comma-separated class names for YOLO ids 0..N-1")
    p.add_argument("--min-box-px", type=int, default=4)
    p.add_argument("--max-box-frac", type=float, default=0.25)
    p.add_argument("--dry-run", action="store_true",
                   help="report statistics without writing any crops")
    p.add_argument("--manifest-csv", action="store_true",
                   help="also write a flat manifest.csv")
    p.add_argument("--seed", type=int, default=1337)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = ParseConfig(
        src=Path(args.src),
        out=Path(args.out),
        crop_size=args.crop_size,
        positive=tuple(s.strip() for s in args.positive.split(",") if s.strip()),
        negative=tuple(s.strip() for s in args.negative.split(",") if s.strip()),
        pad_frac=args.pad_frac,
        max_images=args.max_images,
        negatives_from_background=args.negatives_from_background,
        yolo=args.yolo,
        yolo_classes=tuple(s.strip() for s in (args.yolo_classes or "").split(",")
                           if s.strip()),
        min_box_px=args.min_box_px,
        max_box_frac=args.max_box_frac,
        dry_run=args.dry_run,
        manifest_csv=args.manifest_csv,
        seed=args.seed,
    )
    if cfg.yolo and not cfg.yolo_classes:
        print("[parse] --yolo requires --yolo-classes (the archive has no "
              "classes.txt); falling back to Labels-CSV.csv")
        cfg.yolo = False
    parse_dataset(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
