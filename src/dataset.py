"""Dataset loading and disease-stage / patient-level splitting for TinyMalariaNet.

Critical rule (per design doc section 2): splits MUST be made at the
Slide / Patient ID level, never at the individual image level. Random image
splits leak slide-specific staining and illumination artifacts between train
and validation and inflate reported sensitivity.

Record schema
-------------
Each record is a dict:
    {
        "path":       "/abs/or/rel/path/to/crop.png",
        "label":      0 | 1,          # 0 = Uninfected, 1 = Parasitized
        "patient_id": "slide_0042",   # grouping key used for splitting
        "domain":     "nih" | "makerere" | "phone" | "bbbc",
        "source":     "NIH_Malaria_Dataset",  # optional provenance
        "license":    "CC0",                   # optional provenance
    }
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    _HAS_ALB = True
except ImportError:  # pragma: no cover - fallback path
    _HAS_ALB = False

from .model import ModelConfig

LABEL_NAMES = {0: "Uninfected", 1: "Parasitized"}
DOMAINS = ("nih", "bbbc", "makerere", "phone")
REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_image_path(path: str | Path, root: str | Path | None = None) -> Path:
    """Resolve a manifest path against the filesystem, then the repo root.

    Manifests written by ``src/prepare_data.py`` store paths relative to the
    repository root so they stay portable between machines. Resolve them so the
    data loaders keep working no matter which directory the process was started
    from.
    """
    cand = Path(path)
    if cand.is_absolute() and cand.exists():
        return cand
    if cand.exists():
        return cand
    for base in (root, REPO_ROOT):
        if base is None:
            continue
        alt = Path(base) / cand
        if alt.exists():
            return alt
    raise FileNotFoundError(f"Image not found: {path} (also tried under {REPO_ROOT})")


# --------------------------------------------------------------------------- #
# Record collection
# --------------------------------------------------------------------------- #
def load_manifest(manifest_path: str | Path) -> list[dict[str, Any]]:
    """Load records from a .json list file or .csv with a header row."""
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    if path.suffix.lower() == ".json":
        records = json.loads(path.read_text())
        if isinstance(records, dict):
            records = records.get("images", records.get("records", []))
        return [dict(r) for r in records]

    import csv

    with path.open(newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def patient_id_from_stem(stem: str) -> str:
    """Derive the Slide/Patient id from a crop filename.

    The datasets used here (NIH Kaggle crops, the synthetic sample generator,
    and crops harvested by ``src/segment.py``) all encode the *source field* in
    the filename followed by a per-cell batch index::

        slide_000_cell_0000.png                 -> slide_000
        C100P61ThinF_IMG_20150618_171318_cell_2 -> C100P61ThinF_IMG_20150618_171318
        cell_0123.png                           -> cell_0123  (already unique)

    Trimming that index is what keeps the split at slide level: without it every
    image would collapse into one giant "patient" and the train/val split would
    become degenerate (all images in one split).
    """
    stem = Path(stem).stem if "." in stem else stem
    # <field>_cell_<n> / <field>_cell<n> / <field>_tile<n> / <field>-crop<n>
    m = re.match(r"^(.*?)[ _-](?:cell|tile|crop|patch|region|fov)[ _-]?\d+$",
                 stem, flags=re.IGNORECASE)
    if m and m.group(1):
        return m.group(1)
    # <field>_<digits> (e.g. slide_000_cell_0000 -> slide_000 handled above;
    # a bare trailing counter is still ambiguous, so only trim it when the
    # remainder looks like a field id and not a directory name).
    return stem


def scan_image_folder(
    root: str | Path,
    patient_from: str = "auto",
    domain: str = "nih",
) -> list[dict[str, Any]]:
    """Infer records from the classic ``<root>/{Parasitized,Uninfected}/*.png``
    layout (as used by the NIH Kaggle dataset), or from a flat folder of crops.

    ``patient_from`` controls the slide/patient grouping key:

    * ``"auto"`` (default) - derive it from the filename via
      :func:`patient_id_from_stem`, which strips the per-cell index.
    * ``"parent"`` - use the parent folder name (parasitised class folders).
    * ``"stem_prefix"`` - use the text before the first underscore.

    The grouping key is what makes the split LEARN-resistant: crops from the
    same field share staining, illumination and focus artifacts, so they must
    never straddle train and validation.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    records: list[dict[str, Any]] = []
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        label = 1 if "para" in cls_dir.name.lower() else 0
        for img in sorted(cls_dir.glob("*")):
            if img.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                continue
            if patient_from == "parent":
                pid = img.parent.name
            elif patient_from == "stem_prefix":
                pid = img.stem.split("_")[0]
            else:
                pid = patient_id_from_stem(img.stem)
            records.append(
                {
                    "path": str(img),
                    "label": int(label),
                    "patient_id": str(pid),
                    "domain": domain,
                    "source": str(root.name),
                }
            )
    return records


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def patient_level_split(
    records: Sequence[dict[str, Any]],
    val_fraction: float = 0.15,
    seed: int = 1337,
    stratify_by_label: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split records at the PATIENT/SLIDE id level only.

    If ``stratify_by_label`` is set, patients are grouped by their dominant
    label (a slide is usually uniformly parasitised or not) so both splits keep
    a comparable positive rate.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")

    by_patient: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        by_patient.setdefault(str(rec["patient_id"]), []).append(rec)

    rng = np.random.default_rng(seed)
    val_ids: set[str] = set()

    if stratify_by_label:
        # Group patients by dominant class label.
        buckets: dict[int, list[str]] = {}
        for pid, recs in by_patient.items():
            dominant = int(round(np.mean([r["label"] for r in recs])))
            buckets.setdefault(dominant, []).append(pid)
        for _, pids in sorted(buckets.items()):
            pids = sorted(pids)
            rng.shuffle(pids)
            n_val = max(1, int(round(len(pids) * val_fraction)))
            val_ids.update(pids[:n_val])
    else:
        pids = sorted(by_patient)
        rng.shuffle(pids)
        n_val = max(1, int(round(len(pids) * val_fraction)))
        val_ids = set(pids[:n_val])

    train = [r for r in records if str(r["patient_id"]) not in val_ids]
    val = [r for r in records if str(r["patient_id"]) in val_ids]

    # Guard against dedicated-patient leakage: no id may appear in both.
    overlap = {r["patient_id"] for r in train} & {r["patient_id"] for r in val}
    if overlap:
        raise AssertionError(f"Patient leakage detected in split: {sorted(overlap)[:5]}")

    return train, val


# --------------------------------------------------------------------------- #
# Augmentations
# --------------------------------------------------------------------------- #
def _clf_params(image_size: int, train: bool, domain: str = "nih") -> Any:
    """Build the albumentations pipeline.

    ``domain="phone"`` layers on the smartphone-eyepiece degradations required
    by Stage 2 of the training plan: chromatic aberration, vignetting, blur and
    lens glare. NIH/Makerere slides get the milder stain/jitter augmentations.
    """
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if not train:
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )

    ops: list[Any] = [
        A.RandomResizedCrop(size=(image_size, image_size), scale=(0.7, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        # chromatin/RBC stain variation (Giemsa vs Wright)
        A.HueSaturationValue(
            hue_shift_limit=10, sat_shift_limit=25, val_shift_limit=15, p=0.6
        ),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
        # sensor noise (std as a fraction of 255)
        A.GaussNoise(std_range=(0.02, 0.16), p=0.35),
    ]

    if domain in {"phone", "makerere"}:
        # Smartphone eyepiece optics. Applied with low probability each so the
        # model never sees an exclusively-degraded batch.
        ops += [
            A.RGBShift(r_shift_limit=18, g_shift_limit=12, b_shift_limit=18, p=0.4),
            A.ChannelShuffle(p=0.08),  # crude chromatic-aberration proxy
            A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.3),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=5, p=1.0),
                    A.GaussianBlur(blur_limit=5, p=1.0),
                    A.Defocus(radius=(2, 4), alias_blur=(0.1, 0.4), p=1.0),
                ],
                p=0.4,
            ),
            A.RandomShadow(num_shadows_limit=(1, 2), shadow_dimension=5, p=0.25),
            # Lens flare / glare blob
            A.RandomSunFlare(
                flare_roi=(0.0, 0.0, 1.0, 1.0),
                src_radius=30,
                num_flare_circles_range=(1, 2),
                p=0.12,
            ),
        ]

    ops += [A.Normalize(mean=mean, std=std), ToTensorV2()]
    return A.Compose(ops)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class MalariaDataset(Dataset):
    """Torch dataset over labelled RBC crops."""

    def __init__(
        self,
        records: Iterable[dict[str, Any]],
        image_size: int = 128,
        train: bool = False,
        domain: str = "nih",
        cache: bool = False,
    ) -> None:
        if not _HAS_ALB:
            raise ImportError(
                "albumentations is required for MalariaDataset: pip install albumentations"
            )
        self.records = list(records)
        self.transform = _clf_params(image_size=image_size, train=train, domain=domain)
        self._cache: dict[str, np.ndarray] = {}
        self.cache = cache

    def __len__(self) -> int:
        return len(self.records)

    def _read(self, path: str | Path) -> np.ndarray:
        key = str(path)
        if self.cache and key in self._cache:
            return self._cache[key]
        # Read with alpha dropped, then force RGB. Unchanged path returns None.
        arr = cv2.imread(str(resolve_image_path(path)), cv2.IMREAD_COLOR)
        if arr is None:
            raise FileNotFoundError(f"Could not decode image: {path}")
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        if self.cache:
            self._cache[key] = rgb
        return rgb

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rec = self.records[idx]
        image = self._read(rec["path"])
        label = float(rec["label"])
        out = self.transform(image=image)["image"]
        return out, torch.tensor([label], dtype=torch.float32)

    def label_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {0: 0, 1: 0}
        for r in self.records:
            counts[int(r["label"])] += 1
        return counts


# --------------------------------------------------------------------------- #
# Dataloaders
# --------------------------------------------------------------------------- #
@dataclass
class LoaderSpec:
    batch_size: int = 64
    num_workers: int = 0
    image_size: int = 128
    domain: str = "nih"
    cache: bool = False


def build_dataloaders(
    records: Sequence[dict[str, Any]],
    cfg: ModelConfig | None = None,
    spec: LoaderSpec | None = None,
    val_fraction: float = 0.15,
    seed: int = 1337,
) -> tuple[DataLoader, DataLoader, dict[str, Any]]:
    """Create patient-split train/val loaders.

    Returns ``(train_loader, val_loader, stats)`` where stats records the split
    counts and patient ids for auditability.
    """
    cfg = cfg or ModelConfig()
    spec = spec or LoaderSpec()
    image_size = spec.image_size or cfg.image_size

    # Detection of a single dominant domain for augmentation selection.
    domains = {r.get("domain", "nih") for r in records}
    default_domain = "phone" if "phone" in domains else (spec.domain or "nih")

    train_recs, val_recs = patient_level_split(records, val_fraction=val_fraction, seed=seed)

    train_ds = MalariaDataset(
        train_recs, image_size=image_size, train=True, domain=default_domain, cache=spec.cache
    )
    val_ds = MalariaDataset(val_recs, image_size=image_size, train=False, domain=default_domain)

    train_loader = DataLoader(
        train_ds,
        batch_size=spec.batch_size,
        shuffle=True,
        num_workers=spec.num_workers,
        pin_memory=False,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=spec.batch_size,
        shuffle=False,
        num_workers=spec.num_workers,
        pin_memory=False,
    )

    stats = {
        "train_images": len(train_ds),
        "val_images": len(val_ds),
        "train_patients": len({r["patient_id"] for r in train_recs}),
        "val_patients": len({r["patient_id"] for r in val_recs}),
        "train_label_counts": train_ds.label_counts(),
        "val_label_counts": val_ds.label_counts(),
        "domain_aug": default_domain,
        "val_fraction": val_fraction,
        "seed": seed,
    }
    return train_loader, val_loader, stats


if __name__ == "__main__":
    # Tiny synthetic self-check of the split logic (no image IO needed).
    fake = [
        {"path": f"x{i}.png", "label": i % 2, "patient_id": f"p{i // 20}", "domain": "nih"}
        for i in range(200)
    ]
    tr, va = patient_level_split(fake, val_fraction=0.2)
    print(f"train={len(tr)} val={len(va)}")
    print("train patients:", len({r['patient_id'] for r in tr}))
    print("val patients:", len({r['patient_id'] for r in va}))
