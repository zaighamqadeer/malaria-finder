"""Dataset acquisition + manifest builder for TinyMalariaNet.

``src/train.py`` and ``src/quantize.py`` consume *manifests* (JSON lists of
crop records) rather than raw archives, so every real dataset has to be staged
into ``data/<key>/`` and indexed first. This module is the bridge:

``python src/prepare_data.py --dataset nih_full --local-dir ~/Downloads/cell_images``
    Index an already-downloaded dataset.

``python src/prepare_data.py --dataset makerere_phone --url https://.../frames.zip``
    Download + unpack an archive, then index it.

``python src/prepare_data.py --dataset bbbc041 --url ... --limit 200``
    Download raw FOV frames and (optionally) run the OpenCV segmenter over them
    to produce classifier crops at slide level.

Record schema produced (exactly what ``src/dataset.py`` expects)::

    {"path": "data/nih/crops/Parasitized/C100P61ThinF_..._cell_2.png",
     "label": 1,
     "patient_id": "C100P61ThinF_IMG_20150618_171318",   # slide / field id
     "domain": "nih",
     "source": "NIH_Malaria_Dataset",
     "license": "CC0"}

Paths are stored relative to the repository root so the manifest stays portable
across machines (``src/dataset.py`` resolves them against the repo root).

Licensing
---------
Only stage a dataset here once you have read its license. In particular,
**BBBC041 is CC BY-NC-SA 3.0, not CC BY 3.0** as the original design note
assumed: the NC (non-commercial) and SA (share-alike) clauses make it
incompatible with redistributing Apache-2.0 trained weights. It is listed for
research use only.
"""

from __future__ import annotations

# Support BOTH documented entry points: `python src/prepare_data.py ...` and
# `python -m src.prepare_data`.
if __package__ in (None, ""):  # pragma: no cover - exercised via the shell
    import os
    import sys as _sys

    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "src"  # noqa: A001

import argparse
import hashlib
import json
import shutil
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------- #
# Dataset registry (kept in sync with src/train.py::DATASETS)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SourceInfo:
    key: str
    domain: str
    license: str
    source_name: str
    download_hint: str
    commercial_use_ok: bool = True


SOURCES: dict[str, SourceInfo] = {
    "nih_full": SourceInfo(
        "nih_full", "nih", "CC0",
        "NIH_Malaria_Dataset",
        "Kaggle 'iarunava/cell-images-for-detecting-malaria' (27,558 crops) or the "
        "NIH LHNCBC release (pub9352).",
    ),
    "nih_sample": SourceInfo(
        "nih_sample", "nih", "synthetic-no-license",
        "synthetic_mock",
        "Generated locally by `python src/make_sample_data.py --mode crops`.",
        commercial_use_ok=False,
    ),
    "makerere_phone": SourceInfo(
        "makerere_phone", "phone", "CC BY 4.0",
        "Makerere_LacunaFund_Smartphone",
        "Makerere AI Health Lab / Lacuna Fund smartphone eyepiece smears "
        "(registration-gated; download the release then pass --local-dir).",
    ),
    "bbbc041": SourceInfo(
        "bbbc041", "makerere", "CC BY-NC-SA 3.0",
        "Broad_BBBC041",
        "https://data.broadinstitute.org/bbbc/BBBC041/malaria.zip (~2.3 GB). "
        "NON-COMMERCIAL: do not mix into Apache-2.0 redistributed weights.",
        commercial_use_ok=False,
    ),
    "mp_idb": SourceInfo(
        "mp_idb", "makerere", "MIT",
        "MP-IDB",
        "https://github.com/cosmoimdmp/malaria-parasite-mp-idb (MIT).",
    ),
}

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
CLASS_FOLDERS = {
    "parasitized": 1, "parasitised": 1, "parasite": 1, "positive": 1, "pos": 1,
    "uninfected": 0, "uninfected ": 0, "negative": 0, "neg": 0, "healthy": 0,
}


def _registry_manifest_path(dataset_key: str) -> Path | None:
    """Where ``src/train.py`` expects this dataset's manifest to live.

    Returns ``None`` when the key is not in the trainer's registry, so callers
    fall back to their own default location. Imported lazily to avoid importing
    torch at `--list` time.
    """
    try:
        from .train import DATASETS, REPO_ROOT
    except Exception:  # pragma: no cover - torch missing, run as a script, ...
        return None
    spec = DATASETS.get(dataset_key)
    if spec is None or not spec.manifest:
        return None
    return REPO_ROOT / spec.manifest


def _rime(message: str = "", end: str = "\n") -> None:  # pragma: no cover - cosmetic
    print(message, end=end, flush=True)


# --------------------------------------------------------------------------- #
# Download / extract
# --------------------------------------------------------------------------- #
def download(url: str, dest: Path, chunk: int = 1 << 20) -> Path:
    """Stream ``url`` to ``dest`` with a progress line."""
    from urllib.request import Request, urlopen

    dest.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "TinyMalariaNet-data-prep/1.0"})
    with urlopen(req, timeout=120) as resp, dest.open("wb") as fh:  # noqa: S310
        total = int(resp.headers.get("content-length", 0))
        seen = 0
        while True:
            block = resp.read(chunk)
            if not block:
                break
            fh.write(block)
            seen += len(block)
            if total:
                pct = 100.0 * seen / total
                _rime(f"\r[download] {seen / 1e6:8.1f}/{total / 1e6:.1f} MB ({pct:5.1f}%)",
                      end="")
    _rime(f"\r[download] done -> {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def sha256_of(path: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_archive(archive: Path, dest: Path) -> Path:
    """Unpack .zip / .tar / .tar.gz / .tgz into ``dest`` and return it."""
    dest.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            tf.extractall(dest, filter="data")
    else:
        raise ValueError(f"Unsupported archive type: {archive}")
    _rime(f"[extract] {archive.name} -> {dest}")
    return dest


def _looks_like_archive(path: Path) -> bool:
    return zipfile.is_zipfile(path) or tarfile.is_tarfile(path)


# --------------------------------------------------------------------------- #
# Record collection
# --------------------------------------------------------------------------- #
def _label_for(path: Path) -> int | None:
    """Infer 0/1 from the path, preferring an explicit class folder name."""
    parts = [p.name.lower().strip() for p in path.parents][:3]
    for name in parts:
        if name in CLASS_FOLDERS:
            return CLASS_FOLDERS[name]
    for name in parts:  # substring fallback e.g. "Parasitized_crops"
        for key, label in CLASS_FOLDERS.items():
            if key.strip() and key in name:
                return label
    return None


def collect_records(
    root: Path,
    domain: str,
    source_name: str,
    license: str,
    labels_subdir: str | None = None,
    label_map: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Walk ``root`` and build crop records.

    Two layouts are supported:

    1. ``root/{Parasitized,Uninfected}/**/*.png`` - the classic NIH layout; the
       label comes from the folder and the patient id from the filename.
    2. A flat tree of frames plus ``labels_subdir`` - a CSV/JSON mapping
       ``filename -> label`` (used for BBBC041-style releases).
    """
    from .dataset import patient_id_from_stem

    records: list[dict[str, Any]] = []

    if label_map:
        for img in sorted(p for p in root.rglob("*")
                          if p.suffix.lower() in IMAGE_SUFFIXES):
            label = label_map.get(img.name)
            if label is None:
                continue
            records.append(
                {
                    "path": str(_relative(img)),
                    "label": int(label),
                    # Frames are whole fields, so every frame is its own
                    # "patient" - which keeps the slide-level split honest.
                    "patient_id": patient_id_from_stem(img.stem),
                    "domain": domain,
                    "source": source_name,
                    "license": license,
                }
            )
        return records

    for img in sorted(p for p in root.rglob("*")
                      if p.suffix.lower() in IMAGE_SUFFIXES):
        label = _label_for(img)
        if label is None:
            continue
        records.append(
            {
                "path": str(_relative(img)),
                "label": int(label),
                "patient_id": patient_id_from_stem(img.stem),
                "domain": domain,
                "source": source_name,
                "license": license,
            }
        )
    return records


def _relative(path: Path) -> Path:
    """Make ``path`` relative to the repo root when possible (else absolute)."""
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve())
    except ValueError:
        return path.resolve()


def load_label_map(labels_file: Path) -> dict[str, int]:
    """Load ``filename -> label`` from a .csv (with a header) or .json file."""
    suffix = labels_file.suffix.lower()
    if suffix == ".json":
        raw = json.loads(labels_file.read_text())
        out: dict[str, int] = {}
        for k, v in raw.items():
            out[Path(str(k)).name] = int(v) if not isinstance(v, str) else \
                CLASS_FOLDERS.get(v.strip().lower(), 0)
        return out
    import csv
    with labels_file.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return {}
    keys = rows[0].keys()
    name_col = next((k for k in keys if k.lower() in {"filename", "file", "image",
                                                     "name", "frame"}), keys[0])
    label_col = next((k for k in keys if k.lower() in {"label", "class", "y",
                                                       "infected"}), None)
    if label_col is None:  # no explicit label column: fall back to folder names
        return {}
    out = {}
    for row in rows:
        name = Path(str(row[name_col])).name
        val = str(row[label_col]).strip().lower()
        if val in CLASS_FOLDERS:
            out[name] = CLASS_FOLDERS[val]
        else:
            try:
                out[name] = int(float(val))
            except ValueError:
                continue
    return out


# --------------------------------------------------------------------------- #
# Manifest writing
# --------------------------------------------------------------------------- #
def write_manifest(records: list[dict[str, Any]], info: SourceInfo,
                   manifest_path: Path, extra: dict[str, Any] | None = None) -> Path:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": info.key,
        "domain": info.domain,
        "license": info.license,
        "source": info.source_name,
        "commercial_use_permitted": info.commercial_use_ok,
        "num_images": len(records),
        "num_patients": len({r["patient_id"] for r in records}),
        "label_counts": {
            "0": sum(1 for r in records if int(r["label"]) == 0),
            "1": sum(1 for r in records if int(r["label"]) == 1),
        },
        "images": records,
    }
    if extra:
        payload.update(extra)
    manifest_path.write_text(json.dumps(payload, indent=2))
    _rime(f"[manifest] {manifest_path} -> {len(records)} images, "
          f"{payload['num_patients']} slides")
    return manifest_path


def _dedupe_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Drop records whose *filename* was already indexed.

    ``root.rglob()`` never yields the same path twice, so duplicates mean two
    copies of the tree exist. They are distinct files, so path identity will not
    catch them - but a crop is identified by its filename (``src/dataset.py``
    derives its slide id from exactly that), so the basename is the right key.

    This matters: the NIH Kaggle archive unzipped twice (or into a directory
    that already held an extraction) reports 55,116 images instead of 27,558.
    The set is silently doubled, every epoch sees each crop twice, and the
    patient-level split then puts the *same* image in train and val.
    """
    # Key on (label, filename): basename alone can collide across the two class
    # folders, and collapsing those would silently drop real training data.
    seen: dict[tuple[int, str], int] = {}
    unique: list[dict[str, Any]] = []
    for rec in records:
        key = (int(rec.get("label", -1)), Path(rec["path"]).name)
        if key in seen:
            seen[key] += 1
            continue
        seen[key] = 1
        unique.append(rec)
    return unique, len(records) - len(unique)


def existing_counts(root: Path) -> tuple[int, int]:
    records = collect_records(root, "x", "x", "x")
    return (sum(1 for r in records if int(r["label"]) == 0),
            sum(1 for r in records if int(r["label"]) == 1))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fetch / index a dataset and write data/<key>/manifest.json",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", default=None, choices=sorted(SOURCES),
                   help="dataset registry key (not needed with --list)")
    p.add_argument("--local-dir", default=None,
                   help="existing directory to index instead of downloading")
    p.add_argument("--url", default=None, help="archive or directory URL to fetch")
    p.add_argument("--download-dir", default="data/downloads")
    p.add_argument("--work-dir", default=None,
                   help="where the staged files live (default data/<key>)")
    p.add_argument("--labels-file", default=None,
                   help="CSV/JSON of filename -> label for unlabelled frame releases")
    p.add_argument("--extract", action="store_true",
                   help="unpack the download into <work-dir> before indexing")
    p.add_argument("--force", action="store_true", help="re-download / re-index")
    p.add_argument("--max", type=int, default=None, help="cap the number of records")
    p.add_argument("--list", action="store_true", help="show the registry and exit")
    return p


def run(args: argparse.Namespace) -> int:
    if args.list:
        for key, info in sorted(SOURCES.items()):
            flag = "" if info.commercial_use_ok else "  [NON-COMMERCIAL]"
            print(f"{key:16s} {info.license:16s} {info.source_name}{flag}")
        return 0

    if not args.dataset:
        print("[prepare] ERROR --dataset is required (or use --list)")
        return 2

    info = SOURCES[args.dataset]

    # The dataset registry in src/train.py is the single source of truth for
    # where a manifest must live, otherwise `--dataset nih_full` writes
    # data/nih_full/manifest.json while the trainer looks for data/nih/. Derive
    # the path from the registry when we recognise the key.
    registry_path = _registry_manifest_path(args.dataset)
    if registry_path is not None:
        work_dir = registry_path.parent
        manifest_path = registry_path
    else:
        work_dir = Path(args.work_dir or REPO_ROOT / "data" / args.dataset)
        manifest_path = work_dir / "manifest.json"

    if manifest_path.exists() and not args.force:
        payload = json.loads(manifest_path.read_text())
        _rime(f"[prepare] {manifest_path} already exists "
              f"({payload.get('num_images')} images) - use --force to rebuild")
        return 0

    # ---- 1. obtain the files -------------------------------------------- #
    source_dir: Path
    if args.local_dir:
        source_dir = Path(args.local_dir).expanduser()
        if not source_dir.exists():
            _rime(f"[prepare] ERROR --local-dir not found: {source_dir}")
            return 2
    elif args.url:
        archive = download(args.url, Path(args.download_dir) /
                           Path(urlparse(args.url).path).name)
        if args.extract or _looks_like_archive(archive):
            extract_archive(archive, work_dir)
        source_dir = work_dir
    else:
        _rime("[prepare] No source given. Options:\n"
              f"  --local-dir <existing folder>   ({info.download_hint})\n"
              f"  --url <archive to fetch>\n"
              f"For {args.dataset}: {info.download_hint}")
        return 2

    # ---- 2. index -------------------------------------------------------- #
    label_map = load_label_map(Path(args.labels_file)) if args.labels_file else None
    records = collect_records(source_dir, info.domain, info.source_name,
                              info.license, label_map=label_map)
    records, duplicates = _dedupe_records(records)
    if duplicates:
        _rime(f"[prepare] WARNING dropped {duplicates:,} records whose filename was "
              f"already indexed - the source holds two copies of the tree. This "
              f"usually means an archive was unzipped twice, leaving a nested "
              f"cell_images/ (or similar). Check with:\n"
              f"    find {source_dir} -maxdepth 2 -type d\n"
              f"and point --local-dir at the single copy you want to index.")
    if not records:
        _rime("[prepare] ERROR no labelled images found. Expected either\n"
              "  <dir>/{Parasitized,Uninfected}/**  (label from folder name) or\n"
              "  --labels-file <csv|json>  (label from a lookup table).")
        return 3

    if args.max and args.max > 0:
        # Keep both classes and spread over slides.
        by_label: dict[int, list[dict[str, Any]]] = {}
        for r in records:
            by_label.setdefault(int(r["label"]), []).append(r)
        picked: list[dict[str, Any]] = []
        per = max(1, args.max // max(1, len(by_label)))
        for _, items in sorted(by_label.items()):
            picked.extend(items[:per])
        records = picked

    write_manifest(records, info, manifest_path)
    if not info.commercial_use_ok:
        _rime(f"[prepare] NOTE: {info.key} is licensed {info.license}. "
              "It must not be mixed into a commercially redistributed model.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
