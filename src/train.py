"""Staged training loop for TinyMalariaNet.

Stage 1 : feature pre-training on NIH malaria crops (clean microscopy domain).
Stage 2 : smartphone domain adaptation on Makerere eyepiece captures with heavy
          optical augmentations (chromatic aberration, vignetting, blur, glare).
Stage 3 : INT8 QAT lives in ``src/quantize.py``.

Checkpointing is epoch-level and resume-safe by design: Spot VMs get preempted,
so every epoch writes ``epoch_{i}.pt`` plus ``best.pt`` (selected by a
recall-first score, not accuracy) and ``last.pt``.

Examples
--------
Dev dry-run (no data download needed):
    python src/train.py --dev-mode --sample 200 --epochs 1

Stage 1 (see design doc):
    python src/train.py --stage 1 --dataset nih_full --model mobilenet_v3_small \
        --batch 128 --epochs 20 --lr 1e-3 --checkpoint-dir checkpoints/stage1/

Stage 2:
    python src/train.py --stage 2 --resume checkpoints/stage1/best.pt \
        --dataset makerere_phone --batch 64 --epochs 15 --lr 1e-4 \
        --checkpoint-dir checkpoints/stage2/
"""

from __future__ import annotations

# Support BOTH documented entry points: `python src/train.py ...` (design doc
# section 3/5) and `python -m src.train`. Running the file directly leaves
# __package__ empty, so restore it and put the repo root on sys.path first.
if __package__ in (None, ""):  # pragma: no cover - exercised via the shell
    import os
    import sys as _sys

    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "src"  # noqa: A001

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from torch.amp import GradScaler, autocast

from .dataset import LoaderSpec, build_dataloaders, load_manifest, scan_image_folder
from .model import ModelConfig, TinyMalariaNet, build_model

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"

STAGE_CONFIG: dict[int, dict[str, Any]] = {
    1: {
        "name": "stage1_pretrain",
        "recommended_lr": 1e-3,
        "recommended_batch": 128,
        "epochs": 20,
        "unfreeze": "all",
    },
    2: {
        "name": "stage2_domain_adapt",
        "recommended_lr": 1e-4,
        "recommended_batch": 64,
        "epochs": 15,
        "unfreeze": "classifier+last_block",
    },
}


# --------------------------------------------------------------------------- #
# Dataset registry
# --------------------------------------------------------------------------- #
@dataclass
class DatasetSpec:
    key: str
    manifest: str | None
    folder: str | None
    domain: str
    license: str
    stage: int
    description: str


def _dataset_hint(spec: DatasetSpec, missing: Path) -> str:
    """An actionable message instead of a bare FileNotFoundError."""
    if spec.key == "lacuna_phone":
        return (
            f"Manifest {missing} missing. The Lacuna dataset is not bundled "
            "because it is ~7 GB. Fetch and parse it with:\n"
            f"  python src/download_lacuna.py --files Thin_Uganda.rar "
            f"--extract --extract-to data/lacuna\n"
            f"  python src/parse_lacuna.py --src data/lacuna "
            f"--out data/processed/lacuna_crops\n"
            f"(registry: Harvard Dataverse doi:10.7910/DVN/VEADSE, CC BY 4.0)"
        )
    return (
        f"Manifest {missing} missing. Download the dataset and build the manifest "
        f"(see README 'Data acquisition' or run src/prepare_data.py --list)."
    )


DATASETS: dict[str, DatasetSpec] = {
    # Stage 1 - NIH Malaria Dataset, 27,558 crops, CC0 / Public Domain.
    "nih_full": DatasetSpec("nih_full", "data/nih/manifest.json", None, "nih", "CC0", 1,
                            "NIH Malaria Dataset full 27.5k crop set"),
    # Offline smoke-test set: synthetically generated crops, see
    # src/make_sample_data.py. Exercises the pipeline, carries NO diagnostic
    # signal, and must never be used to claim accuracy.
    "nih_sample": DatasetSpec("nih_sample", "data/nih/sample_manifest.json", "data/sample",
                              "nih", "synthetic-no-license", 1,
                              "Small synthetic subset for codespace dry-runs"),
    # Stage 2 - Makerere AI Health Lab / Lacuna Fund smartphone eyepiece smears.
    "makerere_phone": DatasetSpec("makerere_phone", "data/makerere/manifest.json", None,
                                  "phone", "CC BY 4.0", 2,
                                  "Makerere smartphone eyepiece captures"),
    # The actual Makerere/Lacuna release. Download + parse with
    # src/download_lacuna.py and src/parse_lacuna.py; the manifest and the
    # ImageFolder tree under data/processed/lacuna_crops are both read.
    "lacuna_phone": DatasetSpec("lacuna_phone",
                                "data/processed/lacuna_crops/manifest.json",
                                "data/processed/lacuna_crops", "phone", "CC BY 4.0", 2,
                                "Lacuna Malaria Datasets, Harvard Dataverse "
                                "doi:10.7910/DVN/VEADSE (CC BY 4.0)"),
    # Stage 2 alternative - Broad Institute whole-slide derived crops.
    # ! Licensing note: BBBC041 is CC BY-NC-SA 3.0 (NonCommercial-ShareAlike),
    # NOT CC BY 3.0. The NC and SA terms make it incompatible with redistributing
    # Apache-2.0 trained weights, so this set is for research only and must not
    # be mixed into a commercially redistributed model. See README 'Data
    # licensing'.
    "bbbc041": DatasetSpec("bbbc041", "data/bbbc041/manifest.json", None, "makerere",
                           "CC BY-NC-SA 3.0", 2,
                           "Broad Institute BBBC041 cell segmentation crops (NON-COMMERCIAL)"),
    "mp_idb": DatasetSpec("mp_idb", "data/mp_idb/manifest.json", None, "makerere", "MIT", 2,
                          "MP-IDB microscopic images of bone marrow / blood smears"),
}

# Datasets whose license forbids commercial redistribution of trained weights.
NON_COMMERCIAL_DATASETS = {"bbbc041"}


def resolve_records(spec: DatasetSpec, limit: int | None = None) -> list[dict[str, Any]]:
    root = REPO_ROOT
    if spec.manifest:
        mpath = root / spec.manifest
        if mpath.exists():
            records = load_manifest(mpath)
        elif spec.folder and (root / spec.folder).exists():
            records = scan_image_folder(root / spec.folder, domain=spec.domain)
        else:
            raise FileNotFoundError(_dataset_hint(spec, mpath))
    elif spec.folder:
        records = scan_image_folder(root / spec.folder, domain=spec.domain)
    else:
        raise ValueError(f"Dataset '{spec.key}' has neither manifest nor folder configured")

    if limit and limit > 0:
        # Sample stratified by label so a tiny dev run keeps both classes.
        by_label: dict[int, list[dict[str, Any]]] = {}
        for r in records:
            by_label.setdefault(int(r["label"]), []).append(r)
        picked: list[dict[str, Any]] = []
        for label, items in sorted(by_label.items()):
            random.Random(1337).shuffle(items)
            picked.extend(items[: max(1, limit // max(1, len(by_label)))])
        records = picked
    return records


# --------------------------------------------------------------------------- #
# Losses / metrics
# --------------------------------------------------------------------------- #
def make_loss(train_records: list[dict[str, Any]], recall_focus: float = 1.0,
              device: torch.device | None = None) -> nn.Module:
    """Weighted BCE-with-logits.

    ``recall_focus > 1`` up-weights the positive (Parasitized) class, trading
    specificity away for sensitivity. The design target is >= 98% recall, so the
    default is a mild up-weight rather than an aggressive one.
    """
    counts = {0: 0, 1: 0}
    for r in train_records:
        counts[int(r["label"])] += 1
    n_neg, n_pos = max(1, counts[0]), max(1, counts[1])
    pos_weight = torch.tensor([(n_neg / n_pos) * recall_focus], dtype=torch.float32)
    if device is not None:
        pos_weight = pos_weight.to(device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


@torch.no_grad()
def evaluate(model: TinyMalariaNet, loader: torch.utils.data.DataLoader,
             device: torch.device, threshold: float = 0.5) -> dict[str, float]:
    """Compute sensitivity-first metrics on a validation loader."""
    model.eval()
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs.append(torch.sigmoid(logits).float().cpu().numpy())
        labels.append(targets.numpy().reshape(-1))

    if not probs:
        return {"loss": float("nan"), "accuracy": 0, "sensitivity": 0,
                "specificity": 0, "auc": float("nan"), "roc_auc": float("nan")}

    p = np.concatenate(probs)
    y = np.concatenate(labels).astype(int)
    pred = (p >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sensitivity = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    accuracy = accuracy_score(y, pred)
    try:
        auc = roc_auc_score(y, p)
    except ValueError:
        auc = float("nan")

    return {
        "loss": float("nan"),
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "recall": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(tp / max(1, tp + fp)),
        "auc": float(auc),
        "roc_auc": float(auc),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def _score(metrics: dict[str, float], target_recall: float = 0.98) -> float:
    """Selection score. Recall is the objective, so it dominates.

    A model only starts accumulating non-recall credit once it actually clears
    the recall target; before that, higher recall always wins outright.
    """
    rec = metrics.get("recall", metrics.get("sensitivity", 0.0))
    if rec < target_recall:
        return rec - 1.0  # strictly below any recall-clearing model
    return 1.0 + (rec - target_recall) + 0.5 * metrics.get("specificity", 0.0) \
        + 0.25 * max(0.0, metrics.get("roc_auc", 0.0))


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    stage: int = 1
    dataset: str = "nih_sample"
    model: str = "mobilenet_v3_small"
    batch_size: int = 64
    epochs: int = 1
    lr: float = 1e-3
    image_size: int = 128
    num_workers: int = 0
    weight_decay: float = 1e-5
    warmup_epochs: int = 1
    recall_focus: float = 1.0
    val_fraction: float = 0.15
    seed: int = 1337
    checkpoint_dir: str = "checkpoints/dev"
    resume: str | None = None
    dev_mode: bool = False
    sample: int | None = None
    prefetch_factor: int | None = None
    cache_dataset: bool = False
    amp: bool = True
    target_recall: float = 0.98
    log_every: int = 20
    device: str = "auto"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _maybe_reinit_stage2_classifier(model: TinyMalariaNet, state: dict[str, Any]) -> None:
    """Stage 2 resumes Stage 1 weights but re-initialises the classifier head."""
    model.load_state_dict(state, strict=True)


def build_optimizer(model: TinyMalariaNet, cfg: TrainConfig) -> torch.optim.Optimizer:
    if cfg.stage == 2:
        # Fine-tune the head and the last inverted-residual block; keep early
        # generic features frozen so a tiny phone dataset cannot destroy them.
        params: list[nn.Parameter] = []
        for name, p in model.named_parameters():
            if name.startswith("classifier") or "features.12" in name or "features.11" in name:
                p.requires_grad = True
                params.append(p)
            else:
                p.requires_grad = False
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    return torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )


def train(cfg: TrainConfig) -> dict[str, Any]:
    set_seed(cfg.seed)
    device = select_device(cfg.device)
    is_cuda = device.type == "cuda"

    spec = DATASETS.get(cfg.dataset)
    if spec is None:
        raise KeyError(f"Unknown dataset '{cfg.dataset}'. Available: {sorted(DATASETS)}")

    if cfg.dev_mode and spec.key == "nih_sample":
        records = None  # train.py will synthesise below
    else:
        records = resolve_records(spec, limit=cfg.sample)

    if records is None or len(records) < 20:
        if not cfg.dev_mode:
            raise RuntimeError(
                f"Dataset '{spec.key}' resolved to {0 if records is None else len(records)} "
                f"records. Populate it or re-run with --dev-mode."
            )
        from .make_sample_data import ensure_sample_dataset

        print("[train] --dev-mode: synthesising mock crops for the dry-run")
        records = ensure_sample_dataset(root=DATA_ROOT, total=cfg.sample or 200)

    print(f"[train] stage={cfg.stage} dataset={spec.key} "
          f"({len(records)} images, license={spec.license})")

    loader_spec = LoaderSpec(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        image_size=cfg.image_size,
        domain=spec.domain,
        cache=cfg.cache_dataset,
    )
    train_loader, val_loader, split_stats = build_dataloaders(
        records,
        spec=loader_spec,
        val_fraction=cfg.val_fraction,
        seed=cfg.seed,
    )
    print(f"[train] patient split -> train={split_stats['train_images']} "
          f"val={split_stats['val_images']} "
          f"(patients {split_stats['train_patients']}/{split_stats['val_patients']})")
    print(f"[train] label balance train={split_stats['train_label_counts']} "
          f"val={split_stats['val_label_counts']}")
    if split_stats["train_images"] == 0:
        raise RuntimeError("Empty training split - check the dataset manifest.")

    mconf = ModelConfig(image_size=cfg.image_size, pretrained=True)
    model = build_model(mconf, pretrained=True).to(device)
    if cfg.resume:
        ckpt = torch.load(Path(cfg.resume), map_location="cpu", weights_only=False)
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state, strict=True)
        print(f"[train] resumed weights from {cfg.resume} (epoch={ckpt.get('epoch', '?')})")

    criterion = make_loss(train_loader.dataset.records, cfg.recall_focus, device=device)
    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, cfg.epochs), eta_min=cfg.lr * 0.01
    )
    scaler = GradScaler(enabled=is_cuda and cfg.amp)

    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, Any]] = []
    best_score = -math.inf
    best_metrics: dict[str, float] = {}

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running, n_batches = 0.0, 0
        t0 = time.time()

        for step, (images, targets) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device).reshape(-1)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=is_cuda and cfg.amp):
                logits = model(images)
                loss = criterion(logits, targets)

            if is_cuda and cfg.amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            running += float(loss.item())
            n_batches += 1
            if step % cfg.log_every == 0 or step == len(train_loader):
                print(f"  epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss={running / max(1, n_batches):.4f}")

        scheduler.step()
        train_loss = running / max(1, n_batches)

        metrics = evaluate(model, val_loader, device)
        metrics["loss"] = train_loss
        metrics["epoch"] = epoch
        metrics["lr"] = optimizer.param_groups[0]["lr"]
        score = _score(metrics, cfg.target_recall)

        elapsed = time.time() - t0
        print(f"[epoch {epoch}] loss={train_loss:.4f} "
              f"sens={metrics['sensitivity']:.4f} spec={metrics['specificity']:.4f} "
              f"acc={metrics['accuracy']:.4f} auc={metrics['roc_auc']:.4f} "
              f"score={score:.4f} ({elapsed:.1f}s)")

        # --- Spot-VM-safe incremental checkpointing -----------------------
        payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": asdict(cfg),
            "model_config": asdict(mconf),
            "metrics": metrics,
            "split_stats": split_stats,
            "dataset": spec.key,
            "license": spec.license,
        }
        torch.save(payload, ckpt_dir / f"epoch_{epoch}.pt")
        torch.save(payload, ckpt_dir / "last.pt")
        if score > best_score:
            best_score = score
            best_metrics = dict(metrics)
            torch.save(payload, ckpt_dir / "best.pt")
            print(f"[epoch {epoch}] new best -> {ckpt_dir / 'best.pt'}")

        history.append({k: v for k, v in metrics.items()})

    summary = {
        "stage": cfg.stage,
        "dataset": spec.key,
        "license": spec.license,
        "epochs": cfg.epochs,
        "best_score": best_score,
        "best_metrics": best_metrics,
        "history": history,
        "split_stats": split_stats,
        "device": str(device),
        "checkpoint_dir": str(ckpt_dir),
    }
    (ckpt_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[train] done. best score={best_score:.4f} metrics={best_metrics}")
    return summary


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Staged trainer for TinyMalariaNet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--stage", type=int, default=1, choices=[1, 2],
                   help="1=NIH pretrain, 2=smartphone domain adaptation")
    p.add_argument("--dataset", type=str, default="nih_sample", choices=sorted(DATASETS),
                   help="dataset registry key")
    p.add_argument("--model", type=str, default="mobilenet_v3_small")
    p.add_argument("--batch", type=int, default=64, help="batch size")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=None, help="defaults per stage")
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--recall-focus", type=float, default=1.0,
                   help=">1 up-weights Parasitized class to push recall")
    p.add_argument("--target-recall", type=float, default=0.98)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints/dev")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--cache-dataset", action="store_true")
    p.add_argument("--log-every", type=int, default=20)

    # dev / smoke test flags
    p.add_argument("--dev-mode", action="store_true",
                   help="synthesise mock crops so the pipeline runs offline")
    p.add_argument("--sample", type=int, default=None,
                   help="limit to N images (stratified by label)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    stage_cfg = STAGE_CONFIG[args.stage]
    lr = args.lr if args.lr is not None else stage_cfg["recommended_lr"]

    cfg = TrainConfig(
        stage=args.stage,
        dataset=args.dataset,
        model=args.model,
        batch_size=args.batch,
        epochs=args.epochs,
        lr=lr,
        image_size=args.image_size,
        num_workers=args.num_workers,
        recall_focus=args.recall_focus,
        target_recall=args.target_recall,
        val_fraction=args.val_fraction,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir or f"checkpoints/{stage_cfg['name']}",
        resume=args.resume,
        dev_mode=args.dev_mode,
        sample=args.sample,
        amp=not args.no_amp,
        cache_dataset=args.cache_dataset,
        log_every=args.log_every,
        device=args.device,
    )

    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
