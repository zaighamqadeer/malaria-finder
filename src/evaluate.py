"""Sensitivity-first evaluation for TinyMalariaNet.

The primary metric is RECALL (sensitivity) at a threshold chosen on the
validation ROC, not accuracy and not the default 0.5 cut. The design target is
>= 98% sensitivity, so this module reports:

  * ROC AUC and the optimal operating threshold using Youden's J,
  * the recall-first threshold: the highest threshold that still clears the
    sensitivity target (maximises specificity without sacrificing recall),
  * the full confusion matrix at that operating point.

For an edge triage tool a false negative (missing a parasitized cell) is far
more harmful than a false positive, which is why specificity is only
secondarily optimised.

Usage
-----
    python src/evaluate.py \
        --checkpoint checkpoints/stage2/best.pt \
        --dataset makerere_phone \
        --target-recall 0.98
"""

from __future__ import annotations

# Support BOTH documented entry points: `python src/evaluate.py ...` and
# `python -m src.evaluate`.
if __package__ in (None, ""):  # pragma: no cover - exercised via the shell
    import os
    import sys as _sys

    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "src"  # noqa: A001

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from .dataset import LoaderSpec, build_dataloaders
from .model import ModelConfig, build_model
from .train import DATASETS, resolve_records, select_device, set_seed

REPO_ROOT = Path(__file__).resolve().parent.parent


def confusion_at_threshold(y: np.ndarray, p: np.ndarray, t: float) -> dict[str, int]:
    """Confusion counts at threshold ``t``. Accepts lists or arrays."""
    y_arr = np.asarray(y).reshape(-1).astype(int)
    p_arr = np.asarray(p, dtype=float).reshape(-1)
    pred = (p_arr >= float(t)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_arr, pred, labels=[0, 1]).ravel()
    return {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}


def sensitivity_first_threshold(
    y: np.ndarray, p: np.ndarray, target_recall: float = 0.98
) -> tuple[float, dict[str, Any]]:
    """Return (threshold, info) for the highest threshold meeting target recall."""
    thresholds = np.unique(np.round(p, 5))
    candidates = np.concatenate([[0.0], thresholds, [1.00001]])

    best_t, best_info = 0.5, {}
    for t in candidates:
        cm = confusion_at_threshold(y, p, float(t))
        sens = cm["tp"] / max(1, cm["tp"] + cm["fn"])
        spec = cm["tn"] / max(1, cm["tn"] + cm["fp"])
        if sens >= target_recall:
            acc = (cm["tp"] + cm["tn"]) / max(1, len(y))
            prec = cm["tp"] / max(1, cm["tp"] + cm["fp"])
            info = {
                "threshold": float(t),
                "sensitivity": float(sens),
                "specificity": float(spec),
                "precision": float(prec),
                "accuracy": float(acc),
                "confusion": cm,
                "f1": float(2 * prec * sens / max(1e-9, prec + sens)),
            }
            if not best_info or spec > best_info["specificity"]:
                best_t, best_info = float(t), info

    if not best_info:
        pred = (p >= 0.5).astype(int)
        best_info = {
            "threshold": 0.5,
            "sensitivity": float(np.sum((pred == 1) & (y == 1)) / max(1, np.sum(y == 1))),
            "note": f"target recall {target_recall} unreachable; fell back to 0.5",
        }
    return best_t, best_info


@torch.no_grad()
def collect_predictions(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, labels = [], []
    for images, targets in loader:
        logits = model(images.to(device))
        probs.append(torch.sigmoid(logits).float().cpu().numpy())
        labels.append(targets.numpy().reshape(-1))
    return np.concatenate(labels).astype(int), np.concatenate(probs)


def evaluate_checkpoint(
    checkpoint: str | Path,
    dataset_key: str,
    target_recall: float = 0.98,
    image_size: int = 128,
    batch: int = 32,
    device: str = "auto",
    val_fraction: float = 0.2,
    seed: int = 1337,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    set_seed(seed)
    dev = select_device(device)

    ckpt_path = Path(checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mconf = ModelConfig(**ckpt.get("model_config", asdict(ModelConfig())))
    model = build_model(mconf, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(dev)

    spec = DATASETS.get(dataset_key)
    if spec is None:
        raise KeyError(f"Unknown dataset '{dataset_key}'")

    records = resolve_records(spec)
    loader_spec = LoaderSpec(batch_size=batch, num_workers=0,
                             image_size=image_size or mconf.image_size,
                             domain=spec.domain)
    _, val_loader, split_stats = build_dataloaders(
        records, spec=loader_spec, val_fraction=val_fraction, seed=seed
    )
    print(f"[eval] dataset={dataset_key} val_images={split_stats['val_images']} "
          f"val_patients={split_stats['val_patients']}")

    y, p = collect_predictions(model, val_loader, dev)

    fpr, tpr, _ = roc_curve(y, p)
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    ap = float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    j_scores = tpr - fpr
    youden_t = float(_[np.argmax(j_scores)]) if len(_) else 0.5

    threshold, operating = sensitivity_first_threshold(y, p, target_recall)

    precision, recall, pr_thresholds = precision_recall_curve(y, p)

    results: dict[str, Any] = {
        "checkpoint": str(ckpt_path),
        "dataset": dataset_key,
        "dataset_license": spec.license,
        "n_samples": int(len(y)),
        "class_balance": {"neg": int((y == 0).sum()), "pos": int((y == 1).sum())},
        "roc_auc": auc,
        "average_precision": ap,
        "youden_threshold": youden_t,
        "youden_j": float(np.max(j_scores)) if len(j_scores) else float("nan"),
        "target_recall": target_recall,
        "operating_threshold": threshold,
        "operating_metrics": operating,
        "metrics_at_0.5": None,
        "split_stats": split_stats,
        "recall_target_met": bool(operating.get("sensitivity", 0.0) >= target_recall),
        "roc_curve": {
            "fpr": [round(float(v), 6) for v in fpr[:: max(1, len(fpr) // 200)]],
            "tpr": [round(float(v), 6) for v in tpr[:: max(1, len(tpr) // 200)]],
        },
        "pr_curve": {
            "precision": [round(float(v), 6) for v in precision[:: max(1, len(precision) // 200)]],
            "recall": [round(float(v), 6) for v in recall[:: max(1, len(recall) // 200)]],
        },
    }

    cm05 = confusion_at_threshold(y, p, 0.5)
    results["metrics_at_0.5"] = {
        "sensitivity": cm05["tp"] / max(1, cm05["tp"] + cm05["fn"]),
        "specificity": cm05["tn"] / max(1, cm05["tn"] + cm05["fp"]),
        "accuracy": float(accuracy_score(y, (p >= 0.5).astype(int))),
        "confusion": cm05,
    }

    print(f"[eval] ROC AUC              = {auc:.4f}")
    print(f"[eval] sensitivity-first t* = {threshold:.4f}")
    for k in ("sensitivity", "specificity", "precision", "accuracy", "f1"):
        if k in operating:
            print(f"[eval] {k:<21} = {operating[k]:.4f}")
    print(f"[eval] confusion @ t*       = {operating.get('confusion')}")
    print(f"[eval] recall >= {target_recall:.2f} -> "
          f"{'PASS' if results['recall_target_met'] else 'FAIL'}")
    if results["recall_target_met"] is False:
        print("[eval] NOTE: model does not meet the recall target yet. Re-run "
              "training with --recall-focus > 1 or --target-recall lowered.")

    if output_json:
        Path(output_json).write_text(json.dumps(results, indent=2))
        print(f"[eval] wrote {output_json}")

    return results


def _iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _resolve_gt_path(image_path: str, image_dir: Path) -> Path:
    """Resolve a ground-truth key that may be repo-relative or dir-relative."""
    cand = Path(image_path)
    if cand.exists():
        return cand
    for base in (image_dir, image_dir.parent, REPO_ROOT):
        alt = base / cand
        if alt.exists():
            return alt
    raise FileNotFoundError(f"Ground-truth image not found: {image_path}")


def evaluate_segmentation(
    image_dir: str | Path = REPO_ROOT / "data" / "phone_test",
    iou_threshold: float = 0.3,
    target_recall: float = 0.80,
    cfg: Any | None = None,
) -> dict[str, Any]:
    """Score the OpenCV segmenter against ground-truth boxes.

    A ground-truth cell counts as detected when at least one predicted box has
    IoU >= ``iou_threshold``. The design criterion is >= 80% RBC detection.
    Requires ``ground_truth.json`` produced by ``src/make_sample_data.py --mode fov``.
    """
    from src.segment import SegmentConfig, segment_cells

    image_dir = Path(image_dir)
    gt_path = image_dir / "ground_truth.json"
    if not gt_path.exists():
        raise FileNotFoundError(f"No ground truth at {gt_path}")

    gt = json.loads(gt_path.read_text())
    seg_cfg = cfg or SegmentConfig()

    per_image: dict[str, Any] = {}
    total_gt = total_tp = total_pred = 0
    for image_path, meta in gt["images"].items():
        resolved = _resolve_gt_path(image_path, image_dir)
        cands = segment_cells(resolved, cfg=seg_cfg)
        pred_boxes = [(c.x, c.y, c.w, c.h) for c in cands]
        gt_boxes = [(b["x"], b["y"], b["w"], b["h"]) for b in meta["boxes"]]

        matched = 0
        for gb in gt_boxes:
            if any(_iou(gb, pb) >= iou_threshold for pb in pred_boxes):
                matched += 1

        per_image[resolved.name] = {
            "gt_cells": len(gt_boxes),
            "detected": matched,
            "predicted": len(pred_boxes),
            "recall": matched / len(gt_boxes) if gt_boxes else 0.0,
            "precision": matched / len(pred_boxes) if pred_boxes else 0.0,
        }
        total_gt += len(gt_boxes)
        total_tp += matched
        total_pred += len(pred_boxes)

    recall = total_tp / total_gt if total_gt else 0.0
    precision = total_tp / total_pred if total_pred else 0.0
    results = {
        "mode": "segmentation",
        "ground_truth": str(gt_path),
        "iou_threshold": iou_threshold,
        "target_recall": target_recall,
        "images": len(per_image),
        "gt_cells": total_gt,
        "detected": total_tp,
        "predicted": total_pred,
        "recall": recall,
        "precision": precision,
        "f1": (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0,
        "target_met": recall >= target_recall,
        "per_image": per_image,
    }

    print(f"[seg-eval] images={results['images']} gt_cells={total_gt} "
          f"detected={total_tp} predicted={total_pred}")
    print(f"[seg-eval] detection recall@IoU{iou_threshold} = {recall:.3f} "
          f"(target >= {target_recall:.2f}) -> "
          f"{'PASS' if results['target_met'] else 'FAIL'}")
    print(f"[seg-eval] precision = {precision:.3f}  f1 = {results['f1']:.3f}")
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sensitivity-first evaluation for TinyMalariaNet")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--dataset", type=str, default="makerere_phone", choices=sorted(DATASETS))
    p.add_argument("--target-recall", type=float, default=0.98)
    p.add_argument("--recall", type=float, default=None,
                   help="alias for --target-recall")
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--output-json", type=str, default=None)
    # segmentation mode
    p.add_argument("--mode", type=str, default="classifier",
                   choices=["classifier", "segmentation"])
    p.add_argument("--image-dir", type=str,
                   default=str(REPO_ROOT / "data" / "phone_test"))
    p.add_argument("--iou", type=float, default=0.3)
    p.add_argument("--seg-target-recall", type=float, default=0.80,
                   help="RBC detection target for segmentation mode "
                        "(design doc: more than 80 percent)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    target = args.recall if args.recall is not None else args.target_recall

    if args.mode == "segmentation":
        results = evaluate_segmentation(
            args.image_dir, iou_threshold=args.iou,
            target_recall=args.seg_target_recall,
        )
        if args.output_json:
            Path(args.output_json).write_text(json.dumps(results, indent=2))
            print(f"[seg-eval] wrote {args.output_json}")
        return 0 if results["target_met"] else 1

    if not args.checkpoint:
        print("[eval] --checkpoint is required in classifier mode")
        return 2

    results = evaluate_checkpoint(
        args.checkpoint, args.dataset, target_recall=target,
        image_size=args.image_size, batch=args.batch, device=args.device,
        val_fraction=args.val_fraction, seed=args.seed, output_json=args.output_json,
    )
    return 0 if results["recall_target_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
