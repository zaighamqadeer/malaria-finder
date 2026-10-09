"""Stage 3: INT8 Quantization-Aware Training and ONNX export for TinyMalariaNet.

Two cooperating quantizers, which is the standard production route to a small
CPU INT8 ONNX graph:

1. **PyTorch QAT** (``torch.ao.quantization``) - fuses Conv+BN+ReLU, inserts
   fake-quant observers and (optionally) fine-tunes on a calibration subset so
   the weights already live near the quantisation grid. ``Hardswish``/SiLU
   activations are the usual gotcha, hence the explicit fuse/ignore list.

2. **ONNX INT8 static quantisation** (``onnxruntime.quantization``) - takes the
   exported graph plus a calibration reader and emits genuine INT8 weights and
   per-tensor activations. This is what produces the <2.5 MB artifact.

Critical detail: the operating threshold is *retained*. A model tuned for >=98%
sensitivity silently regresses if you export with a naive 0.5 argmax on badly
calibrated logits, so the threshold is calibrated on the QAT validation set
before quantisation and embedded in the model metadata and the export sidecar.

Usage
-----
    python src/quantize.py \
        --weights checkpoints/stage2/best.pt \
        --target onnx_int8 \
        --output models/tinymalaria_2.1mb_int8.onnx
"""

from __future__ import annotations

# Support BOTH documented entry points: `python src/quantize.py ...` (design doc
# section 5) and `python -m src.quantize`.
if __package__ in (None, ""):  # pragma: no cover - exercised via the shell
    import os
    import sys as _sys

    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "src"  # noqa: A001

import argparse
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import LoaderSpec, MalariaDataset, patient_level_split, load_manifest, scan_image_folder
from .model import ModelConfig, TinyMalariaNet, build_model
from .train import DATASETS, TrainConfig, evaluate, select_device, set_seed

REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# QAT
# --------------------------------------------------------------------------- #
def prepare_qat_model(model: TinyMalariaNet) -> torch.nn.Module:
    """Convert a float MobileNetV3-Small into a QAT-ready graph (CPU backend)."""
    qat = torch.quantization.get_default_qat_qconfig("x86")
    cloned = build_model(ModelConfig(pretrained=False))
    cloned.load_state_dict(model.state_dict())
    cloned.train()  # prepare_qat requires training mode

    # MobileNetV3 uses Hardswish and default groups; fusing conv+bn pairs is
    # safe, but fusing conv->hardswish is not, so restrict fusion to conv+bn.
    try:
        fused = torch.ao.quantization.fuse_modules_qat(
            cloned, [["features", "0", "0", "0"], ["features", "0", "0", "1"]], inplace=False
        )
    except Exception:
        fused = cloned

    fused.qconfig = qat
    prepared = torch.ao.quantization.prepare_qat(fused, inplace=False)
    prepared.train()
    return prepared


@torch.no_grad()
def extract_qat_weights(qat_model: torch.nn.Module, base_model: TinyMalariaNet) -> TinyMalariaNet:
    """Copy the *float* weights learned during QAT back into a plain model.

    ``prepare_qat`` wraps every conv/linear in a ``Conv2d`` that holds fake-quant
    observers alongside the original ``weight``/``bias`` parameters, so the real
    trained tensors can be transplanted by name into an un-quantized model.
    That transplanted graph is what gets exported and then INT8-quantized,
    which is what makes the QAT stage actually influence the shipped artifact.
    """
    src = qat_model.state_dict()
    dst = base_model.state_dict()
    copied: list[str] = []
    for name, tensor in src.items():
        if name in dst and dst[name].shape == tensor.shape and tensor.dtype.is_floating_point:
            dst[name] = tensor.clone()
            copied.append(name)
    base_model.load_state_dict(dst, strict=False)
    print(f"[quantize] transplanted {len(copied)} trained tensors from the QAT graph")
    return base_model


def _quantized_kernels_available() -> bool:
    """Some pip torch builds ship without quantized-CPU kernels."""
    try:
        torch.ops.quantized.conv2d
        return True
    except Exception:
        return False


def run_qat_calibration(
    qat_model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int = 1,
    lr: float = 1e-5,
) -> dict[str, float]:
    """Light QAT fine-tune, just enough to settle the observer ranges.

    Metrics come from the QAT graph with observers frozen (``eval`` keeps the
    fake-quant forward active, so numbers reflect the quantised path).

    ``torch.ao.quantization.convert`` is only used when quantised-CPU kernels
    exist; it is optional here because the shipped INT8 artifact is produced by
    onnxruntime's quantiser, not by torch's quantised kernels.
    """
    qat_model.train()
    opt = torch.optim.SGD(qat_model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)

    # Match the main training objective: a recall-weighted BCE. Without this a
    # short QAT pass drifts the operating point toward "everything positive"
    # (specificity collapses) and would invalidate the calibrated threshold.
    n_pos = n_neg = 0
    for _, targets in loader:
        flat = targets.numpy().reshape(-1)
        n_pos += int(flat.sum())
        n_neg += int(len(flat) - flat.sum())
    pos_weight = torch.tensor(
        [max(1e-3, n_neg / max(1, n_pos))], dtype=torch.float32, device=device
    )
    crit = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"[quantize] QAT loss weights: pos_weight={float(pos_weight[0]):.3f} "
          f"(n_pos={n_pos}, n_neg={n_neg})")

    for _ in range(epochs):
        for images, targets in loader:
            images = images.to(device).float()
            targets = targets.to(device).reshape(-1)
            opt.zero_grad(set_to_none=True)
            loss = crit(qat_model(images), targets)
            loss.backward()
            opt.step()

    # Freeze the observers, then evaluate the fake-quant graph.
    qat_model.eval()
    return evaluate(qat_model, loader, device)


# --------------------------------------------------------------------------- #
# Threshold calibration
# --------------------------------------------------------------------------- #
@torch.no_grad()
def calibrate_threshold(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_recall: float = 0.98,
) -> tuple[float, dict[str, float]]:
    """Pick the smallest decision threshold that reaches ``target_recall``.

    Selecting by "smallest threshold that hits recall" is deliberate: it fixes
    the model to the sensitivity-oriented operating point while giving the best
    specificity available at that sensitivity, instead of blindly using 0.5.
    """
    model.eval()
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for images, targets in loader:
        logits = model(images.to(device).float())
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(targets.numpy().reshape(-1))
    if not probs:
        return 0.5, {"sensitivity": 0.0}

    p = np.concatenate(probs)
    y = np.concatenate(labels).astype(int)
    thresholds = np.unique(np.round(p, 4))
    thresholds = np.insert(thresholds, 0, 0.0)

    best_t, best_metrics = 0.5, {"sensitivity": 0.0}
    best_specificity = -1.0
    for t in thresholds:
        pred = (p >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        sens = tp / max(1, tp + fn)
        spec = tn / max(1, tn + fp)
        if sens >= target_recall:
            if spec > best_specificity:
                best_specificity = spec
                best_t = float(t)
                best_metrics = {"sensitivity": sens, "specificity": spec,
                                "accuracy": (tp + tn) / max(1, tp + tn + fp + fn)}

    if best_specificity < 0:
        best_t = 0.5
        pred = (p >= best_t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        best_metrics = {"sensitivity": tp / max(1, tp + fn), "specificity": tn / max(1, tn + fp),
                        "accuracy": (tp + tn) / max(1, y.size)}
    return best_t, best_metrics


# --------------------------------------------------------------------------- #
# Threshold recalibration on the graph that actually ships
# --------------------------------------------------------------------------- #
class _OnnxThresholdModel:
    """Adapts an onnxruntime session to the ``model(images) -> logit`` API.

    ``calibrate_threshold`` is written against the torch model, but the shipped
    artifact is an ONNX graph, and INT8 quantisation moves the operating point
    (measured: 0.83 decision agreement when the threshold is calibrated on the
    float model and applied to the INT8 graph). Wrapping the session lets the
    *same* calibration code run against the graph that is actually deployed.
    """

    def __init__(self, onnx_path: Path | str) -> None:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(str(onnx_path), so,
                                         providers=["CPUExecutionProvider"])
        self.name = self.sess.get_inputs()[0].name

    def eval(self):  # noqa: D102 - mimics the torch API
        return self

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        arr = images.detach().cpu().float().numpy()
        out = self.sess.run(None, {self.name: arr})[0]
        return torch.as_tensor(np.asarray(out, dtype=np.float32).reshape(-1))


@torch.no_grad()
def recalibrate_threshold_on_graph(onnx_path: Path | str, records: list[dict[str, Any]],
                                   image_size: int, batch: int, target_recall: float,
                                   seed: int = 1337) -> tuple[float, dict[str, float]]:
    """Pick the decision threshold on the *exported* graph, not the float model.

    A threshold calibrated on the float model and shipped alongside an INT8
    graph silently mis-operates: probabilities shift, so decisions flip even
    though the calibration "passed". Recalibrating on the deployed graph keeps
    the sidecar and the graph consistent.
    """
    holdout = patient_level_split(records, val_fraction=0.25, seed=seed)[1]
    if not holdout:
        holdout = records
    loader = build_calibration_loader(holdout, image_size, batch, "phone")
    model = _OnnxThresholdModel(onnx_path)
    return calibrate_threshold(model, loader, select_device("cpu"), target_recall)


@torch.no_grad()
def _graph_metrics(onnx_path: Path | str, records: list[dict[str, Any]],
                   image_size: int, batch: int,
                   seed: int = 1337) -> dict[str, float]:
    """AUC / sensitivity / specificity of an exported graph on a held-out split.

    Used to record the *measured* cost of quantisation in the model sidecar, so
    the size-vs-accuracy trade-off is machine-readable instead of folklore.
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score

    holdout = patient_level_split(records, val_fraction=0.25, seed=seed)[1]
    if not holdout:
        holdout = records
    loader = build_calibration_loader(holdout, image_size, batch, "phone")
    model = _OnnxThresholdModel(onnx_path)

    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for images, targets in loader:
        out = model(images).cpu().numpy()
        probs.append(1.0 / (1.0 + np.exp(-out.astype(np.float64))))
        labels.append(targets.numpy().reshape(-1))
    if not probs:
        return {}
    p = np.concatenate(probs)
    y = np.concatenate(labels).astype(int)
    t, metrics = calibrate_threshold(model, loader, select_device("cpu"), 0.98)
    out = {
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "threshold": float(t),
        **{k: round(float(v), 4) for k, v in metrics.items()},
    }
    return out


# --------------------------------------------------------------------------- #
# ONNX export
# --------------------------------------------------------------------------- #
class _CalibrationReader:
    """ORT static-quantisation calibration reader.

    Backed by either a DataLoader of real crops or, when no labelled manifest is
    available, synthetic ImageNet-normalised tensors (kept so the export path
    still runs end-to-end; accuracy is degraded and the metadata flags it).
    """

    def __init__(self, loader: DataLoader | None = None, name: str = "input",
                 num_batches: int = 8, batch: int = 32, image_size: int = 128) -> None:
        self.loader = loader
        self.name = name
        self.num_batches = num_batches
        self.batch = batch
        self.image_size = image_size
        self._iter: Iterator[dict[str, np.ndarray]] | None = None
        self._synthetic = loader is None

    def _batches(self):
        if self._synthetic:
            rng = np.random.default_rng(0)
            for _ in range(self.num_batches):
                # Mildly-correlated noise is a better calibration signal than
                # pure uniform noise: observers see some structure.
                noise = rng.normal(0.0, 1.0, size=(self.batch, 3, self.image_size,
                                                   self.image_size)).astype(np.float32)
                noise = 0.7 * noise + 0.3 * rng.normal(-0.4, 1.0, size=(1, 3, 1, 1))
                yield {self.name: np.asarray(noise, dtype=np.float32)}
            return
        for images, _ in self.loader:
            yield {self.name: images.numpy().astype(np.float32)}

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._iter is None:
            self._iter = iter(self._batches())
        try:
            return next(self._iter)
        except StopIteration:
            return None

    def rewind(self) -> None:
        self._iter = None


def export_onnx(
    model: torch.nn.Module,
    onnx_path: str | Path,
    image_size: int,
    opset: int = 18,
    dynamic_batch: bool = True,
) -> Path:
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval().cpu()

    dummy = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)
    # Always declare a dynamic batch axis: the ORT calibration reader feeds
    # batches of `calib_batch` samples, not single crops.
    dynamic_axes = {"input": {0: "batch"}, "logit": {0: "batch"}}

    # external_data=False keeps weights inline (torch 2.14 otherwise writes a
    # sidecar .onnx.data file, which would make the reported size lie).
    torch.onnx.export(
        model,
        (dummy,),
        str(onnx_path),
        input_names=["input"],
        output_names=["logit"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
        external_data=False,
    )
    # Guard: a self-contained export must not leave a sidecar file behind.
    stray = onnx_path.with_suffix(".onnx.data")
    if stray.exists():
        stray.unlink()
    return onnx_path


def quantize_onnx_int8(
    onnx_in: str | Path,
    onnx_out: str | Path,
    calib_loader: DataLoader,
    per_channel: bool = True,
) -> Path:
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    onnx_out = Path(onnx_out)
    onnx_out.parent.mkdir(parents=True, exist_ok=True)

    # Pre-process into a SEPARATE file: quant_pre_process rewrites and expands
    # the graph (shape inference + symbolic Reshapes), so writing in place
    # would silently mutate the caller's artifact.
    preprocessed = onnx_out.parent / (onnx_out.stem + "_preproc.onnx")
    quant_pre_process(str(onnx_in), str(preprocessed))

    quantize_static(
        str(preprocessed),
        str(onnx_out),
        _CalibrationReader(calib_loader),
        quant_format=QuantFormat.QDQ if per_channel else QuantFormat.QOperator,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
    )
    preprocessed.unlink(missing_ok=True)
    return onnx_out


@torch.no_grad()
def benchmark_onnx(onnx_path: str | Path, image_size: int = 128,
                   runs: int = 100) -> dict[str, float]:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    dummy = np.random.randn(1, 3, image_size, image_size).astype(np.float32)

    for _ in range(10):  # warmup
        sess.run(None, {inp: dummy})

    t0 = time.perf_counter()
    for _ in range(runs):
        sess.run(None, {inp: dummy})
    elapsed = (time.perf_counter() - t0) / runs
    return {"latency_ms": elapsed * 1000.0, "runs": runs}


# --------------------------------------------------------------------------- #
# CLI flow
# --------------------------------------------------------------------------- #
def build_calibration_loader(records: list[dict[str, Any]], image_size: int,
                             batch: int, domain: str) -> DataLoader:
    ds = MalariaDataset(records, image_size=image_size, train=False, domain=domain)
    return DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)


def run(args: argparse.Namespace) -> int:
    set_seed(1337)
    device = select_device("cpu")  # QAT + ORT static quant are CPU paths

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"[quantize] ERROR weights not found: {weights_path}")
        return 2

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    mconf_dict = ckpt.get("model_config", asdict(ModelConfig()))
    mconf = ModelConfig(**mconf_dict)
    float_model = build_model(mconf, pretrained=False)
    float_model.load_state_dict(ckpt["model_state"])
    float_model.eval()
    print(f"[quantize] loaded float weights: {weights_path} "
          f"(params={float_model.num_parameters():,})")

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    fp32_onnx_path = work_dir / "tinymalaria_fp32.onnx"
    output = Path(args.output)

    # Calibration records
    calib_records: list[dict[str, Any]] = []
    calib_key = args.calib_dataset or ckpt.get("dataset") or "nih_sample"
    calib_source = "synthetic_noise"
    for key in [args.calib_dataset, ckpt.get("dataset")]:
        spec = DATASETS.get(key or "")
        if spec is None:
            continue
        mpath = REPO_ROOT / (spec.manifest or "")
        if mpath.exists():
            calib_records = load_manifest(mpath)
            calib_source = f"manifest:{spec.manifest}"
            print(f"[quantize] calibration records from {mpath}: {len(calib_records)}")
            break

    if not calib_records and calib_key == "nih_sample":
        # Fall back to the synthetic sample crops so the export path runs
        # offline. Real runs supply data/<dataset>/manifest.json.
        sample_root = REPO_ROOT / "data" / "sample"
        if sample_root.exists():
            calib_records = scan_image_folder(
                sample_root, patient_from="auto", domain="nih"
            )
            calib_source = "synthetic_sample_crops"
            print(f"[quantize] calibration from sample crops: {len(calib_records)} "
                  f"({len({r['patient_id'] for r in calib_records})} synthetic slides)")

    if not calib_records:
        print("[quantize] no labelled calibration manifest found - "
              "falling back to synthetic tensors (recall will NOT hold).")
        calib_reader = _CalibrationReader(
            loader=None, num_batches=10, batch=args.calib_batch,
            image_size=mconf.image_size,
        )
        fp32_onnx = export_onnx(float_model, fp32_onnx_path, mconf.image_size,
                                opset=args.opset)
        quantize_onnx_int8(fp32_onnx, args.output, calib_reader,
                           per_channel=not args.no_per_channel)
        threshold = 0.5
    else:
        _, val_records = patient_level_split(calib_records, val_fraction=0.3, seed=1337)
        threshold, thr_metrics = 0.5, {}
        if val_records:
            val_loader = build_calibration_loader(
                val_records, mconf.image_size, args.batch, "nih"
            )
            threshold, thr_metrics = calibrate_threshold(
                float_model, val_loader, device, args.target_recall
            )
        print(f"[quantize] operating threshold={threshold:.4f} "
              f"at recall>={args.target_recall:.2f} -> {thr_metrics}")

        if args.qat_epochs > 0:
            print(f"[quantize] running QAT for {args.qat_epochs} epoch(s) on "
                  f"{min(args.limit_calib, len(calib_records))} calibration crops")
            qat_model = prepare_qat_model(float_model)
            qat_loader = build_calibration_loader(
                calib_records[: args.limit_calib], mconf.image_size,
                args.qat_batch, "nih",
            )
            qat_metrics = run_qat_calibration(
                qat_model, qat_loader, device, epochs=args.qat_epochs, lr=args.qat_lr
            )
            print(f"[quantize] QAT validation: recall={qat_metrics.get('recall', float('nan')):.4f} "
                  f"spec={qat_metrics.get('specificity', float('nan')):.4f}")

            # KEY STEP: transplant the QAT-trained float weights so the final
            # INT8 graph is genuinely seeded by quantization-aware training.
            float_model = extract_qat_weights(
                qat_model, build_model(mconf, pretrained=False)
            )
            if val_records:
                threshold, thr_metrics = calibrate_threshold(
                    float_model, val_loader, device, args.target_recall
                )
                print(f"[quantize] post-QAT threshold={threshold:.4f} "
                      f"-> {thr_metrics}")

        # Export the (possibly QAT-seeded) graph, then ORT INT8-quantize.
        fp32_onnx = export_onnx(float_model, fp32_onnx_path, mconf.image_size,
                                opset=args.opset)
        print(f"[quantize] fp32 onnx: {fp32_onnx} "
              f"({fp32_onnx.stat().st_size / 1e6:.2f} MB)")
        calib_loader = build_calibration_loader(
            calib_records[: args.limit_calib], mconf.image_size,
            args.calib_batch, "nih",
        )
        quantize_onnx_int8(fp32_onnx, args.output, calib_loader,
                           per_channel=not args.no_per_channel)

    # ---- recalibrate the threshold on the graph that actually ships -------
    # The threshold above was chosen on the float model. INT8 quantisation
    # moves the operating point (measured on this model: 0.83 decision
    # agreement float-vs-INT8 when the float threshold is reused), and the app
    # reads the threshold back from this sidecar, so a stale value silently
    # mis-operates. Recalibrate on the exported graph and record both.
    graph_threshold, graph_metrics = recalibrate_threshold_on_graph(
        args.output, calib_records, mconf.image_size, args.batch,
        args.target_recall,
    )
    if calib_records:
        print(f"[quantize] threshold on the EXPORTED graph: {graph_threshold:.4f} "
              f"(float-model threshold was {threshold:.4f}) -> {graph_metrics}")
        threshold = graph_threshold or threshold
        thr_metrics = graph_metrics or thr_metrics

        # Measure both graphs on the SAME held-out split so the quantisation
        # cost is recorded, not guessed. INT8 halves specificity on this model
        # (see MODEL_CARD.md) - if that ever changes, this is where it shows up.
        if not args.no_quant_metrics:
            try:
                fp32_metrics = _graph_metrics(fp32_onnx, calib_records,
                                              mconf.image_size, args.batch)
                int8_metrics = _graph_metrics(args.output, calib_records,
                                              mconf.image_size, args.batch)
                quant_cost = {
                    "fp32_export": fp32_metrics,
                    "int8_export": int8_metrics,
                    "auc_cost": round(fp32_metrics.get("auc", float("nan"))
                                      - int8_metrics.get("auc", float("nan")), 4),
                    "specificity_cost": round(
                        fp32_metrics.get("specificity", float("nan"))
                        - int8_metrics.get("specificity", float("nan")), 4),
                }
                print(f"[quantize] quantisation cost: AUC "
                      f"{fp32_metrics.get('auc', float('nan')):.4f} -> "
                      f"{int8_metrics.get('auc', float('nan')):.4f}, specificity "
                      f"{fp32_metrics.get('specificity', float('nan')):.3f} -> "
                      f"{int8_metrics.get('specificity', float('nan')):.3f}")
            except Exception as exc:  # pragma: no cover - measurement is optional
                print(f"[quantize] could not measure quantisation cost: {exc}")
                quant_cost = {}
        else:
            quant_cost = {}
    else:
        quant_cost = {}

    if args.fp32_output:
        dst = Path(args.fp32_output)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fp32_onnx, dst)
        print(f"[quantize] FP32 export kept at {dst} "
              f"({dst.stat().st_size / 1e6:.2f} MB) - accuracy-first alternative")

    size_mb = output.stat().st_size / 1e6
    bench = benchmark_onnx(output, mconf.image_size, runs=args.bench_runs)

    meta = {
        "model": "TinyMalariaNet",
        "architecture": "MobileNetV3-Small",
        "quantization": "INT8 (QDQ)" if not args.no_per_channel else "INT8 (QOperator)",
        "source_checkpoint": str(weights_path),
        "output": str(output),
        "size_mb": round(size_mb, 3),
        "size_bytes": output.stat().st_size,
        "input_shape": [1, 3, mconf.image_size, mconf.image_size],
        "input_name": "input",
        "output_name": "logit",
        "output_type": "logit",
        "decision_threshold": threshold,
        "positive_label": 1,
        "classes": {"0": "Uninfected", "1": "Parasitized"},
        "recall_target": args.target_recall,
        "threshold_calibration_metrics": thr_metrics,
        "threshold_calibrated_on": "exported_graph" if not args.no_per_channel else "float_model",
        "quantization_accuracy_cost": quant_cost,
        "latency_ms_cpu": round(bench["latency_ms"], 3),
        "calibration_source": calib_source,
        # True when the artifact was calibrated on the synthetic mock crops
        # shipped in data/sample. Such a model exercises the pipeline only and
        # carries no real diagnostic performance (see MODEL_CARD.md).
        "smoke_test_artifact": calib_source.startswith("synthetic"),
        "source_dataset": ckpt.get("dataset"),
        "license": "Apache-2.0",
        "intended_use": "Edge pre-screening, NOT a diagnostic device",
    }
    sidecar = output.with_suffix(".json")
    sidecar.write_text(json.dumps(meta, indent=2))

    print(f"[quantize] INT8 model written: {output} ({size_mb:.2f} MB)")
    print(f"[quantize] latency ~{bench['latency_ms']:.2f} ms/crop (CPU, 1 thread)")
    print(f"[quantize] metadata: {sidecar}")
    ok = size_mb < args.max_mb
    print(f"[quantize] footprint target <{args.max_mb} MB -> "
          f"{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Quantization-Aware Training + INT8 ONNX export for TinyMalariaNet"
    )
    p.add_argument("--weights", type=str, required=True, help="float .pt checkpoint")
    p.add_argument("--target", type=str, default="onnx_int8",
                   choices=["onnx_int8", "onnx_int8_qoperator"])
    p.add_argument("--output", type=str, default="models/tinymalaria_2.1mb_int8.onnx")
    p.add_argument("--calib-dataset", type=str, default="nih_sample",
                   help="dataset used for INT8 calibration + threshold tuning")
    p.add_argument("--calib-batch", type=int, default=32)
    p.add_argument("--limit-calib", type=int, default=256)
    p.add_argument("--target-recall", type=float, default=0.98)
    p.add_argument("--qat-epochs", type=int, default=0,
                   help="0 = PTQ only; >0 runs real QAT fine-tuning")
    p.add_argument("--qat-batch", type=int, default=32)
    p.add_argument("--qat-lr", type=float, default=1e-5)
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--work-dir", type=str, default="build/quant")
    p.add_argument("--no-per-channel", action="store_true",
                   help="use legacy per-tensor QOperator quantisation")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--bench-runs", type=int, default=100)
    p.add_argument("--max-mb", type=float, default=2.5,
                   help="footprint target from the design doc")
    p.add_argument("--fp32-output", type=str, default=None,
                   help="also keep the FP32 ONNX at this path (accuracy-first "
                        "alternative; ~3.5x the size)")
    p.add_argument("--no-quant-metrics", action="store_true",
                   help="skip measuring the FP32-vs-INT8 accuracy cost")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.target == "onnx_int8_qoperator":
        args.no_per_channel = True
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
