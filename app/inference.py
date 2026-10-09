"""End-to-end TinyMalariaNet inference pipeline.

    raw FOV image -> OpenCV segmenter -> MobileNetV3-Small INT8 -> parasitemia
                  -> deterministic triage template (EN / UR / PL)

No LLM is involved at inference: triage is a lookup into clinician-vetted
templates selected by the computed parasitemia, which keeps the on-device path
deterministic, auditable and reproducible.

Usage
-----
    python -m app.inference --image data/phone_test/frame.jpg \
        --model models/tinymalaria_2.1mb_int8.onnx --lang ur
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from src.segment import SegmentConfig, segment_cells  # noqa: E402

TRIAGE_PATH = APP_DIR / "triage.json"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Minimum segmented cells before a parasitemia percentage is meaningful.
MIN_CELLS_FOR_ESTIMATE = 20


# --------------------------------------------------------------------------- #
# Classifier backends
# --------------------------------------------------------------------------- #
class Classifier:
    """Abstract classifier interface (returns Parasitized probabilities)."""

    def predict(self, crops: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover
        pass


class ONNXClassifier(Classifier):
    """INT8 ONNX runtime classifier (the production edge path)."""

    def __init__(self, model_path: str | Path, threshold: float = 0.5,
                 providers: list[str] | None = None) -> None:
        import onnxruntime as ort

        self.model_path = Path(model_path)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(self.model_path), opts,
            providers=providers or ["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.threshold = threshold

        sidecar = self.model_path.with_suffix(".json")
        self.metadata: dict[str, Any] = {}
        if sidecar.exists():
            self.metadata = json.loads(sidecar.read_text())
            self.threshold = float(self.metadata.get("decision_threshold", threshold))
        # Populated from the export sidecar so the UI can flag placeholder
        # artifacts that were calibrated on synthetic mock crops only.
        self.smoke_test = bool(self.metadata.get("smoke_test_artifact", False))

    def predict(self, crops: np.ndarray) -> np.ndarray:
        if len(crops) == 0:
            return np.zeros(0, dtype=np.float32)
        out = self.session.run(None, {self.input_name: crops})[0]
        out = np.asarray(out, dtype=np.float32).reshape(-1)
        # Model emits a logit -> sigmoid. If it already emits probabilities the
        # network ends in a sigmoid and out is already in [0, 1].
        if out.size and (out.min() < 0.0 or out.max() > 1.0):
            out = 1.0 / (1.0 + np.exp(-out))
        return out

    def close(self) -> None:
        self.session = None  # type: ignore[assignment]


class TorchClassifier(Classifier):
    """Fallback classifier over a float/quantized .pt checkpoint."""

    def __init__(self, checkpoint: str | Path, threshold: float = 0.5,
                 device: str = "cpu") -> None:
        import torch

        from src.model import ModelConfig, build_model
        from src.train import select_device

        self.torch = torch
        ckpt = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
        mconf = ModelConfig(**ckpt.get("model_config", {})) if ckpt.get("model_config") \
            else ModelConfig()
        self.model = build_model(mconf, pretrained=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.device = select_device(device)
        self.model.to(self.device).eval()
        self.threshold = threshold

    def predict(self, crops: np.ndarray) -> np.ndarray:
        if len(crops) == 0:
            return np.zeros(0, dtype=np.float32)
        with self.torch.no_grad():
            tensor = self.torch.from_numpy(crops).to(self.device)
            logits = self.model(tensor)
            return self.torch.sigmoid(logits).float().cpu().numpy().reshape(-1)

    def close(self) -> None:
        self.model = None  # type: ignore[assignment]


def build_classifier(model_path: str | Path, threshold: float = 0.5,
                     device: str = "cpu") -> Classifier:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    if path.suffix == ".onnx":
        return ONNXClassifier(path, threshold=threshold)
    return TorchClassifier(path, threshold=threshold, device=device)


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def preprocess_crops(bgr_crops: list[np.ndarray], image_size: int = 128) -> np.ndarray:
    """BGR uint8 crops -> NCHW float32 in ImageNet normalisation."""
    out = np.empty((len(bgr_crops), 3, image_size, image_size), dtype=np.float32)
    for i, crop in enumerate(bgr_crops):
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if rgb.shape[:2] != (image_size, image_size):
            rgb = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        out[i] = np.transpose(rgb, (2, 0, 1))
    return out


# --------------------------------------------------------------------------- #
# Triage
# --------------------------------------------------------------------------- #
def load_triage() -> dict[str, Any]:
    return json.loads(TRIAGE_PATH.read_text(encoding="utf-8"))


def classify_parasitemia(percent: float) -> str:
    """Map a parasitemia percentage onto a severity level (WHO-aligned cutoffs)."""
    if percent <= 0.0:
        return "negative"
    if percent < 1.0:
        return "low"
    if percent < 5.0:
        return "mild"
    if percent < 10.0:
        return "moderate"
    return "severe"


def render_triage(result: dict[str, Any], lang: str = "en") -> dict[str, str]:
    """Select the localized template for the computed severity."""
    triage = load_triage()
    if lang not in triage["supported_languages"]:
        lang = triage["default_language"]

    severity = result["severity"]
    block = triage["severity_levels"][severity]
    warnings = triage.get("segmentation_warnings", {})

    text = {
        "language": lang,
        "severity": severity,
        "urgency": block["urgency"],
        "header": block["header"][lang],
        "action": block["action"][lang],
        "recommendation": block["recommendation"][lang],
        "disclaimer": triage["disclaimer"][lang],
        "expert_review_required": block["require_expert_review"],
        "warnings": [],
    }
    if result["num_cells"] == 0:
        text["warnings"].append(warnings["no_cells"][lang])
    elif result["num_cells"] < MIN_CELLS_FOR_ESTIMATE:
        text["warnings"].append(warnings["too_few_cells"][lang])

    return text


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
@dataclass
class PipelineConfig:
    image_size: int = 128
    threshold_override: float | None = None
    language: str = "en"
    save_crops: bool = False
    crops_dir: str | Path | None = None
    confidence_filter: float = 0.0
    seg_cfg: SegmentConfig | None = None


@dataclass
class InferenceResult:
    image_path: str
    num_cells: int
    parasitized_cells: int
    parasitemia_percent: float
    severity: str
    probabilities: np.ndarray = field(default=None, repr=False)
    box: list[tuple[int, int, int, int]] = field(default_factory=list, repr=False)
    triage: dict[str, str] = field(default_factory=dict)
    timing_ms: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "image_path": self.image_path,
            "num_cells": int(self.num_cells),
            "parasitized_cells": int(self.parasitized_cells),
            "parasitemia_percent": round(float(self.parasitemia_percent), 4),
            "severity": self.severity,
            "triage": self.triage,
            "warnings": self.triage.get("warnings", []),
            "timing_ms": {k: round(v, 3) for k, v in self.timing_ms.items()},
            "model": getattr(self, "_model_name", "unknown"),
        }


class MalariaPipeline:
    def __init__(self, model_path: str | Path, cfg: PipelineConfig | None = None) -> None:
        self.cfg = cfg or PipelineConfig()
        self.classifier = build_classifier(model_path, device="cpu")
        self.seg_cfg = self.cfg.seg_cfg or SegmentConfig(crop_size=self.cfg.image_size)

    def run(self, image_path: str | Path, cfg: PipelineConfig | None = None) -> InferenceResult:
        cfg = cfg or self.cfg
        t_start = time.perf_counter()

        tmp_dir = None
        if not cfg.save_crops:
            import tempfile

            tmp_dir = Path(tempfile.mkdtemp(prefix="tinymalaria_"))
            out_dir = tmp_dir
        else:
            out_dir = Path(cfg.crops_dir or "data/inference_crops") / Path(image_path).stem
            out_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        candidates = segment_cells(image_path, cfg=self.seg_cfg, output_dir=out_dir)
        t_seg = (time.perf_counter() - t0) * 1000.0

        crops = [cv2.imread(c.path, cv2.IMREAD_COLOR) for c in candidates]
        crops = [c for c in crops if c is not None]

        t0 = time.perf_counter()
        if crops:
            batch = preprocess_crops(crops, cfg.image_size)
            probs = self.classifier.predict(batch)
        else:
            probs = np.zeros(0, dtype=np.float32)
        t_cls = (time.perf_counter() - t0) * 1000.0

        expected_thr = getattr(self.classifier, "threshold", 0.5)
        thr = cfg.threshold_override if cfg.threshold_override is not None else expected_thr

        predicted = (probs >= thr).astype(int)
        n_pos = int(predicted.sum())
        n_cells = int(len(probs))
        parasitemia = 100.0 * n_pos / n_cells if n_cells else 0.0

        severity = classify_parasitemia(parasitemia)
        result = InferenceResult(
            image_path=str(image_path),
            num_cells=n_cells,
            parasitized_cells=n_pos,
            parasitemia_percent=parasitemia,
            severity=severity,
            probabilities=probs,
            box=[(c.x, c.y, c.w, c.h) for c in candidates],
            timing_ms={
                "segment_ms": t_seg,
                "classify_ms": t_cls,
                "total_ms": (time.perf_counter() - t_start) * 1000.0,
            },
        )
        result._threshold = thr
        result.triage = render_triage(
            {
                "severity": severity,
                "num_cells": n_cells,
                "parasitemia_percent": parasitemia,
                "parasitized_cells": n_pos,
            },
            lang=cfg.language,
        )
        result._model_name = str(getattr(self.classifier, "model_path", "torch"))
        if getattr(self.classifier, "smoke_test", False):
            # The shipped offline artifact is calibrated on synthetic crops; say
            # so loudly instead of presenting its numbers as clinical output.
            result.triage["warnings"].append(
                "SYNTHETIC-CALIBRATED ARTIFACT: this model was calibrated on "
                "mock crops, so it exercises the pipeline only and carries no "
                "real diagnostic performance. Retrain on the NIH + Makerere data "
                "before relying on any number it produces (see MODEL_CARD.md)."
            )

        if tmp_dir is not None:
            import shutil

            shutil.rmtree(tmp_dir, ignore_errors=True)

        return result

    def close(self) -> None:
        self.classifier.close()

    def __enter__(self) -> "MalariaPipeline":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def annotate_image(image_path: str | Path, result: InferenceResult,
                   output_path: str | Path | None = None) -> np.ndarray:
    """Draw bounding boxes; red = predicted parasitized, green = uninfected.

    Writes ``output_path`` when given and returns the annotated array.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(image_path)
    thr = getattr(result, "_threshold", 0.5)
    for (x, y, w, h), p in zip(result.box, result.probabilities):
        colour = (0, 0, 255) if p >= thr else (0, 200, 0)
        cv2.rectangle(img, (x, y), (x + w, y + h), colour, 2)
    if output_path is not None:
        cv2.imwrite(str(output_path), img)
    return img


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Run TinyMalariaNet on a raw FOV image")
    p.add_argument("--image", required=True)
    p.add_argument("--model", default="models/tinymalaria_2.1mb_int8.onnx")
    p.add_argument("--lang", default="en", choices=["en", "ur", "pl"])
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--save-crops", action="store_true")
    p.add_argument("--crops-dir", default="data/inference_crops")
    args = p.parse_args(argv)

    cfg = PipelineConfig(
        image_size=args.image_size,
        threshold_override=args.threshold,
        language=args.lang,
        save_crops=args.save_crops,
        crops_dir=args.crops_dir,
    )
    pipe = MalariaPipeline(args.model, cfg)
    result = pipe.run(args.image, cfg)
    pipe.close()

    payload = result.to_json()
    payload["_threshold"] = getattr(pipe.classifier, "threshold", 0.5)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
