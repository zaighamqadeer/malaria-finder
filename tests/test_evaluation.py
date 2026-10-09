"""Evaluation and configuration-integrity tests."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.evaluate import (
    _iou,
    confusion_at_threshold,
    evaluate_segmentation,
    sensitivity_first_threshold,
)
from src.quantize import calibrate_threshold


class TestIoUHelper:
    def test_identical_boxes(self):
        assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0

    def test_disjoint_boxes(self):
        assert _iou((0, 0, 10, 10), (100, 100, 10, 10)) == 0.0

    def test_partial_overlap(self):
        # two 10x10 boxes offset by 5 -> intersection 5*10=50, union 150
        assert _iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(50 / 150)

    def test_zero_area_union_is_zero(self):
        assert _iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0


class TestThresholdSelection:
    def test_high_threshold_preferred_when_recall_allows(self):
        # perfectly separable scores -> a high threshold keeps recall at 1.0
        y = np.array([0, 0, 0, 1, 1, 1])
        p = np.array([0.05, 0.10, 0.15, 0.85, 0.90, 0.95])
        t, info = sensitivity_first_threshold(y, p, target_recall=0.98)
        assert info["sensitivity"] == 1.0
        assert t > 0.5
        assert info["specificity"] == 1.0

    def test_falls_back_when_recall_is_unreachable(self):
        # With no positive cases at all, sensitivity can never reach the
        # target, so the code must fall back to 0.5 and say why.
        y = np.array([0, 0, 0, 0])
        p = np.array([0.9, 0.92, 0.95, 0.1])
        t, info = sensitivity_first_threshold(y, p, target_recall=0.98)
        assert t == 0.5
        assert "note" in info

    def test_prefers_higher_specificity_among_recall_clearing_cuts(self):
        # a noisy-but-separable case: recall must stay 1.0, specificity should
        # be the best available at that recall
        rng = np.random.default_rng(0)
        y = np.array([0] * 50 + [1] * 50)
        p = np.concatenate([rng.uniform(0, 0.4, 50), rng.uniform(0.6, 1.0, 50)])
        t, info = sensitivity_first_threshold(y, p, target_recall=0.98)
        assert info["sensitivity"] >= 0.98
        assert info["specificity"] >= 0.9

    def test_confusion_counts(self):
        cm = confusion_at_threshold([0, 1, 1, 0], [0.9, 0.1, 0.8, 0.2], 0.5)
        assert cm == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}

    def test_quantize_calibration_matches_threshold_semantics(self):
        import torch

        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        p = np.array([0.05, 0.10, 0.20, 0.30, 0.70, 0.80, 0.90, 0.95])
        device = torch.device("cpu")
        t, metrics = calibrate_threshold(_FakeModel(p), _FakeLoader(y), device,
                                          target_recall=0.98)
        assert metrics["sensitivity"] >= 0.98
        assert t > 0.3  # not a degenerate threshold


class _FakeModel:
    def __init__(self, probs):
        logits = np.log(np.asarray(probs) / (1 - np.asarray(probs)))
        self._logits = logits

    def eval(self):  # noqa: D102 - mimic torch API
        return self

    def __call__(self, *_):
        import torch

        return torch.from_numpy(self._logits).float()


class _FakeLoader:
    """Yields (float image batch, tensor target) like a real torch DataLoader."""

    def __init__(self, labels):
        self.labels = np.asarray(labels, dtype=np.float32)

    def __iter__(self):
        import torch

        yield (torch.zeros(len(self.labels), 3, 8, 8),
               torch.from_numpy(self.labels).reshape(-1, 1))


class TestSegmentationEvaluation:
    @staticmethod
    @pytest.fixture(scope="class")
    def seg_results(repo_root):
        gt = repo_root / "data" / "phone_test" / "ground_truth.json"
        if not gt.exists():
            pytest.skip("no synthetic phone_test frames")
        return evaluate_segmentation(gt.parent, iou_threshold=0.3,
                                     target_recall=0.80)

    def test_design_target_met(self, seg_results):
        assert seg_results["target_met"], (
            f"segmentation recall {seg_results['recall']:.3f} "
            f"< target {seg_results['target_recall']}"
        )
        assert seg_results["recall"] > 0.80

    def test_reports_reasonable_precision(self, seg_results):
        assert seg_results["predicted"] > 0
        assert seg_results["precision"] > 0.5

    def test_every_image_scored(self, seg_results, repo_root):
        n_gt_images = len(json.loads(
            (repo_root / "data" / "phone_test" / "ground_truth.json").read_text()
        )["images"])
        assert len(seg_results["per_image"]) == n_gt_images


class TestConfigIntegrity:
    def test_devcontainer_is_valid_json(self, repo_root):
        cfg = json.loads((repo_root / ".devcontainer" / "devcontainer.json").read_text())
        assert "image" in cfg
        assert "postCreateCommand" in cfg
        assert isinstance(cfg["forwardPorts"], list)
        assert 7860 in cfg["forwardPorts"]

    def test_triage_is_valid_json(self, repo_root):
        data = json.loads((repo_root / "app" / "triage.json").read_text(encoding="utf-8"))
        assert data["default_language"] in data["supported_languages"]

    def test_requirements_pins_the_onnx_export_dependency(self, repo_root):
        text = (repo_root / "requirements.txt").read_text()
        # torch.onnx.export needs onnxscript; its absence broke the export step
        assert "onnxscript" in text
        assert "onnx" in text
        assert "opencv" in text

    def test_repo_layout_matches_the_design_doc(self, repo_root):
        for rel in ("src/segment.py", "src/dataset.py", "src/model.py",
                    "src/train.py", "src/quantize.py", "src/evaluate.py",
                    "app/triage.json", "app/app.py", "requirements.txt",
                    "README.md", "MODEL_CARD.md", "LICENSE", "Makefile"):
            assert (repo_root / rel).exists(), f"missing {rel}"

    def test_license_is_apache_2(self, repo_root):
        head = (repo_root / "LICENSE").read_text()[:200]
        assert "Apache License" in head

    def test_gitignore_covers_build_output(self, repo_root):
        text = (repo_root / ".gitignore").read_text()
        for pattern in ("checkpoints/", "models/", "models_qat/", "__pycache__/"):
            assert pattern in text, f"{pattern} not ignored"
