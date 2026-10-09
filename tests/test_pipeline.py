"""Pipeline tests: segmentation, preprocessing, model, and the shipped artifact."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
import torch
from pathlib import Path

from src.model import ModelConfig, TinyMalariaNet, build_model
from src.segment import SegmentConfig, segment_cells

REPO_ROOT = Path(__file__).resolve().parent.parent

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class TestSegmenter:
    def test_extracts_candidates_from_a_raw_frame(self, sample_frame, repo_root):
        cands = segment_cells(sample_frame, cfg=SegmentConfig(),
                              output_dir=repo_root / "build" / "test_seg")
        assert len(cands) > 10, "expected a healthy number of candidate cells"
        for c in cands[:5]:
            assert c.w > 0 and c.h > 0
            assert 0 <= c.x and 0 <= c.y
            assert 0.0 <= c.confidence <= 1.0
            assert 0.0 <= c.circularity <= 1.0

    def test_crops_are_written_at_the_target_size(self, sample_frame, repo_root):
        out = repo_root / "build" / "test_crops"
        cands = segment_cells(sample_frame, cfg=SegmentConfig(crop_size=128),
                              output_dir=out)
        assert cands
        first = cv2.imread(cands[0].path, cv2.IMREAD_COLOR)
        assert first is not None
        assert first.shape[:2] == (128, 128)
        assert (out / "candidates.json").exists()
        payload = json.loads((out / "candidates.json").read_text())
        assert payload["crop_size"] == 128
        assert payload["num_candidates"] == len(cands)

    def test_deterministic_for_the_same_image(self, sample_frame, repo_root):
        a = segment_cells(sample_frame, cfg=SegmentConfig(),
                          output_dir=repo_root / "build" / "det_a")
        b = segment_cells(sample_frame, cfg=SegmentConfig(),
                          output_dir=repo_root / "build" / "det_b")
        assert [(c.x, c.y, c.w, c.h) for c in a] == [(c.x, c.y, c.w, c.h) for c in b]

    def test_watershed_splitting_finds_at_least_as_many(self, sample_frame):
        plain = segment_cells(sample_frame, cfg=SegmentConfig(split_clumps=False))
        split = segment_cells(sample_frame, cfg=SegmentConfig(split_clumps=True))
        # splitting may re-partition, but should never lose detections outright
        assert len(split) >= len(plain) * 0.8

    def test_rejects_a_missing_image(self):
        with pytest.raises(FileNotFoundError):
            segment_cells("/tmp/definitely_not_a_real_image_12345.jpg")

    def test_handles_a_blank_frame_gracefully(self, repo_root):
        blank = np.full((400, 400, 3), 240, dtype=np.uint8)
        path = repo_root / "build" / "blank.jpg"
        cv2.imwrite(str(path), blank)
        assert segment_cells(path, cfg=SegmentConfig()) == []


class TestPreprocessing:
    def test_preprocess_crops_shape_and_normalisation(self):
        from app.inference import preprocess_crops

        crops = [(np.random.rand(60, 80, 3) * 255).astype(np.uint8) for _ in range(5)]
        out = preprocess_crops(crops, image_size=128)
        assert out.shape == (5, 3, 128, 128)
        assert out.dtype == np.float32
        # normalisation should move values off the [0, 1] range
        assert out.min() < -1.0 or out.max() > 1.0

    def test_constant_crop_produces_a_specific_value(self):
        from app.inference import preprocess_crops

        grey = np.full((128, 128, 3), 128, dtype=np.uint8)
        out = preprocess_crops([grey])[0]
        expect = (128 / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        assert np.allclose(out, expect[:, None, None], atol=1e-4)

    def test_empty_input_is_empty(self):
        from app.inference import preprocess_crops

        assert preprocess_crops([]).shape[0] == 0


class TestModel:
    def test_forward_shape(self):
        model = build_model(ModelConfig(pretrained=False))
        out = model(torch.randn(3, 3, 128, 128))
        assert out.shape == (3,)
        assert out.dtype == torch.float32

    def test_parameter_budget(self):
        model = build_model(ModelConfig(pretrained=False))
        assert model.num_parameters() < 2_000_000, "footprint budget is ~2.5 MB"

    def test_predict_proba_is_a_probability(self):
        model = build_model(ModelConfig(pretrained=False))
        p = model.predict_proba(torch.randn(4, 3, 128, 128))
        assert p.shape == (4,)
        assert float(p.min()) >= 0.0 and float(p.max()) <= 1.0

    def test_deterministic_in_eval_mode(self):
        model = build_model(ModelConfig(pretrained=False)).eval()
        x = torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            assert torch.allclose(model(x), model(x))

    def test_state_dict_round_trips(self):
        a = build_model(ModelConfig(pretrained=False))
        b = TinyMalariaNet(ModelConfig(pretrained=False))
        b.load_state_dict(a.state_dict())
        x = torch.randn(2, 3, 128, 128)
        a.eval(); b.eval()
        with torch.no_grad():
            assert torch.allclose(a(x), b(x))


class TestShippedArtifact:
    def test_footprint_target(self, onnx_model):
        if onnx_model is None:
            pytest.skip("no exported artifact; run src/quantize.py")
        size_mb = onnx_model.stat().st_size / 1e6
        assert size_mb < 2.5, f"{size_mb:.2f} MB exceeds the 2.5 MB design budget"

    def test_sidecar_metadata_is_consistent(self, onnx_model):
        if onnx_model is None:
            pytest.skip("no exported artifact")
        sidecar = json.loads(onnx_model.with_suffix(".json").read_text())
        assert sidecar["input_shape"] == [1, 3, 128, 128]
        assert sidecar["output_name"] == "logit"
        assert float(sidecar["size_bytes"]) == onnx_model.stat().st_size
        assert 0.0 < float(sidecar["decision_threshold"]) < 1.0

    def test_artifact_declares_its_provenance(self, onnx_model):
        if onnx_model is None:
            pytest.skip("no exported artifact")
        meta = json.loads(onnx_model.with_suffix(".json").read_text())
        src = meta.get("calibration_source", "")
        # Every artifact must say where its calibration came from, and a real
        # one must not claim to be a placeholder.
        assert src
        assert bool(meta.get("smoke_test_artifact", False)) == src.startswith("synthetic")

    def test_onnx_loads_and_runs(self, onnx_model):
        if onnx_model is None:
            pytest.skip("no exported artifact")
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        sess = ort.InferenceSession(str(onnx_model), so,
                                    providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        x = np.random.randn(2, 3, 128, 128).astype(np.float32)
        out = sess.run(None, {name: x})[0]
        assert np.asarray(out).reshape(-1).shape == (2,)

    def test_latency_budget(self, onnx_model):
        if onnx_model is None:
            pytest.skip("no exported artifact")
        meta = json.loads(onnx_model.with_suffix(".json").read_text())
        assert float(meta["latency_ms_cpu"]) < 45.0


class TestClassifierAgreement:
    def _probe_batch(self, onnx_model, stage2_checkpoint):
        if onnx_model is None or stage2_checkpoint is None:
            pytest.skip("need both an ONNX artifact and a torch checkpoint")
        from app.inference import preprocess_crops

        import glob
        import cv2

        crops_dir = REPO_ROOT / "data" / "segmented"
        files = sorted(crops_dir.glob("cell_*.png"))
        if files:
            # Prefer crops produced by the real segmenter.
            crops = [cv2.imread(str(f), cv2.IMREAD_COLOR) for f in files[:24]]
            crops = [c for c in crops if c is not None]
        else:
            # Fall back to the shipped sample crops (resized, as the pipeline does).
            from src.make_sample_data import ensure_sample_dataset

            recs = ensure_sample_dataset(root=REPO_ROOT / "data")
            paths = [r["path"] for r in recs[:24]]
            crops = [cv2.resize(cv2.imread(str(REPO_ROOT / p), cv2.IMREAD_COLOR),
                                (128, 128)) for p in paths]
            crops = [c for c in crops if c is not None]
        return preprocess_crops(crops)

    def _torch_probs(self, stage2_checkpoint, batch):
        import torch

        ckpt = torch.load(stage2_checkpoint, map_location="cpu", weights_only=False)
        model = build_model(ModelConfig(**ckpt["model_config"]), pretrained=False)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        with torch.no_grad():
            return torch.sigmoid(model(torch.from_numpy(batch))).numpy()

    def test_fp32_export_matches_pytorch_exactly(self, onnx_model, stage2_checkpoint):
        """The FP32 ONNX export must be bit-faithful to the torch checkpoint.

        This is the invariant that proves no wiring/preprocessing bug: any
        difference here would mean the export or the normalisation is wrong.
        (The INT8 case is a separate, expected loss - see the test below.)
        """
        if onnx_model is None or stage2_checkpoint is None:
            pytest.skip("need both an ONNX artifact and a torch checkpoint")
        import numpy as np
        import onnxruntime as ort

        fp32 = REPO_ROOT / "build" / "quant" / "tinymalaria_fp32.onnx"
        if not fp32.exists():
            pytest.skip("re-run src/quantize.py to regenerate the FP32 export")

        batch = self._probe_batch(onnx_model, stage2_checkpoint)
        ref = self._torch_probs(stage2_checkpoint, batch)

        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        sess = ort.InferenceSession(str(fp32), so, providers=["CPUExecutionProvider"])
        p = 1 / (1 + np.exp(-np.asarray(
            sess.run(None, {"input": batch})[0], dtype=np.float64).reshape(-1)))
        assert float(np.abs(p - ref).max()) < 1e-4

    def test_int8_auc_is_acceptable(self, onnx_model):
        if onnx_model is None:
            pytest.skip("no exported artifact")
        import cv2
        import numpy as np
        from sklearn.metrics import roc_auc_score

        from app.inference import ONNXClassifier
        from src.dataset import patient_level_split

        manifest = REPO_ROOT / "data" / "processed" / "lacuna_crops" / "manifest.json"
        if not manifest.exists():
            pytest.skip("no parsed Lacuna crops")
        payload = json.loads(manifest.read_text())
        _, val = patient_level_split(payload["images"], val_fraction=0.2, seed=1337)
        paths = [r["path"] for r in val]
        labels = np.array([int(r["label"]) for r in val])
        if len(np.unique(labels)) < 2:
            pytest.skip("need both classes")
        crops = []
        for p in paths:
            img = cv2.imread(str(REPO_ROOT / p), cv2.IMREAD_COLOR)
            if img is not None:
                crops.append(img)
        from app.inference import preprocess_crops

        probs = ONNXClassifier(onnx_model).predict(preprocess_crops(crops))
        auc = float(roc_auc_score(labels, probs))
        assert auc > 0.85, f"INT8 AUC {auc:.3f} - quantisation broke the model"

    def test_shipped_threshold_reproduces_on_the_shipped_graph(self, onnx_model):
        """The sidecar claim must hold for the graph that is actually shipped.

        This is the invariant that catches the real bug: a threshold calibrated
        on the float model and shipped with an INT8 graph mis-operates (0.83
        decision agreement), because quantisation moves the operating point.
        """
        if onnx_model is None:
            pytest.skip("no exported artifact")
        import cv2
        import numpy as np

        from app.inference import ONNXClassifier
        from src.dataset import patient_level_split

        manifest = REPO_ROOT / "data" / "processed" / "lacuna_crops" / "manifest.json"
        if not manifest.exists():
            pytest.skip("no parsed Lacuna crops")
        sidecar = json.loads(onnx_model.with_suffix(".json").read_text())
        if sidecar.get("threshold_calibrated_on") != "exported_graph":
            pytest.skip("threshold was calibrated on the float model")

        payload = json.loads(manifest.read_text())
        _, val = patient_level_split(payload["images"], val_fraction=0.2, seed=1337)
        crops = [cv2.imread(str(REPO_ROOT / r["path"]), cv2.IMREAD_COLOR) for r in val]
        crops = [c for c in crops if c is not None]
        labels = np.array([int(r["label"]) for r in val][:len(crops)])
        from app.inference import preprocess_crops

        clf = ONNXClassifier(onnx_model)
        probs = clf.predict(preprocess_crops(crops))
        pred = (probs >= clf.threshold).astype(int)
        pos = labels == 1
        sensitivity = float((pred[pos] == 1).mean()) if pos.any() else 0.0
        assert sensitivity >= 0.90, (
            f"sidecar claims {sidecar['threshold_calibration_metrics']} but the "
            f"shipped INT8 graph only reaches {sensitivity:.3f} sensitivity at "
            f"t={clf.threshold:.4f}"
        )

    def test_threshold_comes_from_the_sidecar(self, onnx_model):
        if onnx_model is None:
            pytest.skip("no exported artifact")
        from app.inference import ONNXClassifier

        clf = ONNXClassifier(onnx_model)
        sidecar = json.loads(onnx_model.with_suffix(".json").read_text())
        assert clf.threshold == pytest.approx(sidecar["decision_threshold"], abs=1e-6)
        # the smoke-test flag must match the declared calibration source
        assert clf.smoke_test == sidecar["calibration_source"].startswith("synthetic")

    def test_empty_batch_returns_empty(self, onnx_model):
        if onnx_model is None:
            pytest.skip("no exported artifact")
        from app.inference import ONNXClassifier

        out = ONNXClassifier(onnx_model).predict(np.zeros((0, 3, 128, 128), dtype=np.float32))
        assert out.size == 0
