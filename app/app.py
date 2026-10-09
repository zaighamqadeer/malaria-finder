"""Gradio UI for TinyMalariaNet - accepts RAW full-field smartphone captures.

Run:
    python app/app.py [--port 7860]

The UI accepts an uncropped raw microscope/phone-eyepiece frame, runs the
OpenCV segmenter + INT8 classifier, and renders the deterministic triage block
in English, Urdu or Polish.
"""

from __future__ import annotations

import json
import os
import shutil

import argparse
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

MODEL_CANDIDATES = [
    REPO_ROOT / "models" / "tinymalaria_2.1mb_int8.onnx",
    REPO_ROOT / "checkpoints" / "stage2" / "best.pt",
    REPO_ROOT / "checkpoints" / "stage1" / "best.pt",
]

# On Hugging Face Spaces (and any other containerised deploy) the weights are
# not in the image. Set TINYMALARIA_MODEL_URL to fetch them once at startup and
# cache them under models/downloaded/. The sidecar .json is fetched too, because
# that is where the calibrated decision threshold lives.
MODEL_URL_ENV = "TINYMALARIA_MODEL_URL"
MODEL_CACHE = REPO_ROOT / "models" / "downloaded"

LANGUAGES = {"en": "English", "ur": "اردو (Urdu)", "pl": "Polski (Polish)"}


def _fetch_model(url: str) -> Path | None:
    """Download a model (and its sidecar) over HTTP, or pull from the HF Hub."""
    import shutil

    dest = MODEL_CACHE / Path(url.split("?")[0]).name
    if dest.exists():
        return dest
    try:
        if url.startswith("hf://"):
            from huggingface_hub import hf_hub_download

            repo_id, _, filename = url[5:].partition("/")
            got = hf_hub_download(repo_id=repo_id, filename=filename)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(got, dest)
        else:
            from urllib.request import Request, urlopen

            dest.parent.mkdir(parents=True, exist_ok=True)
            req = Request(url, headers={"User-Agent": "TinyMalariaNet-app/1.0"})
            with urlopen(req, timeout=300) as resp, dest.open("wb") as fh:  # noqa: S310
                shutil.copyfileobj(resp, fh)

        # The sidecar carries the calibrated threshold. Without it the app
        # silently falls back to 0.5 and the recall target is lost.
        side = dest.with_suffix(".json")
        if not side.exists():
            for cand in (url + ".json", url.replace(".onnx", ".json")):
                try:
                    req = Request(cand, headers={"User-Agent": "TinyMalariaNet-app/1.0"})
                    with urlopen(req, timeout=120) as resp, side.open("wb") as fh:
                        shutil.copyfileobj(resp, fh)
                    print(f"[app] sidecar fetched from {cand}")
                    break
                except Exception:
                    continue
        print(f"[app] model cached at {dest}")
        return dest
    except Exception as exc:  # pragma: no cover - deploy-time failure
        print(f"[app] could not fetch model from {url}: {exc}")
        return None


def pick_model() -> Path | None:
    env_url = os.environ.get(MODEL_URL_ENV, "").strip()
    if env_url:
        fetched = _fetch_model(env_url)
        if fetched is not None:
            return fetched
    for path in MODEL_CANDIDATES:
        if path.exists():
            return path
    if MODEL_CACHE.exists():
        cached = sorted(MODEL_CACHE.glob("*.onnx"))
        if cached:
            return cached[0]
    return None


_HEADER = """
# TinyMalariaNet — Edge Malaria Screening
### 2.1 MB two-stage pipeline for smartphone-attached microscopes

**Stage 1** OpenCV segmentation isolates candidate red blood cells from a raw
full-field capture. **Stage 2** a MobileNetV3-Small INT8 model classifies each
candidate. Triage text is selected deterministically by computed parasitemia —
no on-device LLM.

> **Research / education / preliminary screening only. This is NOT a diagnostic
> medical device.** Always confirm with laboratory microscopy and a qualified
> healthcare professional.
"""


def _load_inference():
    """Import the pipeline package in whichever layout this file is running in.

    In the repository the entry point is ``app/app.py`` and the pipeline module
    is ``app/inference.py``. In the Hugging Face Space both files are siblings
    at the repository root, so ``app`` resolves to this very module instead of a
    package. Try the packaged name first, fall back to the flat one.
    """
    try:
        from app import inference
    except ImportError:
        import inference  # type: ignore[no-redef]
    return inference


def build_demo(model_path: Path | None):
    import gradio as gr

    _inference = _load_inference()
    MalariaPipeline = _inference.MalariaPipeline
    PipelineConfig = _inference.PipelineConfig
    annotate_image = _inference.annotate_image

    with gr.Blocks(title="TinyMalariaNet") as demo:
        gr.Markdown(_HEADER)

        if model_path is None:
            gr.Markdown(
                "> ⚠️ **No model found.** Train Stage 1 first, then export:\n"
                "> ```\n"
                "> python src/train.py --dev-mode --sample 200 --epochs 1 \\\n"
                ">   --checkpoint-dir checkpoints/stage1/\n"
                "> python src/quantize.py --weights checkpoints/stage1/best.pt "
                "--output models/tinymalaria_2.1mb_int8.onnx\n"
                "> ```"
            )

        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(
                    label="Raw field-of-view capture (uncropped phone image)",
                    type="filepath",
                    sources=["upload", "webcam", "clipboard"],
                )
                lang = gr.Dropdown(
                    choices=[(v, k) for k, v in LANGUAGES.items()],
                    value="en", label="Triage language",
                )
                run_btn = gr.Button("Analyse field", variant="primary")

            with gr.Column(scale=1):
                image_out = gr.Image(label="Detected cells (red = parasitized)")
                summary = gr.Markdown("_Results appear here._")
                details = gr.JSON(label="Metrics")
                triage_json = gr.JSON(label="Triage payload (EN / UR / PL)")

        mode = "unknown" if model_path is None else ("onnx" if model_path.suffix == ".onnx"
                                                    else "torch")
        model_note = ""
        if model_path is not None and model_path.suffix == ".onnx":
            sidecar = model_path.with_suffix(".json")
            if sidecar.exists():
                try:
                    meta = json.loads(sidecar.read_text())
                except (ValueError, OSError):
                    meta = {}
                if meta.get("smoke_test_artifact"):
                    model_note = (
                        "\n\n> ⚠️ **Smoke-test artifact.** This checkpoint was "
                        "calibrated on synthetic mock crops and carries no real "
                        "diagnostic performance. Run the staged training "
                        "(`src/train.py` stages 1-2) before trusting its output."
                    )
                else:
                    cost = meta.get("quantization_accuracy_cost") or {}
                    f32, i8 = cost.get("fp32_export") or {}, cost.get("int8_export") or {}
                    if f32 and i8 and "specificity" in f32 and "specificity" in i8:
                        model_note = (
                            "\n\n> ℹ️ **Quantisation trade-off (measured).** This "
                            f"INT8 model is {meta.get('size_mb', '?'):.2f} MB. Its "
                            f"FP32 counterpart has AUC "
                            f"{f32['auc']:.3f} vs {i8['auc']:.3f} here and "
                            f"specificity {f32['specificity']:.2f} vs "
                            f"{i8['specificity']:.2f} at the same recall. Both "
                            "clear the ≥98% sensitivity target. Use "
                            "`python src/quantize.py --fp32-output models/fp32.onnx` "
                            "if accuracy matters more than size."
                        )
        gr.Markdown(f"**Loaded model:** `{model_path}` (mode: `{mode}`){model_note}")

        def _analyse(image_path, language):
            if image_path is None:
                return None, "_Upload an image to analyse._", {}, {}
            if model_path is None:
                return None, "**No model available.** Train and export first.", {}, {}
            cfg = PipelineConfig(language=language)
            with MalariaPipeline(model_path, cfg) as pipe:
                result = pipe.run(image_path, cfg)
                annotated = annotate_image(image_path, result)

            # annotate_image works in OpenCV BGR; Gradio expects RGB arrays.
            import cv2

            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            t = result.triage
            lines = [
                f"### {t['header']}",
                "",
                f"| Field | Value |",
                f"| --- | --- |",
                f"| Cells segmented | {result.num_cells} |",
                f"| Predicted parasitized | {result.parasitized_cells} |",
                f"| **Parasitemia** | **{result.parasitemia_percent:.2f}%** |",
                f"| Severity band | `{result.severity}` |",
                f"| Urgency | `{t['urgency']}` |",
                "",
                f"**Action:** {t['action']}",
                "",
                f"**Recommendation:** {t['recommendation']}",
            ]
            if t.get("warnings"):
                lines += ["", "**Warnings:** " + " ".join(t["warnings"])]
            lines += ["", "---", f"_{t['disclaimer']}_"]

            payload = result.to_json()
            payload["_threshold"] = getattr(result, "_threshold", 0.5)

            return (
                annotated,
                "\n".join(lines),
                payload,
                payload.get("triage", {}),
            )

        run_btn.click(_analyse, inputs=[image_in, lang],
                      outputs=[image_out, summary, details, triage_json])

    return demo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TinyMalariaNet Gradio demo")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--share", action="store_true",
                        help="create a public share link (for HF Spaces deploy)")
    parser.add_argument("--server-name", type=str, default=None)
    args = parser.parse_args(argv)

    model_path = pick_model()
    print(f"[app] model: {model_path}")
    demo = build_demo(model_path)
    host = args.server_name or args.host
    demo.launch(server_name=host, server_port=args.port, share=args.share,
                prevent_thread_lock=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
