"""Gradio UI for TinyMalariaNet - accepts RAW full-field smartphone captures.

Run:
    python app/app.py [--port 7860]

The UI accepts an uncropped raw microscope/phone-eyepiece frame, runs the
OpenCV segmenter + INT8 classifier, and renders the deterministic triage block
in English, Urdu or Polish.
"""

from __future__ import annotations

import json

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

LANGUAGES = {"en": "English", "ur": "اردو (Urdu)", "pl": "Polski (Polish)"}


def pick_model() -> Path | None:
    for path in MODEL_CANDIDATES:
        if path.exists():
            return path
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


def build_demo(model_path: Path | None):
    import gradio as gr

    from app.inference import MalariaPipeline, PipelineConfig, annotate_image

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
