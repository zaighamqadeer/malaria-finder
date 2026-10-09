"""Publish TinyMalariaNet to the Hugging Face Hub (design doc Day 5).

Two artifacts go to two places:

* a **model** repository holding the INT8 ONNX graph, its calibrated sidecar,
  and the model card;
* a **space** repository holding the Gradio demo, which downloads the weights at
  cold start via the ``TINYMALARIA_MODEL_URL`` secret.

The weights inherit the licence of the data they were trained on, so this script
refuses to publish an artifact whose sidecar does not declare one, and prints the
training-data licences for you to check before the upload happens.

Usage
-----
    huggingface-cli login                       # or: export HF_TOKEN=hf_...
    python scripts/publish_hf.py --namespace your-username
    python scripts/publish_hf.py --namespace you --model-repo tinymalaria-net \
        --space-repo tinymalaria-demo --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

MODEL_FILES = [
    ("models/tinymalaria_2.1mb_int8.onnx", "tinymalaria_int8.onnx"),
    ("models/tinymalaria_2.1mb_int8.json", "tinymalaria_int8.json"),
    ("MODEL_CARD.md", "MODEL_CARD.md"),
]

# Files the Space needs. The weights are NOT among them: they are pulled at
# runtime from TINYMALARIA_MODEL_URL, which keeps the Space image small and lets
# the model be updated without rebuilding the demo.
# NOTE: app/__init__.py is deliberately NOT staged. It makes the repo's `app`
# directory a package, which is wrong for a Space: there the root holds app.py
# directly, and a stray __init__.py there would make the root a package too.
SPACE_FILES = [
    ("app/app.py", "app.py"),
    ("app/inference.py", "inference.py"),
    ("app/triage.json", "triage.json"),
    ("requirements.txt", "requirements.txt"),
]

SPACE_README = """---
title: TinyMalariaNet
emoji: 🔬
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: "6.30.0"
app_file: app.py
app_port: 7860
pinned: false
license: apache-2.0
short_description: 2.1 MB two-stage edge screening for smartphone microscopes
---

# TinyMalariaNet — Edge Malaria Screening

Upload a **raw, uncropped** smartphone photo taken through a microscope eyepiece.
OpenCV isolates candidate red blood cells; a MobileNetV3-Small INT8 model
classifies each one; triage text is selected deterministically by computed
parasitemia in English, Urdu or Polish.

## Setup

Set the `TINYMALARIA_MODEL_URL` secret to the model file, e.g.
`hf://your-username/tinymalaria-net/tinymalaria_int8.onnx`. The Space downloads
it once at cold start and caches it.

> **Research / education / preliminary screening only. This is NOT a diagnostic
> medical device.** Always confirm with laboratory microscopy and a qualified
> healthcare professional.
"""


def _require_hub() -> None:
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        sys.exit("huggingface_hub is required: pip install huggingface_hub")


def _load_sidecar() -> dict:
    path = REPO_ROOT / "models" / "tinymalaria_2.1mb_int8.json"
    if not path.exists():
        sys.exit(f"{path} not found - run src/quantize.py first")
    return json.loads(path.read_text())


def _check_licensing(meta: dict) -> list[str]:
    """Refuse to publish something that would misrepresent its provenance."""
    problems: list[str] = []
    if meta.get("smoke_test_artifact"):
        problems.append(
            "the sidecar flags this as a SYNTHETIC smoke-test artifact; "
            "retrain on real data before publishing"
        )
    if not meta.get("license"):
        problems.append("the sidecar declares no license")
    if meta.get("calibration_source", "").startswith("synthetic"):
        problems.append(
            f"calibrated on {meta['calibration_source']} (non-real data)"
        )
    return problems


def publish_model(namespace: str, repo: str, dry_run: bool) -> str:
    from huggingface_hub import HfApi

    repo_id = f"{namespace}/{repo}"
    missing = [str(REPO_ROOT / src) for src, _ in MODEL_FILES
               if not (REPO_ROOT / src).exists()]
    if missing:
        sys.exit(f"missing files to publish: {missing}")

    api = HfApi()
    print(f"[hf] model repo -> {repo_id}")
    if dry_run:
        print("[hf] dry run: would upload "
              + ", ".join(dst for _, dst in MODEL_FILES))
        return repo_id

    api.create_repo(repo_id, repo_type="model", exist_ok=True, private=False)
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp)
        for src, dst in MODEL_FILES:
            shutil.copy2(REPO_ROOT / src, staged / dst)
        api.upload_folder(folder_path=str(staged), repo_id=repo_id,
                          repo_type="model",
                          commit_message="Update TinyMalariaNet INT8 weights")
    print(f"[hf] published: https://huggingface.co/{repo_id}")
    return repo_id


def publish_space(namespace: str, repo: str, model_repo: str, dry_run: bool) -> str:
    from huggingface_hub import HfApi

    repo_id = f"{namespace}/{repo}"
    print(f"[hf] space repo -> {repo_id}")
    if dry_run:
        print("[hf] dry run: would upload "
              + ", ".join(dst for _, dst in SPACE_FILES) + ", README.md")
        return repo_id

    missing = [src for src, _ in SPACE_FILES if not (REPO_ROOT / src).exists()]
    if missing:
        sys.exit(f"missing files for the Space: {missing}")

    api = HfApi()
    api.create_repo(repo_id, repo_type="space", exist_ok=True,
                    space_sdk="gradio")
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp)
        for src, dst in SPACE_FILES:
            dest = staged / dst
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO_ROOT / src, dest)
        # The Space must import src.* as a package.
        shutil.copytree(REPO_ROOT / "src", staged / "src",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (staged / "README.md").write_text(SPACE_README)
        api.upload_folder(folder_path=str(staged), repo_id=repo_id,
                          repo_type="space",
                          commit_message="Deploy TinyMalariaNet demo")
    print(f"[hf] published: https://huggingface.co/spaces/{repo_id}")
    print(f"[hf] set the TINYMALARIA_MODEL_URL secret to:\n"
          f"    hf://{model_repo}/tinymalaria_int8.onnx")
    return repo_id


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Publish TinyMalariaNet to the Hub")
    p.add_argument("--namespace", required=True, help="HF user or org")
    p.add_argument("--model-repo", default="tinymalaria-net")
    p.add_argument("--space-repo", default="tinymalaria-demo")
    p.add_argument("--skip-space", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="publish even if the licensing check complains")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    _require_hub()
    meta = _load_sidecar()
    print(f"[hf] artifact: {meta['output']} "
          f"({meta.get('size_mb')} MB, {meta.get('quantization')})")

    problems = _check_licensing(meta)
    if problems:
        print("[hf] REFUSING to publish:")
        for problem in problems:
            print(f"  - {problem}")
        if not args.force:
            print("[hf] re-run with --force only if this is intentional")
            return 2

    repo_id = publish_model(args.namespace, args.model_repo, args.dry_run)
    if not args.skip_space:
        publish_space(args.namespace, args.space_repo, repo_id, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
