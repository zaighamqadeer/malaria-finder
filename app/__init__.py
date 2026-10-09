"""TinyMalariaNet edge application package.

* :mod:`app.inference` - raw FOV image -> OpenCV segmenter -> INT8 classifier
  -> parasitemia -> deterministic triage template (EN / UR / PL).
* :mod:`app.app` - Gradio UI that wires the same pipeline to an upload widget.

This module is intentionally dependency-free so that importing ``app`` never
pulls in torch/onnxruntime; those are imported lazily inside the functions that
need them. That keeps CLI helpers such as ``python app/app.py --help`` fast.
"""

__all__ = ["inference"]
__version__ = "0.1.0"
