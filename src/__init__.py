"""TinyMalariaNet edge classifier package.

Two-stage pipeline:
  1. OpenCV RBC segmenter (src/segment.py) isolates candidate cells from raw
     full-field microscope / smartphone eyepiece captures.
  2. MobileNetV3-Small INT8 classifier (this module + src/quantize.py) scores
     each candidate crop as Parasitized vs. Uninfected, tuned for high recall.
"""

__version__ = "0.1.0"
