# TinyMalariaNet

**A 2.1 MB two-stage edge screening model for microscope phone attachments.**

TinyMalariaNet takes a **raw, uncropped smartphone photo through a microscope
eyepiece**, isolates candidate red blood cells (RBCs), classifies each one as
*Parasitized* or *Uninfected*, and returns a localized triage message plus a
parasitemia percentage — all on-device, with no LLM and no network call.

```
[Raw phone FOV image]
        │
        ▼
[OpenCV RBC Segmenter]  ──► candidate crops (128×128)
        │
        ▼
[MobileNetV3-Small INT8] ──► Parasitized / Uninfected (high recall)
        │
        ▼
[Deterministic triage JSON] ──► localized advice + parasitemia
```

> **DISCLAIMER** — This software is for research, education and preliminary
> screening only. It is **NOT** a diagnostic medical device. Always confirm with
> laboratory microscopy and a qualified healthcare professional.

---

## 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` is pinned to the versions the smoke test was verified
against. `onnxscript` is required by `torch.onnx.export` — without it the export
step fails with `ModuleNotFoundError: No module named 'onnxscript'`.

## 2. Codespace quick start (no data download needed)

The repository ships an offline dry-run mode that synthesises mock RBC crops, so
the whole pipeline can be verified in a few minutes without touching a dataset.

```bash
# 1. Stage-1 training dry-run on synthetic crops
python src/train.py --dev-mode --sample 200 --epochs 1 \
    --checkpoint-dir checkpoints/stage1/

# 2. Segmentation on a raw field-of-view frame
python src/segment.py --test-image data/phone_test/sample_slide_000.jpg \
    --output-dir data/segmented

# 3. INT8 export (produces models/tinymalaria_2.1mb_int8.onnx)
python src/quantize.py --weights checkpoints/stage1/best.pt \
    --output models/tinymalaria_2.1mb_int8.onnx

# 4. Full inference on a raw frame
python -m app.inference --image data/phone_test/sample_slide_000.jpg \
    --model models/tinymalaria_2.1mb_int8.onnx --lang ur

# 5. UI
python app/app.py            # http://127.0.0.1:7860
```

Or run everything in one go: `make smoke`.

> ⚠️ The artifacts produced above are **smoke-test artifacts**: they are
> calibrated on synthetic mock crops, exercise the pipeline only and carry **no
> real diagnostic performance**. The UI says so on screen. See
> [MODEL_CARD.md](MODEL_CARD.md).

## 3. Repository layout

```
tiny-malaria-net/
├── .devcontainer/devcontainer.json  # GitHub Codespace definition
├── data/
│   ├── sample/           # 400 synthetic crops for rapid dev/test
│   ├── phone_test/       # 20 synthetic raw eyepiece frames + ground truth
│   └── downloads/        # staging area for real datasets
├── src/
│   ├── segment.py            # OpenCV RBC cropper from whole-field frames
│   ├── dataset.py            # patient-level split loader + domain augmentations
│   ├── model.py              # MobileNetV3-Small backbone (single logit)
│   ├── train.py              # staged training loop (NIH -> phone)
│   ├── prepare_data.py       # dataset download / manifest builder (generic)
│   ├── download_lacuna.py     # Harvard Dataverse fetcher + RAR unpacker
│   ├── parse_lacuna.py        # Lacuna annotations -> ImageFolder crops
│   ├── quantize.py            # QAT + ONNX INT8 static quantisation
│   ├── evaluate.py           # sensitivity-first ROC + segmentation evaluation
│   └── make_sample_data.py   # synthetic mock-crop generator
├── app/
│   ├── triage.json       # localized recommendations (EN, UR, PL)
│   ├── inference.py      # end-to-end edge pipeline
│   └── app.py            # Gradio UI accepting raw phone images
├── tests/                # pytest suite
├── requirements.txt
├── Makefile              # one-command verification targets
├── README.md
├── MODEL_CARD.md
└── LICENSE               # Apache-2.0
```

## 4. How the pipeline works

### Stage 1 — OpenCV RBC segmenter (`src/segment.py`)
Dependency-light (OpenCV + NumPy only), so it stays cheap on device:
grayscale → illumination correction by background division → center-weighted
band-pass → adaptive + Otsu blended thresholding → morphological cleanup →
distance-transform peak detection + watershed (splits touching cells) →
shape/size/circularity filtering → square padded 128×128 crops.

### Stage 2 — MobileNetV3-Small INT8 classifier (`src/quantize.py`)
`MobileNetV3-Small` retrained with a single-logit head, exported to ONNX and
statically quantised to INT8. The **decision threshold is calibrated, not
assumed**: the export path picks the operating point that clears the target
sensitivity while maximising specificity, then embeds it in the model sidecar
so the app and the ONNX graph always agree.

### Stage 3 — Deterministic triage (`app/triage.json`)
Parasitemia percentage → severity band → one clinician-vetted template block,
selected from a static JSON table. No on-device LLM, so the output is
deterministic, auditable and reproducible.

| Severity band | Parasitemia | Urgency |
| --- | --- | --- |
| `negative` | 0% | none |
| `low` | 0–1% | low |
| `mild` | 1–5% | moderate |
| `moderate` | 5–10% | high |
| `severe` | ≥10% | critical |

## 5. Data sourcing and licensing

| Stage | Dataset | License | Commercial reuse of weights? |
| --- | --- | --- | --- |
| 1 — feature pre-training | NIH Malaria Dataset (27,558 crops) | CC0 / public domain | ✅ yes |
| 2 — domain adaptation | Lacuna Malaria Datasets (Makerere AI Lab, Harvard Dataverse) | CC BY 4.0 | ✅ yes (attribution) |
| 2 — segmentation tuning | Broad Institute BBBC041 | **CC BY-NC-SA 3.0** | ❌ **no — non-commercial + share-alike** |
| 2 — alternative | MP-IDB | MIT | ✅ yes |
| dev | `data/sample`, `data/phone_test` (synthetic) | no license — synthetic | exercises the pipeline only |

> **Licensing correction.** The original design note listed BBBC041 as
> *CC BY 3.0*. It is in fact **CC BY-NC-SA 3.0** — the *NonCommercial* and
> *ShareAlike* clauses make it incompatible with redistributing Apache-2.0
> trained weights. Train the distributable model on NIH + Lacuna only, and use
> BBBC041 for internal research experiments. `src/prepare_data.py --list`
> flags the offending sets, and `src/train.py` carries the corrected license
> string.

### The Lacuna dataset (verified against Harvard Dataverse)

| | |
| --- | --- |
| Persistent ID | `doi:10.7910/DVN/VEADSE` |
| Title | Lacuna Malaria Datasets |
| Author | Makerere AI Lab, Makerere University (Uganda) |
| License | **CC BY 4.0** |
| Size | ~7.2 GB across 7 files |

Published files: `Thick_Ghana.part1-3.rar`, `Thin_Images_Ghana.rar`,
`Thin_Uganda.rar`, and the Ghana/Uganda datasheet PDFs.

**Fetch and stage it** (needs an `unrar`-family binary — see the note in
`requirements.txt`):

```bash
python src/download_lacuna.py --list            # live file manifest
python src/download_lacuna.py --files Thin_Uganda.rar \
    --extract --extract-to data/lacuna
python src/parse_lacuna.py --src data/lacuna \
    --out data/processed/lacuna_crops --negatives-from-background 0.35
```

**Three things the original note got wrong**, all found by inspecting the actual
archives:

1. **The annotations are not VIA JSON/CSV.** Each archive ships
   `Labels-CSV.csv` with columns `Image_name, xmin, ymin, width, height, Class`
   in absolute pixels, and that CSV is the authoritative source. A parallel
   `Labels-YOLO/<stem>.txt` tree also exists, but the release contains **no
   `classes.txt`** and its box geometry disagrees with the CSV — so the parser
   uses the CSV unless you explicitly pass `--yolo --yolo-classes`.
2. **There are no "healthy RBC" annotations.** The classes present are
   `Parasitized cell`, `Trophozoite`, `Gametocyte`, `Artifact` and `WBC`. Healthy
   red blood cells are simply not labelled, so they cannot be read off the
   annotation file. Pass `--negatives-from-background` to sample them from
   un-annotated regions; without it the negative class collapses into "WBC and
   debris", which teaches the model that only white cells are uninfected.
3. **~47% of the CSV rows are exact duplicates.** 4,796 of 12,288 rows in
   `Thin_Uganda.rar`'s CSV repeat verbatim. The parser de-duplicates them;
   without that every crop class would have been emitted twice.

Binary mapping used by `src/parse_lacuna.py`:

| Class | Label | Rationale |
| --- | --- | --- |
| `Parasitized cell`, `Trophozoite`, `Gametocyte` | 1 (Parasitized) | parasite life-cycle stages inside an RBC |
| `Artifact`, `WBC` | 0 (Uninfected) | **hard negatives** — debris and white cells are the classic false-alarm source |
| un-annotated background | 0 (Uninfected) | healthy RBCs, opt-in via `--negatives-from-background` |

Each phone capture becomes one patient id, so crops from the same field — shared
illumination, focus and staining — never straddle train and validation.

### The split rule that matters
Splits are **always at slide / patient ID level**, never at random image level.
Crops from the same field share staining, illumination and focus artifacts, so a
random split leaks those between train and validation and inflates reported
sensitivity. `src/dataset.py:patient_level_split()` enforces this and raises if
any patient id appears on both sides.

### Fetching the real data

```bash
# register / inspect what is available and under which license
python src/prepare_data.py --list

# Lacuna (Makerere / Harvard Dataverse, CC BY 4.0) - the phone domain
python src/download_lacuna.py --files Thin_Uganda.rar \
    --extract --extract-to data/lacuna
python src/parse_lacuna.py --src data/lacuna \
    --out data/processed/lacuna_crops --negatives-from-background 0.35

# NIH: point at an already-downloaded copy of the Kaggle cell-images release
python src/prepare_data.py --dataset nih_full --local-dir ~/Downloads/cell_images

# or download an archive directly (works for any dataset)
python src/prepare_data.py --dataset mp_idb \
    --url https://example.org/mp_idb.zip --extract

# produces data/<key>/manifest.json in the schema src/dataset.py expects
```

For unlabelled frame releases, pass `--labels-file <csv|json>` mapping
`filename -> label`.

## 6. Staged training

```bash
# Stage 1 — NIH baseline (20 epochs)
python src/train.py --stage 1 --dataset nih_full --model mobilenet_v3_small \
    --batch 128 --epochs 20 --lr 1e-3 --checkpoint-dir checkpoints/stage1/

# Stage 2 — smartphone domain adaptation (15 epochs, phone augmentations)
python src/train.py --stage 2 --resume checkpoints/stage1/best.pt \
    --dataset lacuna_phone --batch 64 --epochs 15 --lr 1e-4 \
    --checkpoint-dir checkpoints/stage2/

# Stage 3 — INT8 quantisation + edge export
python src/quantize.py --weights checkpoints/stage2/best.pt \
    --target onnx_int8 --output models/tinymalaria_2.1mb_int8.onnx

# Evaluate — sensitivity first, then segmentation recall
python src/evaluate.py --checkpoint checkpoints/stage2/best.pt \
    --dataset makerere_phone --target-recall 0.98
python src/evaluate.py --mode segmentation --image-dir data/phone_test --iou 0.3
```

Stage 2 freezes the backbone and fine-tunes the classifier head plus the last
inverted-residual block, and layers on the phone-eyepiece degradations the
design calls for: chromatic aberration (RGB shift + channel shuffle), vignetting
/ shadows, motion + gaussian + defocus blur, and lens flare.

## 7. Cloud Spot training

A `g2-standard-4` Spot VM with one NVIDIA L4 runs the full pipeline in well
under an hour.

```bash
gcloud compute instances create tinymalaria-trainer \
    --zone=us-central1-a \
    --machine-type=g2-standard-4 \
    --accelerator=type=nvidia-l4,count=1 \
    --provisioning-model=SPOT \
    --instance-termination-action=STOP \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=50GB

gcloud compute ssh tinymalaria-trainer --zone=us-central1-a
git clone https://github.com/YOUR_USERNAME/tiny-malaria-net.git
cd tiny-malaria-net

# 1. Python deps, then the RAR backend rarfile shells out to
sudo apt-get update -qq && sudo apt-get install -y unrar-free
pip install -r requirements.txt

# 2. Fetch + stage the data (this is the long pole: ~7.2 GB)
#    data/ is gitignored, so the dataset must be re-fetched on the VM.
python src/download_lacuna.py --files Thin_Uganda.rar \
    --extract --extract-to data/lacuna
python src/parse_lacuna.py --src data/lacuna \
    --out data/processed/lacuna_crops --negatives-from-background 0.35
python src/prepare_data.py --dataset nih_full --local-dir /path/to/cell_images

# 3. Train
python src/train.py --stage 1 --dataset nih_full --model mobilenet_v3_small \
    --batch 128 --epochs 20 --lr 1e-3 --checkpoint-dir checkpoints/stage1/
python src/train.py --stage 2 --resume checkpoints/stage1/best.pt \
    --dataset lacuna_phone --batch 64 --epochs 15 --lr 1e-4 \
    --checkpoint-dir checkpoints/stage2/

# 4. Export + evaluate
python src/quantize.py --weights checkpoints/stage2/best.pt \
    --output models/tinymalaria_2.1mb_int8.onnx
python -m src.evaluate.py --checkpoint checkpoints/stage2/best.pt \
    --dataset lacuna_phone --target-recall 0.98 --val-fraction 0.2
```

Every epoch writes `epoch_{i}.pt`, `best.pt` and `last.pt`, so a preemption
costs at most one epoch. Sync the checkpoint directory out of the VM regularly:

```bash
gsutil -m rsync -r checkpoints/ gs://tiny-malaria-data/checkpoints/
```

### Before you spend the money

1. **Commit and push first.** Only `README.md` is tracked today; a fresh clone
   gives you an empty repository.
2. **`data/` is gitignored**, so the ~7.2 GB Lacuna release must be re-downloaded
   on the VM (step 2 above). Budget ~15 GB of the 50 GB disk for raw + extracted.
3. **`unrar` is not a pip package.** `requirements.txt` installs `rarfile`, but
   `rarfile` only *reads* RAR headers and shells out to a binary for extraction.
   Install `unrar-free` (or `unrar`, or `7z`) with apt on the VM, or
   `src/download_lacuna.py` will exit with instructions.
4. **Re-measure the quantisation cost after training.** INT8 currently halves
   specificity (0.76 -> 0.36 at equal recall; see MODEL_CARD.md). That loss is
   quantisation error, not something a bigger dataset is guaranteed to fix, and
   it is *not* fixable by adding calibration data. The one config that recovered
   accuracy costs 3.5 MB and breaks the 2.5 MB budget. Decide size-vs-accuracy
   deliberately rather than discovering it at release time.
5. **NIH is registration-gated** on the LHNCBC release; the Kaggle mirror
   (`iarunava/cell-images-for-detecting-malaria`) needs a `~/.kaggle/kaggle.json`
   token. Point `--local-dir` at wherever you stage it.

## 8. Target performance criteria

| Metric | Target | How to verify |
| --- | --- | --- |
| Model footprint | < 2.5 MB | `python src/quantize.py ...` prints PASS/FAIL |
| Sensitivity / recall | ≥ 98% | `python src/evaluate.py --target-recall 0.98` |
| CPU latency | < 45 ms / cell crop | latency reported in the `.json` sidecar |
| Phone slide segmentation | > 80% RBC detection | `python src/evaluate.py --mode segmentation` |

Check the current numbers in [MODEL_CARD.md](MODEL_CARD.md) — remember that a
checkpoint calibrated on synthetic crops will not reach the clinical targets.

## 9. Tests

```bash
make test          # or: python -m pytest tests/ -q
```

The suite covers the paths that are easy to break silently: the patient-level
split guarantees no leakage, the severity banding matches the JSON template
ranges, `render_triage` returns a fallback for an unsupported language, the
INT8 footprint stays under budget, and the classifier backends agree on the
preprocessed input.

## 10. License

Apache License 2.0. See [LICENSE](LICENSE).

The **code** is Apache-2.0. The **weights** inherit the license of the data they
were trained on, so do not ship weights derived from `bbbc041` (CC BY-NC-SA
3.0) under a commercial license. See section 5.
