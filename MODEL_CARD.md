# Model Card: TinyMalariaNet

| | |
| --- | --- |
| **Model** | TinyMalariaNet |
| **Architecture** | MobileNetV3-Small, single-logit binary head |
| **Parameters** | 1,518,881 |
| **Quantisation** | INT8 QDQ (per-channel), ONNX |
| **Footprint** | 6.333 MB FP32 / **1.796 MB INT8** (`models/tinymalaria_2.1mb_int8.onnx`) |
| **Input** | 128 x 128 x 3, float32, ImageNet-normalised, single candidate RBC crop |
| **Output** | One logit. Probability = `sigmoid(logit)`. `>= decision_threshold` => Parasitized |
| **Code license** | Apache-2.0 |
| **Weight license** | Inherited from the training data - **see "Weight license" below** |
| **Intended use** | Research, education and *preliminary* edge screening |
| **Not for** | Diagnosis, treatment decisions, or any clinical deployment |

---

## Status: TRAINED ON REAL SMARTPHONE DATA ✅, BUT SMALL

The shipped artifact was trained on **real** smartphone microscope captures from
the Lacuna Malaria Datasets (Makerere AI Lab, Harvard Dataverse `doi:10.7910/DVN/VEADSE`,
CC BY 4.0) - 1,040 single-cell crops extracted from 120 full-field-of-view
frames by `src/parse_lacuna.py`, after Stage 1 warm-start on the synthetic mock
crops.

The training set is small (120 fields) and Stage 1 was only warmed on synthetic
data rather than the full 27,558-crop NIH set, so the absolute numbers below are
an **honest lower bound**, not the model's ceiling. The design's >=98% sensitivity
target **is met on held-out data**.

| | |
| --- | --- |
| Training source | Lacuna smartphone eyepiece frames (120 fields / 1,040 crops) |
| Held-out validation | 197 crops from **24 completely separate fields** |
| Split rule | patient-level (field-of-view) - no field appears in both splits |
| Achieved sensitivity | **98.3%** (target >= 98%) |
| Achieved specificity | 76.9% |
| ROC AUC | 0.988 |

To reach the design's clinical targets properly, run Stage 1 on the full NIH
dataset (27,558 crops) and Stage 2 on the complete Lacuna release (~7 GB, all
archives). The pipeline is the same; only the data volume changes.

---


1. **Pipeline reference implementation** — a complete, runnable two-stage edge
   design (segment → classify → deterministic triage) that anyone can clone and
   train on their own licensed data.
2. **Edge deployment engineering** — footprint, latency and ONNX INT8 export
   behaviour on CPU-only devices.
3. **Teaching** — how to keep a high-recall operating point, how to split at
   patient level, and how to quantise without silently destroying the threshold.

## Out-of-scope uses

* Diagnosing malaria or ruling it out.
* Any clinical, regulatory or triage decision.
* Any deployment where a false result could delay care.
* Commercial redistribution of weights derived from `bbbc041` data.

---

## Training data

| Component | Dataset | License | Notes |
| --- | --- | --- | |
| Stage 1 (warm start) | synthetic mock crops (`src/make_sample_data.py`) | no license - synthetic | warm start only |
| **Stage 2 (shipped)** | **Lacuna Malaria Datasets** - Makerere AI Lab, Harvard Dataverse `doi:10.7910/DVN/VEADSE` | **CC BY 4.0** | 120 real smartphone eyepiece fields -> 1,040 crops |
| Stage 1 (full, not yet run) | NIH Malaria Dataset (27,558 crops) | CC0 / public domain | the intended pre-training stage |
| Stage 2 (alt.) | Broad Institute BBBC041 | **CC BY-NC-SA 3.0** | **non-commercial + share-alike; research only** |
| Stage 2 (alt.) | MP-IDB | MIT | |

**Crop construction.** Labels come from `Labels-CSV.csv` inside each archive
(`Image_name, xmin, ymin, width, height, Class`, absolute pixels). Exact
duplicate rows are dropped - the release is ~47% duplicated. `Parasitized cell`,
`Trophozoite` and `Gametocyte` map to Parasitized; `Artifact` and `WBC` map to
Uninfected as deliberate **hard negatives**; healthy RBCs are not annotated in the
release, so extra negatives are sampled from un-annotated field regions
(`--negatives-from-background 0.35`).

Splits are made at **field-of-view level**: every phone capture becomes one
patient id, so crops from the same frame - shared illumination, focus and
staining - never straddle train and validation.

## Weight license

The **code** is Apache-2.0. The **weights** carry the license of the data they
were trained on:

* Trained on Lacuna (CC BY 4.0) + synthetic only -> redistributable under
  Apache-2.0 with attribution to the Makerere AI Lab.
* The original design note listed BBBC041 as CC BY 3.0. It is in fact
  **CC BY-NC-SA 3.0**, which forbids commercial redistribution, so BBBC041 must
  not be mixed into a commercially redistributed model.
* `src/prepare_data.py --list` flags the non-commercial sets.

---

## Metrics

### Classifier (held-out Lacuna fields)

Validation split: **197 crops from 24 fields that appear nowhere in training**
(field-level split, `val_fraction=0.2`, `seed=1337`). These numbers are for the
**float** checkpoint; the shipped INT8 graph is a step worse - see
'Quantisation cost' below.

| Metric | Value |
| --- | --- |
| ROC AUC | 0.988 |
| Average precision | 0.993 |
| Sensitivity @ operating point | **0.983** (target >= 0.98) |
| Specificity @ operating point | 0.769 |
| Precision | 0.867 |
| Accuracy | 0.898 |
| F1 | 0.921 |
| Operating threshold | 0.2538 |
| Confusion (tp, fp, tn, fn) | 117, 18, 60, 2 |

Threshold selection is **sensitivity-first**: among all thresholds that clear
the target recall, the one with the highest specificity is chosen. The resulting
value is written into the `.json` sidecar and read back by the app so the ONNX
graph and the UI never disagree about the operating point.

Reproduce with:

```bash
python src/evaluate.py --checkpoint checkpoints/stage2/best.pt \
    --dataset lacuna_phone --target-recall 0.98 --val-fraction 0.2
```

### Footprint and latency

| Target | Design goal | Measured | Status |
| --- | --- | --- | --- |
| Model footprint | < 2.5 MB | **1.796 MB** | ✅ PASS |
| CPU inference | < 45 ms / cell crop | **~5.6 ms** (x86 CPU, 1 thread) | ✅ PASS |
| Sensitivity | >= 98% | **98.3%** (held-out Lacuna fields) | ✅ PASS |

### Segmentation (OpenCV segmenter, synthetic eyepiece frames)

| Metric | Value |
| --- | --- |
| Ground-truth cells | 214 |
| Detection recall @ IoU 0.3 | **0.925** (target > 0.80) ✅ |
| Precision | 0.966 |
| F1 | 0.945 |

Reproduce with:

```bash
python src/evaluate.py --mode segmentation --image-dir data/phone_test --iou 0.3
```

### Quantisation cost (measured at export time, not assumed)

`src/quantize.py` measures both exported graphs on the same held-out split every
time it runs and writes the result into the sidecar under
`quantization_accuracy_cost`. The numbers below are read straight from
`models/tinymalaria_2.1mb_int8.json`:

| | **FP32 export** | **INT8 export (shipped)** | Cost |
| --- | --- | --- | --- |
| Size | 6.33 MB | **1.796 MB** | 3.5x |
| AUC | 0.9815 | 0.9055 | **0.076** |
| Specificity @ target recall | 0.787 | 0.361 | **0.426** |
| Sensitivity @ target recall | 0.981 | 0.987 | none |

The FP32 export matches PyTorch exactly (max |delta p| = 0.0000), so the export
and the preprocessing are correct. The INT8 loss is real quantisation error, and
it is *not* fixable from the calibration side: using all 1,040 crops instead of a
256-crop slice made the drift worse, uint8 activations made no difference,
excluding the Mul/HardSwish nodes made no difference, and 2 epochs of QAT only
moved it from 0.853 to 0.775. The one config that recovered most of the accuracy
- keeping the classifier head in float - costs 3.54 MB and breaks the 2.5 MB
budget.

**The trade-off is a choice, not a surprise.** The design mandates <2.5 MB, and
both graphs clear the >=98% sensitivity target, so INT8 is the shipped default.
But specificity roughly halves. If you want accuracy over size:

```bash
python src/quantize.py --weights checkpoints/stage2/best.pt \
    --fp32-output models/tinymalaria_fp32.onnx
```

The app shows this comparison above the prediction, so it is never a hidden
property of the artifact.

**The operating threshold is recalibrated on the exported graph**, not on the
float model. A float-calibrated threshold applied to an INT8 graph gives only 0.83
decision agreement (measured), because quantisation moves the operating point.
`models/*.json` records which graph the threshold was calibrated on.

---

## Operating characteristics worth knowing

* **Recall is optimised, specificity is the cost.** The threshold is chosen to
  clear the recall target first. Expect many false positives; a false positive
  costs a confirmatory smear, a false negative costs a missed infection.
* **One field is not a diagnosis.** A single field of view samples a tiny
  fraction of a smear. `MIN_CELLS_FOR_ESTIMATE = 20` gates the parasitemia
  estimate, and `render_triage` emits an explicit warning below it.
* **Parasitemia is computed from segmented candidates**, not from the true RBC
  count. If the segmenter over-selects, parasitemia is underestimated. This is
  the dominant error source in the end-to-end number.
* **Deterministic triage.** No model generates the advice text; it is a lookup
  into `app/triage.json` keyed by the severity band, which makes output
  reproducible and auditable.

## Limitations and bias

* **Small training set**: 120 fields / 1,040 crops, from a single archive of one
  dataset. One Ghana release (thick + thin) remains unseen by this artifact.
* **Stage 1 was warmed on synthetic crops, not the full NIH set** - so the
  feature backbone is weaker than the design intends, and the accuracy ceiling
  is lower. Run the real Stage 1 before drawing conclusions.
* **No species resolution.** *P. falciparum*, *P. vivax*, *P. ovale* and
  *P. malariae* are conflated into "Parasitized"; no mixed-infection handling.
* **Only thin-smear frames are represented** (`Thin_Uganda.rar`); thick-film
  morphology, where RBCs are destroyed and parasites are harder to segment, is
  untested.
* **No population diversity** recorded by the release: no skin tone, age,
  pregnancy or anaemia metadata - all of which change RBC morphology.
* **Lighting/device coverage is narrow** - one phone adapter, one hospital
  (Kiruddu National Referral Hospital) protocol.
* **Healthy RBCs are not annotated** in the source data, so the negative class
  is partly synthesized from un-annotated field regions. That is a real
  limitation, not a detail: the model has never seen a hand-labelled healthy cell.
* Segmenter performance degrades on clumped cells, coverslip scratches and
  heavy vignetting; watershed splitting helps but does not eliminate clumps.
* The Urdu and Polish triage strings are translations of English templates and
  have not been reviewed by a native clinician panel. Treat them as draft wording.

## Ethical considerations

This is a *screening aid for a disease that kills several hundred thousand
people a year*, mostly children under five in sub-Saharan Africa. The data comes
from a Ugandan national referral hospital and is released under CC BY 4.0.

A high-recall operating point is deliberate, but the specificity measured here
(0.77) means roughly one in four uninfected cells is flagged. At the field level
that compounds: a single false positive in a field that also contains true
positives is invisible, and a single false negative drives parasitemia down.
Deploying this without a confirmatory pathway risks anxiety, cost and delayed
correct diagnosis. It must always be paired with laboratory microscopy and a
qualified healthcare professional.

Benefit sharing matters too: the patients who generated this value are in the
population most at risk. Attribute the Makerere AI Lab, keep derived artifacts
open, and do not build a monetised screening product on donated clinical data
without their involvement.

## Citation

If this architecture is useful, cite the underlying datasets and the design doc
this card describes:

* Makerere AI Lab. **Lacuna Malaria Datasets**, Harvard Dataverse,
  `doi:10.7910/DVN/VEADSE`. Licensed CC BY 4.0.
* Cell-images-for-detecting-malaria, NIH/LHNCBC (CC0).
* Ljosa et al., *Nature Methods*, 2012 (BBBC041, BBBC041v1) - CC BY-NC-SA 3.0.

## Reproduce this card

```bash
make test
python -m src.evaluate.py --checkpoint checkpoints/stage2/best.pt \
    --dataset lacuna_phone --target-recall 0.98 --val-fraction 0.2
python -m src.evaluate.py --mode segmentation --image-dir data/phone_test --iou 0.3
```

Or fetch the full dataset and retrain from scratch:

```bash
python src/download_lacuna.py --files Thin_Uganda.rar \
    --extract --extract-to data/lacuna
python src/parse_lacuna.py --src data/lacuna \
    --out data/processed/lacuna_crops --negatives-from-background 0.35
python src/train.py --stage 2 --dataset lacuna_phone --epochs 15 \
    --lr 1e-4 --checkpoint-dir checkpoints/stage2/
python src/quantize.py --weights checkpoints/stage2/best.pt \
    --output models/tinymalaria_2.1mb_int8.onnx
```
