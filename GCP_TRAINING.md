# GCP Spot Training Runbook — TinyMalariaNet

Step-by-step for training the real model on a `g2-standard-4` Spot VM.

This assumes a **private** repository (`zaighamqadeer/malaria-finder`), so a bare
`git clone` will fail. Section 3 covers getting the code onto the VM.

---

## 0. Before you spend anything

Do these locally first — they take minutes and catch the failures that would
otherwise cost you a Spot hour.

```bash
# 1. The pipeline must be green
make test                                   # expect: 118 passed
python src/evaluate.py --mode segmentation --image-dir data/phone_test --iou 0.3
                                            # expect: 0.925 recall, PASS

# 2. Confirm the repo is actually pushed
git log --oneline -1 origin/main
git status --porcelain                       # expect: empty

# 3. Make sure you have a way to authenticate GitHub from the VM (section 3)
```

**Ballpark cost.** A `g2-standard-4` (1× L4) Spot VM is roughly **$0.40–0.70/hour**
depending on zone and current spot price. The training itself is short (Stage 1
~20 min, Stage 2 ~20 min on this size of data); the **7.2 GB Lacuna download is
network-bound and dominates the wall-clock time**. Budget $1–3 for a first run,
plus a few dollars of egress/storage. Not the "$1 total" the design note claims —
that assumed the data was already staged.

---

## 1. Stage the data to Cloud Storage *once* (do this locally, not on the VM)

Because the VM is Spot and can be preempted, do **not** download 7.2 GB onto the
VM's boot disk. Download locally, push to GCS, then pull onto the VM. If the VM
dies you re-pull instead of re-downloading.

```bash
# Locally — Pull the Lacuna release (needs unrar; see section 5)
python src/download_lacuna.py --files Thin_Uganda.rar --extract --extract-to data/lacuna

# Push to a bucket (create it once)
gsutil mb -l us-central1 gs://YOUR_BUCKET
gsutil -m rsync -r data/lacuna gs://YOUR_BUCKET/lacuna
```

### NIH (Stage 1 pre-training)

The NIH Malaria Dataset is registration-gated on LHNCBC; the Kaggle mirror
(`iarunava/cell-images-for-detecting-malaria`) needs a `~/.kaggle/kaggle.json`.
Easiest path: download it locally once, then push up.

```bash
# Locally, after you have the cell_images/ folder
gsutil -m rsync -r ~/Downloads/cell_images gs://YOUR_BUCKET/nih/cell_images
```

> If you cannot get NIH before the VM run, run Stage 2 from the Lacuna-warmed
> checkpoint you already have and re-do Stage 1 properly later. Do **not** skip
> to Stage 1 — it is what makes Stage 2's frozen backbone useful.

---

## 2. Provision the Spot VM

```bash
gcloud compute instances create tinymalaria-trainer \
    --zone=us-central1-a \
    --machine-type=g2-standard-4 \
    --accelerator=type=nvidia-l4,count=1 \
    --provisioning-model=SPOT \
    --instance-termination-action=STOP \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=60GB \
    --metadata=install-nvidia-driver=True
```

Notes:
- `--instance-termination-action=STOP` (not `DELETE`) means a preemption can be
  resumed with your boot disk intact.
- **60 GB**, not 50: raw (7.2 GB) + extracted + crops + checkpoints adds up.
- Check the spot price before committing:
  `gcloud compute machine-types describe g2-standard-4 --zone us-central1-a`

---

## 3. Get the private repo onto the VM

Pick **one**. Option A (PAT) is the least friction.

### Option A — Personal Access Token (quickest)

Create a token at <https://github.com/settings/tokens> with the **`repo`** scope.
Then on the VM:

```bash
gcloud compute ssh tinymalaria-trainer --zone=us-central1-a

export GH_TOKEN=github_pat_xxxxxxxxxxxx     # keep this off the command log
git clone https://${GH_TOKEN}@github.com/zaighamqadeer/malaria-finder.git
cd malaria-finder
```

> Put the token in a variable first, as shown. Pasting it directly into the
> clone URL puts it in your shell history and in `~/.bash_history`.

### Option B — SSH key (reusable, no token in history)

```bash
ssh-keygen -t ed25519 -f ~/.ssh/gcp_tinymalaria -N ""
cat ~/.ssh/gcp_tinymalaria.pub   # add this to GitHub -> Settings -> SSH keys

# On the VM:
eval "$(ssh-agent -s)" && ssh-add ~/.ssh/gcp_tinymalaria
git clone git@github.com:zaighamqadeer/malaria-finder.git
cd malaria-finder
```

### Option C — GitHub CLI

```bash
gh auth login --hostname github.com --git-protocol https
gh repo clone zaighamqadeer/malaria-finder
cd malaria-finder
```

> **Git LFS:** this repo has a `pre-push` hook that requires `git-lfs`. It blocks
> *pushes* only, but install it anyway so you can push checkpoints back:
> `sudo apt-get update -qq && sudo apt-get install -y git-lfs`

---

## 4. Install dependencies — including the RAR backend

```bash
sudo apt-get update -qq && sudo apt-get install -y unrar-free git-lfs

pip install -r requirements.txt
```

**`unrar-free` is not optional.** `requirements.txt` installs `rarfile`, but
`rarfile` only *reads* RAR headers — it shells out to an external binary for
extraction. Without one, `src/download_lacuna.py --extract` exits with a
message telling you this, and you lose a Spot hour finding out.

Verify the GPU is visible before training:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 5. Stage the datasets from GCS

```bash
# Lacuna (~7.2 GB, the long pole)
gsutil -m rsync -r gs://YOUR_BUCKET/lacuna data/lacuna

python src/parse_lacuna.py --src data/lacuna \
    --out data/processed/lacuna_crops --negatives-from-background 0.35
# Expect: ~12k boxes kept, ~47% duplicate rows dropped, Parasitized+Uninfected/
# written, plus manifest.json

# NIH
gsutil -m rsync -r gs://YOUR_BUCKET/nih/cell_images data/nih/cell_images
python src/prepare_data.py --dataset nih_full --local-dir data/nih/cell_images
```

Sanity-check both manifests before training (uses the registry, so no guessed
paths):

```bash
python - <<'PY'
import json
from pathlib import Path
from src.train import DATASETS, REPO_ROOT

for key in ("lacuna_phone", "nih_full"):
    path = Path(REPO_ROOT) / DATASETS[key].manifest
    if not path.exists():
        print(key, "MISSING", path)
        continue
    m = json.loads(path.read_text())
    print(f"{key:14s} {m['num_images']:6d} images  {m['num_patients']:4d} fields  "
          f"labels={m['label_counts']}  license={m['license']}")
PY
```

Expected: both print real counts with two non-empty label buckets. If
`num_patients` is 1, the split is degenerate and `make test` will tell you why.

> `--negatives-from-background 0.35` is deliberate. The Lacuna release does not
> annotate healthy red blood cells, so without this the negative class becomes
> "WBCs and debris" and the model learns to flag anything big and round.

---

## 6. Train

```bash
# Stage 1 — NIH baseline
python src/train.py --stage 1 --dataset nih_full --model mobilenet_v3_small \
    --batch 128 --epochs 20 --lr 1e-3 \
    --checkpoint-dir checkpoints/stage1/ --num-workers 8

# Stage 2 — smartphone domain adaptation on the real phone frames
python src/train.py --stage 2 --resume checkpoints/stage1/best.pt \
    --dataset lacuna_phone --batch 64 --epochs 15 --lr 1e-4 \
    --checkpoint-dir checkpoints/stage2/ --num-workers 8
```

Every epoch writes `epoch_{i}.pt`, `best.pt` and `last.pt`. `best.pt` is selected
by a recall-first score, not accuracy — that is intentional. Sync out
continuously (see section 7) so a preemption costs at most one epoch.

---

## 7. Protect against Spot preemption

`--instance-termination-action=STOP` means the VM stops rather than deletes, but
**the boot disk is only preserved if it's the same VM** — a new instance would
lose it. Push checkpoints to GCS as you go.

Run this in a second SSH session while training:

```bash
gsutil -m rsync -r checkpoints/ gs://YOUR_BUCKET/checkpoints/
```

Or, better, make it periodic:

```bash
while true; do gsutil -m rsync -r checkpoints/ gs://YOUR_BUCKET/checkpoints/; sleep 120; done
```

To resume after a preemption, re-run the same `--stage`/`--resume` command with
`--checkpoint-dir` pointing at the synced directory — training picks up from the
`best.pt` you resume from.

---

## 8. Export and evaluate

```bash
# INT8 export + the measured FP32-vs-INT8 accuracy cost
python src/quantize.py --weights checkpoints/stage2/best.pt \
    --output models/tinymalaria_2.1mb_int8.onnx
# Expect: "footprint target <2.5 MB -> PASS" and a quantisation-cost line

# Sensitivity-first evaluation on held-out Lacuna fields
python -m src.evaluate.py --checkpoint checkpoints/stage2/best.pt \
    --dataset lacuna_phone --target-recall 0.98 --val-fraction 0.2

# Segmentation recall
python src/evaluate.py --mode segmentation --image-dir data/phone_test --iou 0.3
```

### What to check after training

| Metric | Target | Where |
| --- | --- | --- |
| INT8 footprint | < 2.5 MB | quantize output |
| Sensitivity | ≥ 98% | evaluate output `recall_target_met` |
| CPU latency | < 45 ms/crop | `.json` sidecar `latency_ms_cpu` |
| Seg. detection recall | > 80% | segmentation eval |

**Re-read `quantization_accuracy_cost` in the sidecar.** On the current model
INT8 halves specificity (0.79 → 0.36). If it is still bad after real NIH
pre-training, decide deliberately:

```bash
# Accuracy-first alternative, ~3.5x the size
python src/quantize.py --weights checkpoints/stage2/best.pt \
    --fp32-output models/tinymalaria_fp32.onnx
```

---

## 9. Pull the results back and stop the VM

```bash
# Locally
gsutil -m rsync -r gs://YOUR_BUCKET/checkpoints ./checkpoints
gsutil cp gs://YOUR_BUCKET/models/tinymalaria_2.1mb_int8.* ./models/

# On the VM (or from your shell)
gcloud compute instances stop tinymalaria-trainer --zone=us-central1-a
gcloud compute instances delete tinymalaria-trainer --zone=us-central1-a
```

**Stop the VM as soon as training finishes.** A stopped Spot VM still costs for
its 60 GB boot disk (~$2.5/month), which is trivial — but a *running* one is not.
Delete it once you have everything in GCS.

Push checkpoints back to the private repo (the LFS hook needs `git-lfs`):

```bash
sudo apt-get install -y git-lfs
git add -A && git commit -m "Stage 1+2 trained on NIH + Lacuna" && git push
```

> `data/`, `checkpoints/` and `models/` are gitignored. Push only what you want
> in the repo — the model artifacts belong in GCS or the Hugging Face Hub, not
> git. Use `make publish HF_NAMESPACE=your-username` for that.

---

## 10. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `git clone` asks for a password, fails | Private repo — use section 3. PAT needs `repo` scope. |
| `No module named 'onnxscript'` | `pip install -r requirements.txt` not run, or interrupted. |
| `RarCannotExec: Cannot find working tool` | Missing RAR backend: `sudo apt-get install -y unrar-free` |
| `No RAR extractor found` | Same as above. Run `which unrar unrar-free 7z bsdtar` — at least one must exist. |
| `Archive not found: ...` (from `src/download_lacuna.py`) | You passed `--extract-only` or an `--extract-to` dir with no archive in it. Fetch it first with `--files`, or point `--local-dir` at where it already lives. |
| `Failed to extract ... make sure every part is present` | Multi-part archive and a part is missing, or you passed `.part2.rar` instead of `.part1.rar`. |
| `ModuleNotFoundError: No module named 'app.inference'` | You are running a *staged* flat Space copy. In-repo this should not happen. |
| `"dataset root not found"` | Manifest missing. Run the `prepare_data.py` / `parse_lacuna.py` step first. |
| Training OOMs on the L4 | Drop `--batch` to 64 (Stage 1) or 32 (Stage 2). |
| `CUDA out of memory` at epoch 0 | Wrong image — must be a GPU DLVM; `nvidia-smi` to check. |
| Val split is empty / train=0 | Patient-level split degenerated. `make test` catches this. |
| Threshold in the sidecar is ~0.5 on a good model | The `.json` sidecar was not co-located with the `.onnx`. It must sit next to it. |
| Preempted mid-epoch | `gsutil rsync` from section 7 was not running. Re-run with `--resume`. |
| `test_extract_without_backend_gives_instructions` fails | You have a RAR backend installed, so the "no backend" branch is skipped and rarfile raises `FileNotFoundError` on the dummy path. Fixed by monkeypatching `find_unrar`; update with `git pull`. |

---

## One-page condensed version

```bash
# LOCAL
make test
python src/download_lacuna.py --files Thin_Uganda.rar --extract --extract-to data/lacuna
gsutil mb -l us-central1 gs://YOUR_BUCKET
gsutil -m rsync -r data/lacuna gs://YOUR_BUCKET/lacuna
gsutil -m rsync -r ~/Downloads/cell_images gs://YOUR_BUCKET/nih/cell_images

# VM
gcloud compute instances create tinymalaria-trainer --zone=us-central1-a \
    --machine-type=g2-standard-4 --accelerator=type=nvidia-l4,count=1 \
    --provisioning-model=SPOT --instance-termination-action=STOP \
    --image-family=pytorch-latest-gpu --image-project=deeplearning-platform-release \
    --boot-disk-size=60GB
gcloud compute ssh tinymalaria-trainer --zone=us-central1-a
export GH_TOKEN=github_pat_xxx
git clone https://${GH_TOKEN}@github.com/zaighamqadeer/malaria-finder.git && cd malaria-finder
sudo apt-get install -y unrar-free git-lfs
pip install -r requirements.txt

gsutil -m rsync -r gs://YOUR_BUCKET/lacuna data/lacuna
python src/parse_lacuna.py --src data/lacuna --out data/processed/lacuna_crops --negatives-from-background 0.35
gsutil -m rsync -r gs://YOUR_BUCKET/nih/cell_images data/nih/cell_images
python src/prepare_data.py --dataset nih_full --local-dir data/nih/cell_images

python src/train.py --stage 1 --dataset nih_full --epochs 20 --lr 1e-3 --checkpoint-dir checkpoints/stage1/ --num-workers 8
python src/train.py --stage 2 --resume checkpoints/stage1/best.pt --dataset lacuna_phone --epochs 15 --lr 1e-4 --checkpoint-dir checkpoints/stage2/ --num-workers 8

# second SSH session: while true; do gsutil -m rsync -r checkpoints/ gs://YOUR_BUCKET/checkpoints/; sleep 120; done

python src/quantize.py --weights checkpoints/stage2/best.pt --output models/tinymalaria_2.1mb_int8.onnx
python -m src.evaluate.py --checkpoint checkpoints/stage2/best.pt --dataset lacuna_phone --target-recall 0.98

gcloud compute instances stop tinymalaria-trainer --zone=us-central1-a
```
