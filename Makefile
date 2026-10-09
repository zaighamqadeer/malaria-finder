# TinyMalariaNet - one-command verification targets.
#
#   make smoke   # offline end-to-end dry run (synthetic data, no downloads)
#   make test    # pytest suite
#   make lint    # ruff / compileall syntax check
#   make seg     # segmentation recall against the ground-truth boxes
#   make help    # list targets

PYTHON ?= python
.DEFAULT_GOAL := help

.PHONY: help smoke test lint seg app data clean

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

smoke: ## Offline end-to-end dry run on synthetic mock crops
	$(PYTHON) src/train.py --dev-mode --sample 200 --epochs 1 \
		--checkpoint-dir checkpoints/stage1/
	$(PYTHON) src/segment.py --test-image data/phone_test/sample_slide_000.jpg \
		--output-dir data/segmented
	$(PYTHON) src/quantize.py --weights checkpoints/stage1/best.pt \
		--output models/tinymalaria_2.1mb_int8.onnx
	$(PYTHON) -m app.inference --image data/phone_test/sample_slide_000.jpg \
		--model models/tinymalaria_2.1mb_int8.onnx --lang en
	@echo "[make] smoke run complete"

test: ## Run the pytest suite
	$(PYTHON) -m pytest tests/ -q

lint: ## Byte-compile every module (catches syntax errors fast)
	$(PYTHON) -m compileall -q src app tests
	@echo "[make] syntax OK"

seg: ## Segmentation recall / precision vs the synthetic ground truth
	$(PYTHON) src/evaluate.py --mode segmentation \
		--image-dir data/phone_test --iou 0.3 \
		--seg-target-recall 0.80

app: ## Launch the Gradio UI
	$(PYTHON) app/app.py --port 7860

data: ## Generate the synthetic mock data used by the smoke run
	$(PYTHON) src/make_sample_data.py --mode crops --total 400
	$(PYTHON) src/make_sample_data.py --mode fov --frames 20

clean: ## Remove local build / test artifacts
	rm -rf build .pytest_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	@echo "[make] cleaned"
