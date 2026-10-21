PY ?= python3

.PHONY: help setup build check verify verify-v6e1 lint clean sessions

help:
	@echo "setup        install the authoring toolchain (jupytext, colab CLI)"
	@echo "build        src/*.py (py:percent) -> notebooks/*.ipynb"
	@echo "check        fail if committed notebooks are stale"
	@echo "verify       run every notebook on a real Colab TPU and fail on any error"
	@echo "verify-v6e1  same on v6e-1 (32 GB) -- needs an entitlement this account lacks"
	@echo "lint         ruff over src/ and tools/"
	@echo "sessions     list live Colab runtimes (these cost compute units)"

setup:
	$(PY) -m pip install -r requirements-dev.txt
	@command -v colab >/dev/null || echo "note: 'colab' not on PATH; try: uv tool install google-colab-cli"

build:
	$(PY) tools/build_notebooks.py

check:
	$(PY) tools/build_notebooks.py --check

# Depends on check, not build: verifying a notebook that differs from its
# source tells you nothing about the source.
verify: check
	$(PY) tools/verify_on_tpu.py

# v6e-1 is rejected on this account ("Backend rejected accelerator 'V6E1'") --
# it needs a Colab tier we do not have. Kept because the source measurements in
# the parent rigs and NEXT.md were taken on v6e-1, so this is the comparison to
# run the day the entitlement exists.
verify-v6e1: check
	$(PY) tools/verify_on_tpu.py --tpu v6e1

lint:
	ruff check src tools
	ruff format --check src tools

sessions:
	colab --auth adc sessions

clean:
	rm -f notebooks/*_output.ipynb
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
