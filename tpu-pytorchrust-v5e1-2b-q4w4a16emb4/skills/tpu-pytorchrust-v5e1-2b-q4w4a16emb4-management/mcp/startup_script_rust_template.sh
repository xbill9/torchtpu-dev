#!/bin/bash
# Startup script for a TPU VM serving Gemma 4 E2B with the model in Rust on TorchTPU.
#
# Rendered by server.py with str.format(): every literal brace in bash is doubled.
#
# Stages, in order. Each is printed as "STAGE: <name>" and written to the guest attribute
# torchtpu-rs/stage, which `get_rig_progress` reads (`gcloud alpha compute tpus tpu-vm
# get-guest-attributes`): a TPU VM has no serial-port command, and SSH can be blocked.
#   network -> python -> torch -> rust -> crate -> weights -> smoke -> unit -> serving
# A failure writes "ERROR: <stage>: <reason>" to the same attribute and exits.
exec > >(tee /var/log/torchtpu-rs-startup.log) 2>&1
set -euo pipefail

GA=http://metadata.google.internal/computeMetadata/v1/instance/guest-attributes/torchtpu-rs
stage() {{
  echo "STAGE: $1"
  curl -s -X PUT --data "$1 $(date -u +%H:%M:%SZ)" -H "Metadata-Flavor: Google" "$GA/stage" || true
}}
fail() {{
  echo "ERROR: $1"
  curl -s -X PUT --data "ERROR: $1" -H "Metadata-Flavor: Google" "$GA/stage" || true
  exit 1
}}
trap 'fail "line $LINENO exited $?"' ERR

echo "Starting the torchtpu_rs bootloader"
echo "Project {project_id}, zone {zone}, model {model_name}"

stage network
for i in $(seq 1 30); do
  curl -sf -o /dev/null https://pypi.org/simple/ && break
  [ "$i" -eq 30 ] && fail "network: pypi.org unreachable"
  sleep 5
done

# --- {python}: the newest Python torch_tpu 0.2.0 ships wheels for (3.11 to 3.14). --------
# No venv (repo standard): it is its own interpreter beside the image's python3. -dev carries
# Python.h, which the ATen shim includes; -venv carries ensurepip.
stage python
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
if ! apt-get install -y {python} {python}-dev {python}-venv; then
  apt-get install -y software-properties-common
  add-apt-repository -y ppa:deadsnakes/ppa
  apt-get update -y
  apt-get install -y {python} {python}-dev {python}-venv
fi
apt-get install -y build-essential pkg-config curl
g++ --version | head -1
{python} -m ensurepip --upgrade
PIP="{python} -m pip install --break-system-packages --no-input"

# --- PyTorch + TorchTPU, both from public PyPI. The crate builds against whichever torch
# this installs (its ATen shim compiles against the installed headers); nothing pins it.
stage torch
$PIP {pip_spec} --extra-index-url https://download.pytorch.org/whl/cpu \
  || fail "torch: pip install failed"
{python} -c "import torch, torch_tpu; print('torch', torch.__version__)"

# --- Rust: the crate's rust-toolchain.toml names the channel ({rust_toolchain}). ----------
stage rust
export HOME=/root CARGO_HOME=/opt/cargo RUSTUP_HOME=/opt/rustup
curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain none --no-modify-path \
  || fail "rust: rustup failed"
export PATH=/opt/cargo/bin:$PATH
rustup toolchain install {rust_toolchain} --profile minimal || fail "rust: toolchain {rust_toolchain}"

# --- The crate and the server, shipped in this script (SSH may be blocked; no scp). -------
stage crate
mkdir -p /opt/torchtpu_rs
echo "{payload_b64}" | base64 -d | tar -xz -C /opt/torchtpu_rs
# build.rs asks PYO3_PYTHON's torch for its headers and libraries; PyO3 uses the same Python.
(
  cd /opt/torchtpu_rs/rust
  rustc --version
  PYO3_PYTHON={python} cargo build --release
) || fail "crate: cargo build failed"
mkdir -p /opt/torchtpu_rs/module
# The #[pymodule] is named torchtpu_rs, so the file must be torchtpu_rs.so.
cp /opt/torchtpu_rs/rust/target/release/libtorchtpu_rs.so /opt/torchtpu_rs/module/torchtpu_rs.so
cp /opt/torchtpu_rs/rust/python/torchtpu_rs.pyi /opt/torchtpu_rs/module/

# --- Weights: the W4A16 repack, plus the int4 vocabulary tables from its -emb4 twin. -------
stage weights
export HF_HOME=/opt/hf
HF_TOKEN=""
TOKEN_JSON=$(curl -sf -H "Metadata-Flavor: Google" \
  "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" || true)
if [ -n "$TOKEN_JSON" ]; then
  AT=$(echo "$TOKEN_JSON" | {python} -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')
  HF_TOKEN=$(curl -sf -H "Authorization: Bearer $AT" \
    "https://secretmanager.googleapis.com/v1/projects/{project_id}/secrets/{hf_secret_id}/versions/latest:access" \
    | {python} -c 'import base64,json,sys; print(base64.b64decode(json.load(sys.stdin)["payload"]["data"]).decode())' \
    2>/dev/null || true)
fi
[ -n "$HF_TOKEN" ] && echo "Using the {hf_secret_id} secret for Hugging Face." || echo "No HF token; the repos are public."
export HF_TOKEN
{python} - <<'PYEOF' || fail "weights: download failed"
import os
from huggingface_hub import snapshot_download
base = snapshot_download("{base_repo}", token=os.environ.get("HF_TOKEN") or None)
emb4 = snapshot_download("{emb4_repo}", allow_patterns=["model-rest-001.safetensors"], token=os.environ.get("HF_TOKEN") or None)
open("/opt/torchtpu_rs/dirs.env", "w").write(f"MODEL_DIR={{base}}\nEMB4_DIR={{emb4}}\n")
print("base", base)
print("emb4", emb4)
PYEOF
. /opt/torchtpu_rs/dirs.env

# --- Smoke test: a Rust ATen call on a tpu tensor. ------------------------------------------
stage smoke
cat > /opt/torchtpu_rs/smoke.py <<'PYEOF'
import sys
import torch
import torch_tpu  # noqa: F401  registers the TPU kernels before Rust calls into libtorch
sys.path.insert(0, "/opt/torchtpu_rs/module")
import torchtpu_rs

x = torch.ones(8, 8, dtype=torch.bfloat16, device="tpu")
r = torchtpu_rs.matmul_sum(x)
print("torchtpu_rs", torchtpu_rs.__version__, "matmul_sum", r.item(), "on", r.device)
assert r.item() == 512.0 and r.device.type == "tpu"
print("TPU smoke OK")
PYEOF
{python} /opt/torchtpu_rs/smoke.py || fail "smoke: Rust call on a tpu tensor failed"
echo "TPU environment ready."

if [ -z "{serve}" ]; then
  stage ready-no-serve
  exit 0
fi

# --- The service. -------------------------------------------------------------------------
stage unit
cat > /etc/systemd/system/torchtpu-rs.service <<UNIT
[Unit]
Description=Gemma 4 E2B in Rust on TorchTPU (torchtpu_rs), OpenAI-compatible on :{serve_port}
After=network-online.target

[Service]
Environment=HF_HOME=/opt/hf
Environment=TORCH_TPU_DEFER_AND_FUSE={defer_and_fuse}
Environment=PYTHONUNBUFFERED=1
ExecStart=$(command -v {python}) /opt/torchtpu_rs/rust_openai_server.py \
  --model-dir $MODEL_DIR --emb4-dir $EMB4_DIR --module-dir /opt/torchtpu_rs/module \
  --served-name {model_name} --context {serve_context} --read-every {read_every} --port {serve_port}
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --now torchtpu-rs.service
echo "TPU serving started."

# Load plus the first decode compile happen inside the unit; report when /health answers.
for i in $(seq 1 120); do
  if curl -sf -o /dev/null "http://127.0.0.1:{serve_port}/health"; then
    stage serving
    exit 0
  fi
  sleep 15
done
fail "serving: /health did not answer within 30 min (journalctl -u torchtpu-rs)"
