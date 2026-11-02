#!/usr/bin/env bash
# Install the local demo pieces for the Rust TorchTPU rig:
#   - demo/shim.py as a systemd user service (torchtpu-rs-shim) on http://127.0.0.1:8015
#   - demo/mcp_demo.py registered with Claude Code at user scope as torchtpu-rs-demo
#
#   demo/install.sh [--project ID] [--port N]   install or refresh (idempotent)
#   demo/install.sh --uninstall                 remove both
#
# --project is the GCP project holding the TPU queued resource; default: $GOOGLE_CLOUD_PROJECT,
# else the active gcloud project. Uses the python3 on PATH as an absolute path (no venv).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT=torchtpu-rs-shim
MCP_NAME=torchtpu-rs-demo
CONF_DIR="$HOME/.config/torchtpu-rs"
UNIT_DIR="$HOME/.config/systemd/user"
PROJECT="${GOOGLE_CLOUD_PROJECT:-}"
PORT=8015

while [ $# -gt 0 ]; do
  case "$1" in
    --project) PROJECT="$2"; shift ;;
    --port) PORT="$2"; shift ;;
    --uninstall)
      systemctl --user disable --now "$UNIT" 2>/dev/null || true
      rm -f "$UNIT_DIR/$UNIT.service" "$CONF_DIR/shim.env"
      systemctl --user daemon-reload
      claude mcp remove --scope user "$MCP_NAME" >/dev/null 2>&1 || true
      echo "removed $UNIT and the $MCP_NAME MCP server"
      exit 0 ;;
    -h|--help) sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

PY="$(python3 -c 'import sys; print(sys.executable)')"
"$PY" -c 'import fastapi, uvicorn, httpx, mcp' 2>/dev/null || {
  echo "missing packages: $PY -m pip install fastapi uvicorn httpx 'mcp>=2,<3'" >&2; exit 1; }
GCLOUD="$(command -v gcloud)" || { echo "gcloud not on PATH" >&2; exit 1; }
[ -n "$PROJECT" ] || PROJECT="$(gcloud config get-value project 2>/dev/null)"
[ -n "$PROJECT" ] || { echo "no project: pass --project ID" >&2; exit 1; }

mkdir -p "$CONF_DIR" "$UNIT_DIR"
cat > "$CONF_DIR/shim.env" <<EOF
# demo/shim.py settings. The queued resource name, zone and port come from tpu.env in the repo.
TORCHTPU_RS_PROJECT=$PROJECT
TORCHTPU_RS_SHIM_PORT=$PORT
# Set to pin the upstream instead of looking it up with gcloud, e.g. http://1.2.3.4:8000
#TORCHTPU_RS_UPSTREAM=
EOF

cat > "$UNIT_DIR/$UNIT.service" <<EOF
[Unit]
Description=Local OpenAI-compatible shim for Gemma 4 E2B in Rust on TorchTPU (127.0.0.1:$PORT)

[Service]
EnvironmentFile=$CONF_DIR/shim.env
Environment=PATH=$(dirname "$GCLOUD"):/usr/local/bin:/usr/bin:/bin
Environment=PYTHONUNBUFFERED=1
ExecStart=$PY $HERE/shim.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF
systemctl --user daemon-reload
systemctl --user enable "$UNIT" >/dev/null 2>&1
systemctl --user restart "$UNIT"
echo "shim: systemd user service $UNIT on http://127.0.0.1:$PORT (project $PROJECT)"

if command -v claude >/dev/null; then
  claude mcp remove --scope user "$MCP_NAME" >/dev/null 2>&1 || true
  claude mcp add --scope user "$MCP_NAME" -e "TORCHTPU_RS_SHIM_URL=http://127.0.0.1:$PORT" -- "$PY" "$HERE/mcp_demo.py" >/dev/null
  echo "mcp: $MCP_NAME registered at user scope (restart Claude Code to load it)"
else
  echo "claude CLI not found; register by hand: claude mcp add --scope user $MCP_NAME -- $PY $HERE/mcp_demo.py"
fi

for _ in $(seq 20); do
  curl -sf -m 3 "http://127.0.0.1:$PORT/shim/status" >/dev/null && break
  sleep 1
done
curl -s -m 90 "http://127.0.0.1:$PORT/shim/status" && echo
