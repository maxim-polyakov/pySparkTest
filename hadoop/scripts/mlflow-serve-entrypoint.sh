#!/bin/bash
set -euo pipefail
# Keep LF line endings: Linux bash reads CRLF as "pipefail\r" and exits.

MODEL_NAME="${MLFLOW_MODEL_NAME:-spells-classifier}"
TRACKING_URI="${MLFLOW_TRACKING_URI:-http://mlflow:5000}"
PORT="${MLFLOW_SERVE_PORT:-5001}"
MAX_WAIT_SEC="${MLFLOW_SERVE_WAIT_SEC:-300}"
# Production,Staging или только Production (для alpaca-causal-lm)
SERVE_STAGES="${MLFLOW_SERVE_STAGE:-Production,Staging}"

export MLFLOW_TRACKING_URI="$TRACKING_URI"

if [ -n "${MLFLOW_SERVE_PIP:-}" ]; then
  echo "Installing serve dependencies: $MLFLOW_SERVE_PIP"
  if [ -n "${MLFLOW_SERVE_PIP_EXTRA_INDEX:-}" ]; then
    pip3 install --no-cache-dir --extra-index-url "$MLFLOW_SERVE_PIP_EXTRA_INDEX" \
      $MLFLOW_SERVE_PIP
  else
    pip3 install --no-cache-dir $MLFLOW_SERVE_PIP
  fi
fi

echo "MLflow tracking: $TRACKING_URI"
echo "Waiting for tracking server..."
python3 <<'PY'
import os
import time
import urllib.request

uri = os.environ["MLFLOW_TRACKING_URI"].rstrip("/") + "/health"
for _ in range(60):
    try:
        with urllib.request.urlopen(uri, timeout=3) as resp:
            if resp.status == 200:
                break
    except OSError:
        time.sleep(2)
else:
    raise SystemExit("MLflow tracking server not reachable")
PY

echo "Resolving model ${MODEL_NAME} (stages: ${SERVE_STAGES})..."
MODEL_URI=$(python3 <<'PY'
import os
import sys
from mlflow.tracking import MlflowClient

name = os.environ["MLFLOW_MODEL_NAME"]
stages = [s.strip() for s in os.environ.get("MLFLOW_SERVE_STAGE", "Production,Staging").split(",") if s.strip()]
client = MlflowClient()
for stage in stages:
    versions = client.get_latest_versions(name, stages=[stage])
    if versions:
        print(f"models:/{name}/{stage}")
        sys.exit(0)
print(f"No version for {name} in stages {stages}", file=sys.stderr)
sys.exit(1)
PY
)

LOCAL_DIR="${ALPACA_LOCAL_MODEL_DIR:-}"
export MODEL_URI

# Быстрый старт: локальные веса (Registry часто без MLmodel + долгая загрузка checkpoint-*)
if [ -n "$LOCAL_DIR" ] && [ -f "$LOCAL_DIR/config.json" ]; then
  if [ "${ALPACA_SERVE_PREFER_LOCAL:-1}" != "0" ]; then
    echo "Using local HF weights at $LOCAL_DIR (ALPACA_SERVE_PREFER_LOCAL=0 → попробовать Registry)"
    export ALPACA_LOCAL_MODEL_DIR="$LOCAL_DIR"
    exec python3 /scripts/alpaca-local-serve.py
  fi
fi

echo "Checking Registry artifact for MLmodel..."
if python3 <<'PY'
import os
import sys
from pathlib import Path

import mlflow

uri = os.environ["MODEL_URI"]
path = mlflow.artifacts.download_artifacts(artifact_uri=uri)
if Path(path, "MLmodel").is_file():
    print(f"Registry OK: {path}")
    sys.exit(0)
print(f"No MLmodel in {path}", file=sys.stderr)
sys.exit(1)
PY
then
  echo "Starting MLflow model server: $MODEL_URI on port $PORT"
  exec mlflow models serve \
    -m "$MODEL_URI" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --env-manager local \
    --no-conda
fi

if [ -n "$LOCAL_DIR" ] && [ -f "$LOCAL_DIR/config.json" ]; then
  echo "Fallback: local HF weights at $LOCAL_DIR (re-register for MLflow serve)"
  export ALPACA_LOCAL_MODEL_DIR="$LOCAL_DIR"
  exec python3 /scripts/alpaca-local-serve.py
fi

echo "ERROR: Registry model has no MLmodel and no local fallback." >&2
echo "In Jupyter: register_alpaca_from_local(OUTPUT_DIR, target_stage='Production')" >&2
exit 1
