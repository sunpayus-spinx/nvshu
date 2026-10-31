#!/usr/bin/env bash
# NuShuPARS Web 服务启动（Linux / 阿里云）
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export YOLO_CONFIG_DIR="$ROOT/models_cache/ultralytics"
export NVSHU_REC_MODEL_DIR="$ROOT/trocr_model_nushu"
export NVSHU_CHARCLS_CKPT="$ROOT/models/nushu_unicode_best.pt"

PY="${PY:-python3}"
if [ -x "$ROOT/venv/bin/python" ]; then PY="$ROOT/venv/bin/python"; fi

echo "[1/3] starting uvicorn on 0.0.0.0:8000 ..."
nohup "$PY" -m uvicorn server:app --host 0.0.0.0 --port 8000 > "$ROOT/server.log" 2>&1 &
echo "      pid $! (log: $ROOT/server.log)"

echo "[2/3] waiting for /health and applying fast profile ..."
"$PY" deploy/post_start_tune.py http://127.0.0.1:8000

echo "[3/3] smoke test ..."
"$PY" deploy/smoke_test.py http://127.0.0.1:8000

echo
echo "Done. Frontend: http://<服务器IP>:8000/web"
