#!/usr/bin/env bash
# 在 Linux GPU 机器上安装 CUDA 版 PyTorch（替换掉 CPU 版）
# 默认 cu121 + 阿里云镜像；服务器 CUDA 版本不同就改 --index-url 里的 cu12x
set -e
PY="${PY:-python3}"
[ -x "venv/bin/python" ] && PY="venv/bin/python"

echo "uninstalling CPU torch (if any) ..."
"$PY" -m pip uninstall -y torch torchvision || true

echo "installing CUDA torch (cu121) via aliyun mirror ..."
"$PY" -m pip install torch torchvision \
  --index-url https://mirrors.aliyun.com/pytorch-wheels/cu121/

"$PY" - <<'EOF'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
else:
    print("!! CUDA not available —— 检查 nvidia-smi / 驱动版本 / torch cu 版本是否匹配")
EOF
