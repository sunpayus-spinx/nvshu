# -*- coding: utf-8 -*-
"""部署冒烟测试：上传女书测试图 -> 校验返回含女书 Unicode（U+1Bxxx）。用法: smoke_test.py [base_url]"""
import json
import mimetypes
import os
import sys
import time
import urllib.request

base = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8000"
img = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_char.png")
B = "----nvshuSmokeTest"

data = open(img, "rb").read()
body = (
    f"--{B}\r\nContent-Disposition: form-data; name=\"file\"; "
    f"filename=\"test_char.png\"\r\nContent-Type: image/png\r\n\r\n"
).encode() + data + f"\r\n--{B}--\r\n".encode()
req = urllib.request.Request(
    base + "/rest/infer", data=body, method="POST",
    headers={"Content-Type": f"multipart/form-data; boundary={B}"},
)
t0 = time.time()
try:
    r = urllib.request.urlopen(req, timeout=300)
    res = json.loads(r.read().decode("utf-8", "replace"))
except Exception as e:
    print(f"[smoke] FAIL: {type(e).__name__}: {e}")
    sys.exit(1)
dt = time.time() - t0

texts = [it.get("text", "") for it in res.get("results", [])]
modes = [it.get("rec_mode", "") for it in res.get("results", [])]
full_u = res.get("full_unicode", "")
has_nushu = any("U+1B" in (it.get("unicode") or "") for it in res.get("results", []))
print(f"[smoke] HTTP {r.status} | {dt:.1f}s | boxes={res.get('n_boxes')} det={res.get('det_source')}")
print(f"[smoke] texts={texts} modes={modes}")
print(f"[smoke] full_unicode={full_u}")
print(f"[smoke] full_pinyin={res.get('full_pinyin')!r} full_chinese={res.get('full_chinese')!r}")

if has_nushu:
    print("[smoke] PASS —— 识别出女书 Unicode")
else:
    print("[smoke] WARN —— 未识别出女书 Unicode（U+1Bxxx）。检查 cls 模型是否加载、"
          "或换一张单字图重试；TrOCR 回退吐 ASCII 属已知问题（见部署指南第 6 节）")
