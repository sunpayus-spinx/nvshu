# -*- coding: utf-8 -*-
"""等待 /health 就绪 -> 应用快速识别 profile（TTA off, beam 8）-> 打印关键状态。用法: post_start_tune.py [base_url]"""
import json
import sys
import time
import urllib.request

base = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8000"

for i in range(60):
    try:
        r = urllib.request.urlopen(base + "/health", timeout=4)
        h = json.loads(r.read().decode())
        print(f"[health] up after {i*2}s")
        for k in ("device", "cuda", "rec_model_loaded", "rec_model_source",
                  "cls_model_loaded", "cls_model_source", "phonetic_lexicon_size"):
            print(f"  {k}: {h.get(k)}")
        if not h.get("cls_model_loaded"):
            print("  !! cls_model_loaded=false —— 检查 NVSHU_CHARCLS_CKPT 是否指向 models/nushu_unicode_best.pt")
        if not h.get("rec_model_loaded"):
            print("  !! rec_model_loaded=false —— 检查 NVSHU_REC_MODEL_DIR 是否指向 trocr_model_nushu/")
        break
    except Exception:
        time.sleep(2)
else:
    print("[health] NOT READY after 120s —— 看 server.log / 控制台报错")
    sys.exit(1)

try:
    r = urllib.request.urlopen(base + "/rest/rec_profile", timeout=10)
    p = json.loads(r.read().decode())
    print("[profile] fast baseline applied:",
          {k: p.get(k) for k in ("rec_tta_enabled", "trocr_beam_size", "rec_topk")})
except Exception as e:
    print("[profile] apply failed:", e)
