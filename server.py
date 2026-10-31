"""
女书识别系统 - FastAPI 后端服务
================================

本模块实现了基于 FastAPI 的女书识别 API 服务，集成了 YOLOv8 检测模型与 TrOCR 识别模型。

主要功能：
1. 模型动态加载：支持从本地路径或 Hugging Face 下载并加载 TrOCR 识别模型。
2. 图像推理：接收上传图片，进行区域检测并输出对应的女书字符。
3. 系统监控：提供健康检查接口，实时反馈 GPU/CPU 状态及模型加载情况。

实现原理：
1. 检测阶段：使用 Ultralytics YOLOv8n 进行目标检测，定位图片中的字符区域。
2. 切割阶段：根据检测出的 Bounding Box 对原图进行裁剪。
3. 识别阶段：利用 Microsoft TrOCR (Transformer-based Optical Character Recognition) 模型对裁剪后的区域进行文字转换。

版本记录：
- v1.0.0 (2026-03-14): 初始版本，集成 YOLOv8 与 TrOCR。
"""

from fastapi import FastAPI, UploadFile, File, Body, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pathlib import Path
from PIL import Image, ImageOps
import io, json, os, csv, base64
import numpy as np
import cv2
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import yaml
from nvshu_paths import project_root, resolve_path
from nvshu_phonetic import NvshuPhoneticLexicon

# 初始化 FastAPI 应用
app = FastAPI(title="Nvshu OCR API", description="女书检测与识别 API 服务")

# CORS：允许本地静态页（如 docs/index.html 翻译站）跨域调用本服务。
# 本服务仅面向本机/可信网络使用（无鉴权），allow_origins=["*"] 不新增实质风险；
# 对公网部署请在反向代理层收紧来源或加鉴权。
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
API_VERSION = "2026-07-14-phonetic-candidate-pool"

# Phonetic translation layer: Nvshu glyph -> dialect reading -> Hanzi candidate pool.
_phonetic_lexicon = NvshuPhoneticLexicon.from_project_root(project_root())

# 自动检测可用计算设备（优先使用 CUDA）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载女书字符 -> 汉字/发音对照表
_nushu_to_chinese: dict = {}
_nushu_to_pinyin: dict = {}
_hanzi_to_nushu: dict = {}


def _normalize_pinyin(py: str) -> str:
    """发音标准化：小写、移除空白与声调数字，保留 a-z。"""
    if not py:
        return ""
    py = py.strip().lower().replace(" ", "")
    py = "".join(ch for ch in py if ("a" <= ch <= "z"))
    return py


def _load_nushu_map():
    csv_path = resolve_path(
        "githubdata/nvshudata/unicode_nushu-master/data.csv",
        root=project_root()
    )
    if not csv_path.exists():
        print(f"[server] 女书对照表不存在：{csv_path}")
        return
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            char = row.get("女书字符", "").strip()
            hanzi = row.get("对应汉字", "").strip()
            py = _normalize_pinyin(row.get("江永方言代表发音", ""))
            if char and hanzi:
                _nushu_to_chinese[char] = hanzi
                # 反向映射：汉字首选 -> 女书字符（用于前端输出被汉字污染时的纠偏）
                for hz in hanzi:
                    if hz and hz not in _hanzi_to_nushu:
                        _hanzi_to_nushu[hz] = char
            if char and py:
                _nushu_to_pinyin[char] = py
    print(
        f"[server] 女书对照表加载完成，共 {len(_nushu_to_chinese)} 个字符，"
        f"发音映射 {len(_nushu_to_pinyin)} 条"
    )


_load_nushu_map()

def _char_to_unicode(ch: str) -> str:
    if not ch:
        return ""
    return f"U+{ord(ch):04X}"


def text_to_unicode(text: str) -> str:
    return " ".join(_char_to_unicode(ch) for ch in (text or ""))


def _looks_like_nushu(ch: str) -> bool:
    if not ch:
        return False
    cp = ord(ch)
    # Unicode Nushu block: U+1B170..U+1B2FF
    return 0x1B170 <= cp <= 0x1B2FF


def normalize_to_nushu_text(text: str) -> str:
    """
    尽量将输出文本归一为女书字符串：
    - 若本身是女书字符，原样保留
    - 若被误识别为汉字，尝试通过汉字->女书反向表纠偏
    - 其他字符原样保留
    """
    out = []
    for ch in (text or ""):
        if _looks_like_nushu(ch):
            out.append(ch)
            continue
        out.append(_hanzi_to_nushu.get(ch, ch))
    return "".join(out)


def nushu_to_chinese(text: str) -> str:
    """将女书字符串逐字翻译为汉字（每字取第一个对应汉字，无对应则保留原字符）"""
    result = []
    for ch in text:
        hanzi = _nushu_to_chinese.get(ch, ch)
        # 对照表中一个女书字可对应多个同音汉字，取第一个作为代表
        result.append(hanzi[0] if hanzi else ch)
    return "".join(result)


def nushu_to_pinyin(text: str) -> str:
    """将女书字符串逐字映射为拼音（无映射则返回 ?）。"""
    return " ".join(_nushu_to_pinyin.get(ch, "?") for ch in (text or ""))


def nushu_to_chinese_all(text: str) -> list:
    """返回每个女书字符的候选信息（汉字、拼音、Unicode）。"""
    result = []
    for ch in text:
        hanzi = _nushu_to_chinese.get(ch, ch)
        result.append({
            "nushu": ch,
            "unicode": _char_to_unicode(ch),
            "pinyin": _nushu_to_pinyin.get(ch, ""),
            "candidates": list(hanzi) if hanzi else [ch]
        })
    return result

# 全局变量，缓存模型实例以避免重复加载
rec_model = None        # TrOCR 识别模型
rec_processor = None    # TrOCR 图像处理器
cls_model = None        # 单字分类模型
cls_char2id = None      # 分类映射
cls_id2char = None
cls_arch = ""
cls_image_size = 128
cls_source = ""
cls_transform = None
# 识别策略参数（回退到纯 TrOCR 稳定基线）
REC_STRATEGY_LOCKED = True
REC_OPTIMAL_PROFILE = "charcls_only_no_fallback"

rec_topk = 8
rec_conf_threshold = 0.80
rec_force_trocr = False
rec_fusion_enabled = False

# 拼音中间层默认关闭（先保证 Unicode 主识别）
pinyin_bridge_enabled = False
pinyin_weight_cls = 0.0
pinyin_weight_trocr = 1.0
pinyin_weight_py = 0.0

# 稳健识别参数（TrOCR 开启TTA）
rec_tta_enabled = True
trocr_beam_size = 12
rec_profile = REC_OPTIMAL_PROFILE
det_force_source = "auto"  # auto|yolo|proj_fallback|cc_fallback
det_model = None        # YOLO 检测模型
det_error = ""          # 检测模型加载错误信息
det_model_source = ""   # 当前加载的检测模型路径
det_model_override = "" # 手动指定检测模型路径（优先）
rec_model_source = ""   # 当前加载的识别模型来源路径

def load_rec_model(model_path_or_id: str):
    """
    动态加载 TrOCR 识别模型。

    意图：根据提供的路径或 ID 初始化 Transformer 识别管道。
    输入：model_path_or_id (str) - 本地文件夹路径或 Hugging Face 模型 ID。
    输出：无（更新全局变量 rec_model, rec_processor, rec_model_source）。
    原理：使用 transformers 库的 from_pretrained 方法加载权重和配置。
    潜在风险：
        - 模型文件损坏或不完整导致加载失败。
        - 显存（VRAM）不足，特别是在加载大型模型时。
    """
    global rec_model, rec_processor, rec_model_source
    if not model_path_or_id:
        model_path_or_id = "microsoft/trocr-small-handwritten"
    
    p = Path(model_path_or_id)
    try:
        if p.exists():
            # 从本地磁盘加载
            rec_processor = TrOCRProcessor.from_pretrained(str(p))
            rec_model = VisionEncoderDecoderModel.from_pretrained(str(p))
        else:
            # 从网络（Hugging Face Hub）下载加载
            rec_processor = TrOCRProcessor.from_pretrained(model_path_or_id)
            rec_model = VisionEncoderDecoderModel.from_pretrained(model_path_or_id)
        
        rec_model.to(device)
        rec_model_source = model_path_or_id
    except Exception as e:
        print(f"Failed to load recognition model from {model_path_or_id}: {e}")
        raise

def _build_cls_model(arch: str, num_classes: int) -> nn.Module:
    arch = (arch or "").lower()
    if arch == "convnext_tiny" and hasattr(models, "convnext_tiny"):
        m = models.convnext_tiny(weights=None)
        in_features = m.classifier[2].in_features
        m.classifier[2] = nn.Linear(in_features, num_classes)
        return m

    m = models.resnet18(weights=None)
    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, num_classes)
    return m


def _default_cls_ckpt() -> str:
    env = os.environ.get("NVSHU_CHARCLS_CKPT", "").strip()
    if env and Path(env).exists():
        return env

    root = resolve_path("models/char_cls", root=project_root())
    if root.exists():
        cands = sorted(root.glob("run_*/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if cands:
            return str(cands[0])
    # 发布仓库兜底：models/nushu_unicode_best.pt（release 布局）
    fallback = resolve_path("models/nushu_unicode_best.pt", root=project_root())
    if fallback.exists():
        return str(fallback)
    return ""


def load_cls_model(ckpt_path: str = "") -> bool:
    global cls_model, cls_char2id, cls_id2char, cls_arch, cls_image_size, cls_source, cls_transform

    target = ckpt_path.strip() or _default_cls_ckpt()
    if not target:
        return False

    p = Path(target)
    if not p.is_absolute():
        p = resolve_path(str(p), root=project_root())
    if not p.exists():
        print(f"[server] 分类模型不存在：{p}")
        return False

    try:
        ckpt = torch.load(str(p), map_location=device)
        char2id = ckpt.get("char2id", {})
        if not char2id:
            print("[server] 分类模型缺少 char2id，跳过加载")
            return False

        arch = str(ckpt.get("arch", "convnext_tiny"))
        image_size = int(ckpt.get("image_size", 128))
        model = _build_cls_model(arch=arch, num_classes=len(char2id))
        model.load_state_dict(ckpt["model"])
        model.to(device)
        model.eval()

        cls_model = model
        cls_char2id = char2id
        cls_id2char = {int(v): k for k, v in char2id.items()}
        cls_arch = arch
        cls_image_size = image_size
        cls_source = str(p)
        cls_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        print(f"[server] 分类模型加载完成：{p}")
        return True
    except Exception as e:
        print(f"[server] 分类模型加载失败：{e}")
        return False


def _classify_single_char_topk(crop: Image.Image, top_k: int = 5) -> dict:
    if cls_model is None or cls_transform is None or cls_id2char is None:
        return {"best": "", "confidence": 0.0, "topk": []}

    x = cls_transform(crop).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = cls_model(x)
        probs = F.softmax(logits, dim=1)
        kk = min(max(1, top_k), probs.shape[1])
        score, idx = torch.topk(probs, k=kk, dim=1)

    top = []
    for s, i in zip(score[0].tolist(), idx[0].tolist()):
        ch = cls_id2char.get(int(i), "")
        if ch:
            top.append({"text": ch, "score": float(s)})

    if not top:
        return {"best": "", "confidence": 0.0, "topk": []}

    return {
        "best": top[0]["text"],
        "confidence": float(top[0]["score"]),
        "topk": top,
    }


def _fuse_char_predictions(cls_rec: dict, trocr_rec: dict, top_k: int = 5) -> dict:
    """
    融合 char_cls + trocr + 拼音桥接：
    - char_cls 概率（强监督）
    - trocr 相对概率（生成式补充）
    - pinyin 中间层：当两路对同拼音字存在分歧时，做温和重排
    """
    merged = {}

    cls_top = cls_rec.get("topk", []) if cls_rec else []
    tr_top = trocr_rec.get("topk", []) if trocr_rec else []

    # TrOCR 的 sequences_scores 是log分，先转概率分布
    tr_scores = np.array([float(x.get("score", 0.0)) for x in tr_top], dtype=np.float32)
    if tr_scores.size > 0:
        tr_probs = np.exp(tr_scores - tr_scores.max())
        tr_probs = tr_probs / max(1e-9, tr_probs.sum())
    else:
        tr_probs = np.array([], dtype=np.float32)

    # 主融合（默认更偏 char_cls，适合 3070Ti 推理稳定性）
    for i, c in enumerate(cls_top):
        ch = (c.get("text") or "").strip()
        if not ch:
            continue
        s = float(c.get("score", 0.0))
        merged[ch] = merged.get(ch, 0.0) + pinyin_weight_cls * s

    for i, c in enumerate(tr_top):
        ch = (c.get("text") or "").strip()
        if not ch:
            continue
        s = float(tr_probs[i]) if i < len(tr_probs) else 0.0
        merged[ch] = merged.get(ch, 0.0) + pinyin_weight_trocr * s

    # 拼音桥接层：把两路候选映射到“拼音桶”，再回流到字符分数
    if pinyin_bridge_enabled and merged:
        py_bucket = {}

        for c in cls_top:
            ch = (c.get("text") or "").strip()
            if not ch:
                continue
            py = _nushu_to_pinyin.get(ch, "")
            if not py:
                continue
            py_bucket[py] = py_bucket.get(py, 0.0) + float(c.get("score", 0.0))

        for i, c in enumerate(tr_top):
            ch = (c.get("text") or "").strip()
            if not ch:
                continue
            py = _nushu_to_pinyin.get(ch, "")
            if not py:
                continue
            tp = float(tr_probs[i]) if i < len(tr_probs) else 0.0
            py_bucket[py] = py_bucket.get(py, 0.0) + tp

        if py_bucket:
            vals = np.array(list(py_bucket.values()), dtype=np.float32)
            vals = np.exp(vals - vals.max())
            vals = vals / max(1e-9, vals.sum())
            py_norm = {k: float(v) for k, v in zip(py_bucket.keys(), vals)}

            # 将拼音概率回注到每个字符，作为温和增益
            for ch in list(merged.keys()):
                py = _nushu_to_pinyin.get(ch, "")
                if py and py in py_norm:
                    merged[ch] += pinyin_weight_py * py_norm[py]

    if not merged:
        return {"best": "", "confidence": 0.0, "topk": []}

    top = sorted(({"text": k, "score": float(v)} for k, v in merged.items()), key=lambda x: x["score"], reverse=True)[:max(1, top_k)]
    best = top[0]["text"]

    # 融合置信度：归一化后 top1 与 top2 差距
    vals = np.array([x["score"] for x in top], dtype=np.float32)
    if vals.size == 1:
        conf = 1.0
    else:
        conf = float(max(0.0, min(1.0, vals[0] - vals[1] + 0.5)))

    return {"best": best, "confidence": conf, "topk": top}


def _recognize_processed_crop(crop_for_cls: Image.Image, crop_for_trocr: Image.Image) -> tuple[dict, str]:
    """在单个预处理版本上执行当前识别策略。"""
    # 方案1：双路融合（最优）
    if rec_fusion_enabled and (cls_model is not None) and (rec_model is not None and rec_processor is not None) and not rec_force_trocr:
        cls_rec = _classify_single_char_topk(crop_for_cls, top_k=rec_topk)
        trocr_rec = _decode_single_char_topk(crop_for_trocr, top_k=rec_topk)
        rec = _fuse_char_predictions(cls_rec=cls_rec, trocr_rec=trocr_rec, top_k=rec_topk)
        used_mode = "fusion"

        if rec.get("confidence", 0.0) < rec_conf_threshold and trocr_rec.get("best", ""):
            rec = trocr_rec
            used_mode = "trocr_fallback"
        return rec, used_mode

    # 方案2：仅分类（并支持低置信回退 trocr）
    if cls_model is not None and not rec_force_trocr:
        rec = _classify_single_char_topk(crop_for_cls, top_k=rec_topk)
        used_mode = "char_cls"

        if rec_model is not None and rec_processor is not None and rec.get("confidence", 0.0) < rec_conf_threshold:
            trocr_rec = _decode_single_char_topk(crop_for_trocr, top_k=rec_topk)
            if trocr_rec.get("best", ""):
                rec = trocr_rec
                used_mode = "trocr_fallback"
        return rec, used_mode

    # 方案3：强制 trocr
    rec = _decode_single_char_topk(crop_for_trocr, top_k=rec_topk)
    return rec, "trocr"


def _recognize_char_robust(crop: Image.Image) -> tuple[dict, str]:
    """
    单字稳健识别：
    - char_cls 路径尽量贴合训练分布（原始灰度风格，不做强增强）
    - trocr 路径保留预处理与TTA
    """
    base_cls = crop.convert("RGB")
    base_trocr = _preprocess_crop_for_rec(crop)

    variants = [("base", base_cls, base_trocr)]
    if rec_tta_enabled:
        variants.extend([
            (
                "autocontrast",
                ImageOps.autocontrast(base_cls.convert("L")).convert("RGB"),
                ImageOps.autocontrast(base_trocr.convert("L")).convert("RGB"),
            ),
            (
                "invert",
                ImageOps.invert(base_cls.convert("L")).convert("RGB"),
                ImageOps.invert(base_trocr.convert("L")).convert("RGB"),
            ),
        ])

    best_rec = {"best": "", "confidence": -1.0, "topk": []}
    best_mode = "trocr"
    for _, cls_v, tr_v in variants:
        rec, mode = _recognize_processed_crop(cls_v, tr_v)
        if rec.get("confidence", 0.0) > best_rec.get("confidence", -1.0):
            best_rec = rec
            best_mode = mode

    if best_mode == "fusion":
        best_mode = "fusion_tta"
    return best_rec, best_mode


def default_model_dir() -> str:
    """
    获取默认的识别模型目录。

    逻辑优先级：
    1. 环境变量 NVSHU_REC_MODEL_DIR。
    2. models/ 下最新的 stage2_full_finetune/best/ 目录。
    3. 配置文件 training_config.yaml 中的 output_dir。
    4. 本地 trocr_model_nushu/ 或 trocr_model/ 目录。
    """
    # 优先级1：环境变量
    env_dir = os.environ.get("NVSHU_REC_MODEL_DIR", "").strip()
    if env_dir and Path(env_dir).exists():
        return env_dir

    # 优先级2：自动寻找 models/ 下最新的 stage2_full_finetune/best/
    models_root = resolve_path("models", root=project_root())
    if models_root.exists():
        candidates = sorted(
            models_root.glob("*/stage2_full_finetune/best"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        if candidates:
            best = candidates[0]
            print(f"[server] 自动加载最新训练模型：{best}")
            return str(best)
    # 优先级3：配置文件 output_dir
    cfg_path = resolve_path("configs/training_config.yaml", root=project_root())
    if cfg_path.exists():
        try:
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            out_dir = str(cfg.get("output_dir", "")).strip()
            if out_dir:
                return str(resolve_path(out_dir, root=project_root()))
        except Exception:
            pass

    # 优先级4：本地 TrOCR 目录
    for local_name in ("trocr_model_nushu", "trocr_model"):
        local = resolve_path(local_name, root=project_root())
        if local.exists():
            print(f"[server] 使用本地预训练模型：{local}")
            return str(local)

    return "microsoft/trocr-small-handwritten"



def _expand_box(box: list[float], w: int, h: int, ratio: float = 0.10) -> tuple[int, int, int, int]:
    """框外扩，避免笔画被截断。"""
    x1, y1, x2, y2 = box
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    ex = bw * ratio
    ey = bh * ratio
    nx1 = int(max(0, np.floor(x1 - ex)))
    ny1 = int(max(0, np.floor(y1 - ey)))
    nx2 = int(min(w, np.ceil(x2 + ex)))
    ny2 = int(min(h, np.ceil(y2 + ey)))
    return nx1, ny1, nx2, ny2


def _preprocess_crop_for_rec(crop: Image.Image) -> Image.Image:
    """
    识别前预处理：保留灰度细节 + 轻增强 + 正方形留白。
    相比强二值化，更适合手写细线条。
    """
    arr = np.array(crop.convert("L"))
    arr = cv2.GaussianBlur(arr, (3, 3), 0)
    arr = cv2.equalizeHist(arr)

    h, w = arr.shape
    side = max(h, w)
    canvas = np.full((side, side), 255, dtype=np.uint8)
    y0 = (side - h) // 2
    x0 = (side - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = arr

    out = Image.fromarray(canvas)
    out = ImageOps.expand(out, border=10, fill=255)
    return out.convert("RGB")


def _decode_single_char_topk(crop: Image.Image, top_k: int = 5) -> dict:
    """
    单字识别 top-k：返回 best/topk/confidence，避免仅用 top1。
    """
    pv = rec_processor(images=crop, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        out = rec_model.generate(
            pv,
            max_length=6,
            num_beams=max(4, trocr_beam_size, top_k),
            num_return_sequences=top_k,
            output_scores=True,
            return_dict_in_generate=True,
        )

    seqs = out.sequences
    texts = rec_processor.batch_decode(seqs, skip_special_tokens=True)
    score_tensor = out.sequences_scores.detach().cpu().numpy()

    cand = []
    for t, s in zip(texts, score_tensor):
        tt = (t or "").strip()
        if not tt:
            continue
        one = tt[0]
        cand.append({"text": one, "score": float(s)})

    # 去重，保留每个字符最高分
    best_by_char = {}
    for c in cand:
        ch = c["text"]
        if ch not in best_by_char or c["score"] > best_by_char[ch]["score"]:
            best_by_char[ch] = c

    uniq = sorted(best_by_char.values(), key=lambda x: x["score"], reverse=True)
    if not uniq:
        return {"best": "", "confidence": 0.0, "topk": []}

    top = uniq[:top_k]
    best = top[0]["text"]

    # 粗略置信度：best 与次优分差映射到 [0,1]
    s0 = top[0]["score"]
    s1 = top[1]["score"] if len(top) > 1 else s0 - 2.0
    margin = s0 - s1
    conf = float(1.0 / (1.0 + np.exp(-margin)))

    return {"best": best, "confidence": conf, "topk": top}


def _sort_boxes_reading_order(boxes: list[list[float]]) -> list[list[float]]:
    """按阅读顺序排序（先上后下，同一行内从左到右）。"""
    if not boxes:
        return boxes

    heights = [max(1.0, b[3] - b[1]) for b in boxes]
    row_tol = float(np.median(heights)) * 0.6

    def _key(b):
        y_center = (b[1] + b[3]) / 2.0
        row_id = int(y_center / max(1.0, row_tol))
        return (row_id, b[0])

    return sorted(boxes, key=_key)


def _merge_boxes(boxes: list[list[float]], merge_gap: float = 10.0) -> list[list[float]]:
    """
    将距离相近的框合并（NMS-free 贪心合并）。
    merge_gap: 两框之间允许的最大间隔像素，超过此距离则不合并。
    """
    if not boxes:
        return boxes

    changed = True
    merged = [list(b) for b in boxes]
    while changed:
        changed = False
        out = []
        used = [False] * len(merged)
        for i in range(len(merged)):
            if used[i]:
                continue
            ax1, ay1, ax2, ay2 = merged[i]
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                bx1, by1, bx2, by2 = merged[j]
                # 检查两框是否足够近（任一方向间隔 < merge_gap）
                gap_x = max(0.0, max(ax1, bx1) - min(ax2, bx2))
                gap_y = max(0.0, max(ay1, by1) - min(ay2, by2))
                if gap_x <= merge_gap and gap_y <= merge_gap:
                    ax1 = min(ax1, bx1)
                    ay1 = min(ay1, by1)
                    ax2 = max(ax2, bx2)
                    ay2 = max(ay2, by2)
                    used[j] = True
                    changed = True
            out.append([ax1, ay1, ax2, ay2])
        merged = out

    return merged


def _detect_boxes_by_components(img: Image.Image, expected_n: int = 0) -> list[list[float]]:
    """连通域切分：两阶段策略，先稳健合并，再在欠检时细粒度重试。"""
    arr = np.array(img.convert("L"))
    _, bw = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = bw.shape
    img_area = h * w

    def _extract(mask: np.ndarray, min_area_ratio: float, min_wh: int, merge_scale: float) -> list[list[float]]:
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out = []
        for i in range(1, num_labels):
            x, y, ww, hh, area = stats[i]
            if area < max(16, img_area * min_area_ratio):
                continue
            if ww < min_wh or hh < min_wh:
                continue
            if ww > w * 0.98 and hh > h * 0.98:
                continue
            pad = 2
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w, x + ww + pad)
            y2 = min(h, y + hh + pad)
            out.append([float(x1), float(y1), float(x2), float(y2)])

        if out:
            gap = max(2.0, min(w, h) * merge_scale)
            out = _merge_boxes(out, merge_gap=gap)
        if len(out) > 80:
            out = sorted(out, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)[:80]
        return _sort_boxes_reading_order(out)

    # 第一阶段：适度膨胀，增强连通性
    char_size_hint = min(h, w) // max(1, 12)
    dk = max(2, min(7, char_size_hint // 4))
    kernel = np.ones((dk, dk), np.uint8)
    bw_merged = cv2.dilate(bw, kernel, iterations=2)
    boxes = _extract(bw_merged, min_area_ratio=0.00045, min_wh=8, merge_scale=0.02)

    # 第二阶段（关键）：若仍欠检，改用轻处理，防止整行被并成一个大框
    if len(boxes) <= 1:
        bw_light = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
        boxes_light = _extract(bw_light, min_area_ratio=0.00020, min_wh=5, merge_scale=0.008)
        if len(boxes_light) > len(boxes):
            boxes = boxes_light

    return _sort_boxes_reading_order(boxes)

def _detect_boxes_by_projection(img: Image.Image) -> list[list[float]]:
    """
    垂直投影切分：专门用于横向文字行图（宽远大于高）。
    原理：对每列像素求前景像素数量，找出列投影的谷底（空白间隔），以此切分字符。
    """
    arr = np.array(img.convert("L"))
    h, w = arr.shape

    # OTSU 二值化，前景为白（1）
    _, bw = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 垂直投影：每列前景像素数
    col_proj = bw.sum(axis=0)  # shape: (w,)

    # 动态阈值：投影值低于 max*0.05 视为空白列
    threshold = col_proj.max() * 0.05

    # 找出字符区段（连续非空列）
    in_char = False
    segs = []  # [(x_start, x_end), ...]
    x_start = 0
    for x in range(w):
        if col_proj[x] > threshold:
            if not in_char:
                x_start = x
                in_char = True
        else:
            if in_char:
                segs.append((x_start, x - 1))
                in_char = False
    if in_char:
        segs.append((x_start, w - 1))

    if not segs:
        return []

    # 过滤太窄的段（噪点）
    min_w = max(4, w // 40)
    segs = [(x1, x2) for x1, x2 in segs if x2 - x1 >= min_w]

    # 如果段数太多，合并相邻间距很小的段
    if len(segs) > 20:
        min_gap = max(2, w // 100)
        merged = [list(segs[0])]
        for x1, x2 in segs[1:]:
            if x1 - merged[-1][1] <= min_gap:
                merged[-1][1] = x2
            else:
                merged.append([x1, x2])
        segs = [(s[0], s[1]) for s in merged]

    # 自适应宽段拆分：避免两个字被粘成一个段
    if len(segs) >= 2:
        widths = np.array([x2 - x1 + 1 for x1, x2 in segs], dtype=np.float32)
        med_w = float(np.median(widths))
        split_segs = []
        for x1, x2 in segs:
            ww = x2 - x1 + 1
            # 过宽段按列投影谷值进行拆分
            if ww > max(22, med_w * 1.65):
                segment = col_proj[x1:x2 + 1].astype(np.float32)
                k = int(round(ww / max(1.0, med_w)))
                k = max(2, min(3, k))

                # 在候选切点窗口中找谷值（低投影列）
                cuts = []
                for t in range(1, k):
                    center = int(round(t * ww / k))
                    l = max(1, center - max(3, ww // 14))
                    r = min(ww - 2, center + max(3, ww // 14))
                    local = segment[l:r + 1]
                    # 只在谷值足够低时才切（不低于 max 的 60% 不切）
                    min_val = float(local.min())
                    max_val = float(segment.max())
                    if min_val > max_val * 0.62:
                        continue
                    cut = l + int(np.argmin(local))
                    cuts.append(cut)

                prev = 0
                for c in cuts + [ww - 1]:
                    sx1 = x1 + prev
                    sx2 = x1 + c
                    if sx2 - sx1 + 1 >= max(4, int(med_w * 0.45)):
                        split_segs.append((sx1, sx2))
                    prev = c + 1
            else:
                split_segs.append((x1, x2))
        segs = split_segs

    # 后处理：把过窄碎片（< 中位宽的 0.4）合并到邻近段，防止一字多段
    if len(segs) >= 2:
        widths2 = np.array([x2 - x1 + 1 for x1, x2 in segs], dtype=np.float32)
        med_w2 = float(np.median(widths2))
        min_accept = max(4.0, med_w2 * 0.40)
        merged2 = [list(segs[0])]
        for x1, x2 in segs[1:]:
            ww = x2 - x1 + 1
            if ww < min_accept:
                # 太窄：合并到前一段
                merged2[-1][1] = x2
            elif merged2[-1][1] - merged2[-1][0] + 1 < min_accept:
                # 前一段太窄：合并到它
                merged2[-1][1] = x2
            else:
                merged2.append([x1, x2])
        segs = [(s[0], s[1]) for s in merged2]

    # 找行的上下边界（水平投影）
    row_proj = bw.sum(axis=1)
    row_threshold = row_proj.max() * 0.05
    y_starts = [y for y in range(h) if row_proj[y] > row_threshold]
    y1_global = max(0, (y_starts[0] - 2) if y_starts else 0)
    y2_global = min(h, (y_starts[-1] + 2) if y_starts else h)

    boxes = []
    pad = 2
    for xs, xe in segs:
        x1 = max(0, xs - pad)
        x2 = min(w, xe + pad)
        y1 = y1_global
        y2 = y2_global
        xc = (x1 + x2) / 2.0 / w
        yc = (y1 + y2) / 2.0 / h
        bw_ = (x2 - x1) / w
        bh_ = (y2 - y1) / h
        if bw_ > 0.01 and bh_ > 0.01:
            boxes.append([float(x1), float(y1), float(x2), float(y2)])

    return _sort_boxes_reading_order(boxes)


def _sanitize_boxes(boxes: list[list[float]], w: int, h: int) -> list[list[float]]:
    """裁剪并过滤无效框，避免异常框影响打分。"""
    out = []
    for b in boxes:
        if len(b) != 4:
            continue
        x1, y1, x2, y2 = map(float, b)
        x1 = max(0.0, min(float(w), x1))
        y1 = max(0.0, min(float(h), y1))
        x2 = max(0.0, min(float(w), x2))
        y2 = max(0.0, min(float(h), y2))
        if x2 - x1 < 3 or y2 - y1 < 3:
            continue
        out.append([x1, y1, x2, y2])
    return _sort_boxes_reading_order(out)


def _score_detection(boxes: list[list[float]], w: int, h: int, aspect: float) -> float:
    """
    对候选检测结果打分：覆盖率高、框数合理、重叠少、尺寸分布稳定得分更高。
    """
    n = len(boxes)
    if n == 0:
        return -1e9

    areas, ws, hs = [], [], []
    for x1, y1, x2, y2 in boxes:
        ww = max(1.0, x2 - x1)
        hh = max(1.0, y2 - y1)
        ws.append(ww)
        hs.append(hh)
        areas.append(ww * hh)

    img_area = max(1.0, float(w * h))
    coverage = min(1.0, sum(areas) / (img_area * 0.95))

    # 估计合理字符数：横向图随宽高比增加，字符数上限更高
    expected = int(round(max(1.0, min(24.0, aspect * 2.2))))
    tol = max(2.0, expected * 0.7)
    count_score = max(0.0, 1.0 - abs(n - expected) / tol)

    # 框重叠惩罚（IoU > 0.35 视为明显重叠）
    overlap_pen = 0.0
    pair_cnt = 0
    for i in range(n):
        ax1, ay1, ax2, ay2 = boxes[i]
        a_area = max(1.0, (ax2 - ax1) * (ay2 - ay1))
        for j in range(i + 1, n):
            bx1, by1, bx2, by2 = boxes[j]
            b_area = max(1.0, (bx2 - bx1) * (by2 - by1))
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            iou = inter / max(1.0, a_area + b_area - inter)
            if iou > 0.35:
                overlap_pen += iou
            pair_cnt += 1
    overlap_pen = overlap_pen / max(1, pair_cnt)

    w_cv = float(np.std(ws) / max(1.0, np.mean(ws)))
    h_cv = float(np.std(hs) / max(1.0, np.mean(hs)))
    size_outlier = min(1.5, (w_cv + h_cv) * 0.5)

    # 单大框惩罚：避免把整行误当作一个字符
    largest_ratio = max(areas) / img_area if areas else 0.0
    single_big_pen = 0.0
    if n <= 2 and largest_ratio > 0.28:
        single_big_pen = min(1.2, (largest_ratio - 0.28) * 3.0)

    return 2.0 * coverage + 1.2 * count_score - 0.9 * overlap_pen - 0.6 * size_outlier - single_big_pen


def _select_best_boxes(
    img: Image.Image,
    yolo_boxes: list[list[float]],
    proj_boxes: list[list[float]],
    cc_boxes: list[list[float]],
) -> tuple[list[list[float]], str]:
    """三路候选（YOLO/投影/连通域）并行打分，自动选择最优结果。"""
    w, h = img.size
    aspect = w / max(1, h)

    candidates = {
        "yolo": _sanitize_boxes(yolo_boxes, w, h),
        "proj_fallback": _sanitize_boxes(proj_boxes, w, h),
        "cc_fallback": _sanitize_boxes(cc_boxes, w, h),
    }

    scores = {k: _score_detection(v, w, h, aspect) for k, v in candidates.items()}

    # 先按综合分选最优
    best_source = max(scores, key=lambda k: scores[k])
    best_boxes = candidates[best_source]
    best_score = scores[best_source]

    # 召回优先兜底：YOLO 框明显偏少时，优先采用更多框的候选（避免漏字）
    yolo_n = len(candidates["yolo"])
    alt_by_n = max(("proj_fallback", "cc_fallback"), key=lambda k: len(candidates[k]))
    alt_n = len(candidates[alt_by_n])
    alt_score = scores[alt_by_n]

    force_recall = (
        (yolo_n <= 4 and alt_n >= yolo_n + 2)
        or (yolo_n <= 6 and alt_n >= int(max(7, yolo_n * 1.8)))
    ) and alt_n <= 24

    if best_source == "yolo" and force_recall:
        best_source = alt_by_n
        best_boxes = candidates[alt_by_n]
    elif best_source == "yolo" and yolo_n <= 4 and alt_n >= yolo_n + 2:
        # 温和兜底：允许小幅降分换取更高召回
        if alt_score >= best_score - 0.35:
            best_source = alt_by_n
            best_boxes = candidates[alt_by_n]

    return best_boxes, best_source

# 服务启动时初始化识别模型（分类优先，TrOCR 作为回退）
_cls_loaded = load_cls_model("")
if not _cls_loaded:
    try:
        load_rec_model(default_model_dir())
    except Exception as _e:
        print(f"[startup] 识别模型不可用（TrOCR 权重未随包分发）：{_e}")
        print("[startup] 服务继续启动；字符分类可用时走分类链，否则视觉识别降级")
else:
    # 保持 TrOCR 也可用，方便回退；加载失败不阻断服务
    try:
        load_rec_model(default_model_dir())
    except Exception:
        pass

def try_load_det_model():
    """
    延迟加载 YOLO 检测模型。

    背景：YOLO 模型初始化较快，但为节省初始显存，采用单例模式延迟加载。
    实现原理：使用 ultralytics 库加载用户指定或自动探测的 best.pt。
    """
    global det_model, det_error, det_model_source
    if det_model is not None:
        return det_model
    try:
        # 设置 YOLO 缓存目录，避免在系统默认位置产生杂乱文件
        cfg_dir = os.environ.get("YOLO_CONFIG_DIR", "").strip()
        if not cfg_dir:
            cfg_dir = str(resolve_path("models_cache/ultralytics", root=project_root()))
            os.environ["YOLO_CONFIG_DIR"] = cfg_dir
        Path(cfg_dir).mkdir(parents=True, exist_ok=True)

        from ultralytics import YOLO

        # 1) 手动指定优先
        yolo_path = ""
        if det_model_override:
            p = Path(det_model_override)
            if not p.is_absolute():
                p = resolve_path(str(p), root=project_root())
            if p.exists():
                yolo_path = str(p)
            else:
                print(f"[server] 指定检测模型不存在：{p}")

        # 2) 自动搜索项目内 best.pt
        if not yolo_path:
            _yolo_search_dirs = [
                resolve_path("models/yolo_synth", root=project_root()),
                resolve_path("models/yolo", root=project_root()),
            ]
            _yolo_candidates = []
            for _search_dir in _yolo_search_dirs:
                if _search_dir.exists():
                    _yolo_candidates.extend(
                        sorted(
                            _search_dir.glob("run*/weights/best.pt"),
                            key=lambda p: p.stat().st_mtime,
                            reverse=True,
                        )
                    )
            if _yolo_candidates:
                yolo_path = str(_yolo_candidates[0])

        # 3) 兜底通用模型
        if not yolo_path:
            yolo_path = "yolov8n.pt"
            print("[server] 未找到专用 YOLO 模型，使用通用 yolov8n.pt")
        else:
            print(f"[server] 加载 YOLO 模型：{yolo_path}")

        det_model = YOLO(yolo_path)
        det_model_source = yolo_path
        det_error = ""
        return det_model
    except Exception as e:
        det_model = None
        det_model_source = ""
        det_error = str(e)
        return None

# 挂载静态资源（前端页面）
web_dir = resolve_path("web", root=project_root())
if web_dir.exists():
    app.mount("/web", StaticFiles(directory=str(web_dir), html=True), name="web")

@app.post("/rest/infer")
async def infer(file: UploadFile = File(...), debug_crops: bool = Query(False)):
    """
    图像推理主接口。

    意图：实现从检测到识别的端到端流水线。
    输入：
      - file (UploadFile): 用户上传的图像文件
      - debug_crops (bool): 为 true 时，返回每个检测框裁剪图（base64 PNG），用于排查切字/识别问题
    输出：dict - 包含检测框坐标、识别结果、可选的诊断裁剪图。
    边界条件：
        - 若未检测到目标，则将整张图视为一个识别区域。
        - 若图片格式不支持或损坏，PIL 将抛出异常。
    """
    bytes_data = await file.read()
    img = Image.open(io.BytesIO(bytes_data)).convert("RGB")
    
    yolo_boxes = []
    det_source = "yolo"
    dm = try_load_det_model()
    if dm is not None:
        # 检测任务以召回优先
        results = dm.predict(img, conf=0.15, iou=0.5, max_det=128)
        for r in results:
            for b in r.boxes.xyxy.cpu().numpy():
                yolo_boxes.append(b.tolist())

    # 三路候选并行生成：YOLO + 投影 + 连通域
    proj_boxes = _detect_boxes_by_projection(img)
    cc_boxes = _detect_boxes_by_components(img)

    boxes, det_source = _select_best_boxes(
        img=img,
        yolo_boxes=yolo_boxes,
        proj_boxes=proj_boxes,
        cc_boxes=cc_boxes,
    )

    forced_candidates = {
        "yolo": _sanitize_boxes(yolo_boxes, *img.size),
        "proj_fallback": _sanitize_boxes(proj_boxes, *img.size),
        "cc_fallback": _sanitize_boxes(cc_boxes, *img.size),
    }
    if det_force_source in forced_candidates and forced_candidates[det_force_source]:
        boxes = forced_candidates[det_force_source]
        det_source = det_force_source
    print(
        f"[infer] cand_counts yolo={len(yolo_boxes)} proj={len(proj_boxes)} cc={len(cc_boxes)}"
    )

    # 三路都失败时退化为整图
    if not boxes:
        w, h = img.size
        boxes = [[0, 0, w, h]]
        det_source = "fullimage"

    # 统一按阅读顺序
    boxes = _sort_boxes_reading_order(boxes)
    print(f"[infer] det_source={det_source} boxes={len(boxes)}")
    
    texts = []
    img_w, img_h = img.size
    fallback_count = 0

    for idx, b in enumerate(boxes):
        # 框外扩：过大外扩会引入邻字干扰，保持小比例
        expand_ratio = 0.03 if det_source in {"proj_fallback", "cc_fallback"} else 0.05
        ex1, ey1, ex2, ey2 = _expand_box(b, img_w, img_h, ratio=expand_ratio)
        crop = img.crop((ex1, ey1, ex2, ey2))

        rec, used_mode = _recognize_char_robust(crop)
        if "fallback" in used_mode:
            fallback_count += 1

        best_text_raw = rec["best"]
        best_text = normalize_to_nushu_text(best_text_raw)
        conf = rec["confidence"]
        topk = rec["topk"]

        # top-k 也做同样纠偏，避免前端混出汉字字符串
        fixed_topk = []
        for tk in topk:
            t = str(tk.get("text", ""))
            fixed_topk.append({
                **tk,
                "text": normalize_to_nushu_text(t[:1])
            })

        phonetic = _phonetic_lexicon.analyze_text(best_text, topk_by_index={0: fixed_topk})
        candidate_pool = phonetic["tokens"][0]["same_reading_pool"] if phonetic["tokens"] else []

        item = {
            "index": idx,
            "box": b,
            "expand_box": [ex1, ey1, ex2, ey2],
            "text": best_text,
            "text_raw": best_text_raw,
            "unicode": text_to_unicode(best_text),
            "pinyin": nushu_to_pinyin(best_text),
            "chinese": nushu_to_chinese(best_text),
            "candidates": nushu_to_chinese_all(best_text),
            "phonetic": phonetic,
            "candidate_pool": candidate_pool,
            "confidence": conf,
            "topk": fixed_topk,
            "rec_mode": used_mode,
            "needs_review": bool(conf < rec_conf_threshold),
        }

        if debug_crops:
            buf = io.BytesIO()
            crop.save(buf, format="PNG")
            item["crop_png_base64"] = base64.b64encode(buf.getvalue()).decode("ascii")

        texts.append(item)

    if rec_force_trocr:
        final_mode = "trocr"
    elif rec_fusion_enabled and cls_model is not None and rec_model is not None and rec_processor is not None:
        final_mode = "fusion"
    elif cls_model is not None:
        final_mode = "char_cls"
    else:
        final_mode = "trocr"
    full_text = "".join(x.get("text", "") for x in texts)
    full_phonetic = _phonetic_lexicon.analyze_text(full_text)
    return {
        "results": texts,
        "det_source": det_source,
        "n_boxes": len(boxes),
        "rec_mode": final_mode,
        "rec_topk": rec_topk,
        "rec_conf_threshold": rec_conf_threshold,
        "fallback_count": fallback_count,
        "debug_crops": bool(debug_crops),
        "full_text": full_text,
        "full_unicode": text_to_unicode(full_text),
        "full_pinyin": nushu_to_pinyin(full_text),
        "full_phonetic": full_phonetic,
        "full_chinese": nushu_to_chinese(full_text),
        "api_version": API_VERSION,
    }

@app.get("/health")
def health():
    """
    健康检查接口。
    
    意图：供外部监控系统（如 Prometheus）或前端查询服务运行状态。
    返回：计算设备信息、模型加载状态及可能的错误。
    """
    return {
        "api_version": API_VERSION,
        "device": str(device),
        "cuda": torch.cuda.is_available(),
        "rec_model_loaded": rec_model is not None,
        "rec_model_source": rec_model_source,
        "cls_model_loaded": cls_model is not None,
        "cls_model_source": cls_source,
        "cls_model_arch": cls_arch,
        "rec_topk": rec_topk,
        "rec_conf_threshold": rec_conf_threshold,
        "rec_force_trocr": rec_force_trocr,
        "rec_fusion_enabled": rec_fusion_enabled,
        "rec_profile": rec_profile,
        "strategy_locked": REC_STRATEGY_LOCKED,
        "pinyin_bridge_enabled": pinyin_bridge_enabled,
        "pinyin_weight_cls": pinyin_weight_cls,
        "pinyin_weight_trocr": pinyin_weight_trocr,
        "pinyin_weight_py": pinyin_weight_py,
        "nushu_pinyin_map_size": len(_nushu_to_pinyin),
        "phonetic_lexicon_size": len(_phonetic_lexicon.by_char),
        "det_force_source": det_force_source,
        "rec_tta_enabled": rec_tta_enabled,
        "trocr_beam_size": trocr_beam_size,
        "det_model_loaded": det_model is not None,
        "det_model_source": det_model_source,
        "det_error": det_error,
    }

@app.get("/")
def root():
    """根路径重定向到接口文档"""
    return RedirectResponse(url="/docs")

@app.post("/rest/translate")
def translate(payload: dict = Body(...)):
    """
    女书字符串翻译接口。
    输入：{"text": "𛅼𛋫𛇼..."}
    输出：{text, unicode, pinyin, chinese, details}
    """
    text = str(payload.get("text", ""))
    text_nushu = normalize_to_nushu_text(text)
    return {
        "text": text_nushu,
        "text_raw": text,
        "unicode": text_to_unicode(text_nushu),
        "pinyin": nushu_to_pinyin(text_nushu),
        "chinese": nushu_to_chinese(text_nushu),
        "details": nushu_to_chinese_all(text_nushu),
        "phonetic": _phonetic_lexicon.analyze_text(text_nushu),
    }

@app.post("/rest/set_model_dir")
def set_model_dir(payload: dict = Body(...)):
    """
    热更新 TrOCR 识别模型接口。
    """
    mdir = str(payload.get("model_dir", "")).strip()
    target = mdir or default_model_dir()
    try:
        load_rec_model(target)
        return {"ok": True, "model_dir": target, "type": "trocr"}
    except Exception as e:
        return {"ok": False, "model_dir": target, "error": str(e), "type": "trocr"}


@app.post("/rest/set_charcls_ckpt")
def set_charcls_ckpt(payload: dict = Body(...)):
    """
    热更新单字分类模型接口。
    输入：{"ckpt": "models/char_cls/run_xxx/best.pt"}
    """
    ckpt = str(payload.get("ckpt", "")).strip()
    ok = load_cls_model(ckpt)
    return {
        "ok": bool(ok),
        "ckpt": ckpt or _default_cls_ckpt(),
        "cls_model_loaded": cls_model is not None,
        "cls_model_source": cls_source,
        "cls_model_arch": cls_arch,
    }


@app.post("/rest/set_det_source")
def set_det_source(payload: dict = Body(...)):
    """
    强制检测来源：auto / yolo / proj_fallback / cc_fallback。
    """
    global det_force_source
    src = str(payload.get("source", "auto")).strip()
    if src not in {"auto", "yolo", "proj_fallback", "cc_fallback"}:
        return {"ok": False, "error": "invalid source", "det_force_source": det_force_source}
    det_force_source = src
    return {"ok": True, "det_force_source": det_force_source}


@app.post("/rest/set_det_model")
def set_det_model(payload: dict = Body(...)):
    """
    手动指定并热更新 YOLO 检测模型。
    输入：{"model_path": "models/yolo/run_xxx/weights/best.pt"}
    """
    global det_model_override, det_model, det_error, det_model_source

    model_path = str(payload.get("model_path", "")).strip()
    if not model_path:
        return {"ok": False, "error": "model_path is required"}

    p = Path(model_path)
    if not p.is_absolute():
        p = resolve_path(str(p), root=project_root())
    if not p.exists():
        return {"ok": False, "error": f"model not found: {p}"}

    det_model_override = str(p)

    # 清空旧模型并重新加载
    det_model = None
    det_model_source = ""
    det_error = ""
    dm = try_load_det_model()

    return {
        "ok": dm is not None,
        "det_model_override": det_model_override,
        "det_model_loaded": dm is not None,
        "det_model_source": det_model_source,
        "det_error": det_error,
    }


def _apply_optimal_profile() -> dict:
    """应用并返回稳定基线（char_cls 单路，不回退）。"""
    global rec_profile
    global rec_topk, rec_conf_threshold, rec_force_trocr, rec_fusion_enabled
    global pinyin_bridge_enabled, pinyin_weight_cls, pinyin_weight_trocr, pinyin_weight_py
    global rec_tta_enabled, trocr_beam_size

    rec_profile = REC_OPTIMAL_PROFILE
    rec_topk = 8
    rec_conf_threshold = 0.0
    rec_force_trocr = False
    rec_fusion_enabled = False
    pinyin_bridge_enabled = False
    pinyin_weight_cls, pinyin_weight_trocr, pinyin_weight_py = 1.0, 0.0, 0.0
    rec_tta_enabled = False
    trocr_beam_size = 8

    return {
        "profile": rec_profile,
        "rec_topk": rec_topk,
        "rec_conf_threshold": rec_conf_threshold,
        "rec_force_trocr": rec_force_trocr,
        "rec_fusion_enabled": rec_fusion_enabled,
        "pinyin_bridge_enabled": pinyin_bridge_enabled,
        "pinyin_weight_cls": pinyin_weight_cls,
        "pinyin_weight_trocr": pinyin_weight_trocr,
        "pinyin_weight_py": pinyin_weight_py,
        "rec_tta_enabled": rec_tta_enabled,
        "trocr_beam_size": trocr_beam_size,
        "strategy_locked": REC_STRATEGY_LOCKED,
    }


@app.get("/rest/rec_profile")
def get_rec_profile():
    """读取当前识别配置（已固化最优）。"""
    return {"ok": True, **_apply_optimal_profile()}


@app.post("/rest/set_rec_params")
def set_rec_params(payload: dict = Body(...)):
    """
    识别参数接口：当前策略锁定为稳定基线，不接受动态修改。
    """
    if REC_STRATEGY_LOCKED:
        return {
            "ok": True,
            "locked": True,
            "message": "recognition strategy is locked to charcls_only_no_fallback",
            **_apply_optimal_profile(),
        }

    # 预留（默认不会走到）：仅当关闭锁定时可扩展动态调参
    return {
        "ok": True,
        "locked": False,
        **_apply_optimal_profile(),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
