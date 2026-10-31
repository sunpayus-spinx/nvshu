# NuShuPARS Web 服务部署指南（双目标：阿里云 / 老笔记本）

日期：2026-10-30　|　适用：`server.py`（FastAPI 女书识别+翻译服务）
本指南所有命令在本机实测过（Windows + Git Bash / PowerShell）；标注〔估〕的数字未经实测。

> **与本发布仓库（GitHub 版）的差异**：本仓库不含 `trocr_model_nushu/`（约 780MB，不随包分发）、`yolov8n.pt`、`web/`、`configs/`。
> 从本仓库启动时：① `NVSHU_CHARCLS_CKPT` 可省略——server 会自动探测 `models/nushu_unicode_best.pt`；
> ② TrOCR 权重缺失时服务**仍可启动**（已打容错补丁），仅视觉回退不可用；③ 第 0 节清单中仅需部署仓库内文件。
> 需要完整视觉链（TrOCR 回退）时，自行准备微调权重目录并设 `NVSHU_REC_MODEL_DIR`。其余步骤照旧。
> 注意：`test_char.png` 为 Noto 字体渲染样本，与分类模型训练字形风格有差异，识别结果可能与目标码位不一致——冒烟测试只验证链路通畅与 Unicode 输出（不应出现 ASCII 乱码），不作为字准依据。

---

## 0. 部署物清单（缺一不可）

从 `D:\code\nvshu` 拷贝以下内容到目标机器（保持相对目录结构）：

| 路径 | 作用 | 大小 |
|---|---|---|
| `server.py` | FastAPI 主服务 | 51KB |
| `nvshu_paths.py` / `nvshu_phonetic.py` | 路径解析 / 音韵词典 | 小 |
| `web/`（index.html + main.js） | 前端 | 27KB |
| `trocr_model_nushu/` | TrOCR 识别模型（回退用） | ~780MB |
| `models/nushu_unicode_best.pt` | **字符分类模型（主力，实测唯一有效）** | 45MB |
| `yolov8n.pt` | YOLO 检测（目前是 COCO 通用版，仅兜底） | 6.5MB |
| `models_cache/ultralytics/` | YOLO 配置缓存 | 小 |
| `githubdata/nvshudata/unicode_nushu-master/data.csv` | **女书对照表（396 字，音韵链核心数据）** | 小 |
| `configs/training_config.yaml` | 模型路径配置（服务启动会读） | 小 |
| `deploy/`（本目录） | 启动脚本+冒烟测试 | 小 |

## 1. 关键配置（实测踩坑后的固化值）

```bash
# 必设——不设则启动必挂（配置文件指向不存在的 models\trocr-nvshu-finetuned）
NVSHU_REC_MODEL_DIR=<项目根>/trocr_model_nushu
# 必设——不设则分类模型加载失败，识别退化为 TrOCR 吐 ASCII 乱码（实测唯一有效 ckpt）
NVSHU_CHARCLS_CKPT=<项目根>/models/nushu_unicode_best.pt
# YOLO 配置缓存
YOLO_CONFIG_DIR=<项目根>/models_cache/ultralytics
# Windows + conda/Anaconda Python 必设（否则 OMP dll 冲突报错）
KMP_DUPLICATE_LIB_OK=1
PYTHONNOUSERSITE=1
```

**启动后立即调一次** `GET /rest/rec_profile`：会把识别参数从"慢而稳"（TTA 开、beam 12）
切到快速基线（TTA 关、beam 8），CPU 上预计提速 2–4 倍〔估〕。`deploy/post_start_tune.py` 自动做。

## 2. 目标 A：阿里云云服务器

### 选型参考（2026-10 价格，以官网为准）

| 规格 | 参考价 | 预期表现 |
|---|---|---|
| 轻量应用服务器 2C4G（CPU） | ~60–100 元/月 | 能跑，比今天实测的台式机 CPU 再慢 1.5–3 倍〔估〕 |
| GPU 按量实例（gn7i/A10 或同级） | ~2–8 元/小时 | 单字 <0.2s、整行秒级〔估〕；演示体验最好 |
| GPU 包月 | 数百–上千元/月 | 同上，长期用才划算 |

### 步骤（以 Ubuntu 22.04 为例，裸装）

```bash
# 1) 系统依赖
sudo apt-get update && sudo apt-get install -y python3-venv libgl1

# 2) 项目文件上传（在本机执行；把 D:\code\nvshu 换成你的本地路径）
scp -r deploy server.py nvshu_paths.py nvshu_phonetic.py web trocr_model_nushu \
      yolov8n.pt configs githubdata user@<服务器IP>:/opt/nvshu/
scp models/nushu_unicode_best.pt user@<服务器IP>:/opt/nvshu/models/

# 3) Python 环境（PyPI 直连慢/不通时加阿里云镜像）
cd /opt/nvshu
python3 -m venv venv && . venv/bin/activate
pip install -i https://mirrors.aliyun.com/pypi/simple/ -r deploy/requirements_web.txt
# GPU 机器才装 CUDA 版 torch（默认装的 CPU 版跑不动 GPU）：
bash deploy/install_cuda_torch.sh

# 4) 启动
bash deploy/start_server.sh
```

- **安全组**：放行 TCP 8000（或自定义端口）。仅对可信网段开放更佳。
- **Docker 可选**：项目根的 `Dockerfile`（CUDA 11.8 基底）可直接 `docker build`；
  但注意它不会自动注入第 1 节的环境变量，需 `-e` 传入。

## 3. 目标 B：老笔记本做服务器（Windows）

### 3.1 装环境

1. 装 Python 3.10+（python.org 或 Anaconda）
2. `pip install -r deploy\requirements_web.txt`（PyPI 慢就加 `-i https://mirrors.aliyun.com/pypi/simple/`）
3. 把部署物清单的文件拷到如 `D:\nvshu\`

### 3.2 启动

```bat
deploy\start_server.bat
```

启动后自动做两件事：应用快速识别 profile、跑冒烟测试。

### 3.3 内网穿透（没有公网 IP，必须做这步才有公网 URL）

| 方案 | 成本 | 特点 |
|---|---|---|
| **Cloudflare Tunnel（推荐）** | 免费 | 免公网 IP、自动 HTTPS、域名可自选；装 cloudflared 一条命令 |
| cpolar / 花生壳 | 免费档够用 | 国内访问快；免费域名随机、带宽受限 |
| frp 自建 | 需一台有公网 IP 的小服务器 | 最可控；与阿里云轻量搭配 |

Cloudflare Tunnel 最小步骤：

```bat
winget install --id Cloudflare.cloudflared
cloudflared tunnel --url http://localhost:8000
```

输出的 `https://xxx.trycloudflare.com` 就是公网地址（临时域名；要固定域名需绑定自有域名建 Named Tunnel）。

### 3.4 老笔记本常驻注意

- 电源计划设"从不睡眠"；合盖不休眠（控制面板→电源选项）
- Windows 防火墙放行 8000（仅内网穿透场景可不放行公网，穿透工具走出站）
- **性能预期**：老 CPU 比今天实测的台式机慢 2–3 倍〔估〕——演示以单字/短行、预置样本为主

## 4. 通用安全（两目标都要）

1. **上传限制**：server.py 当前无文件大小上限。用 Nginx 反代时加 `client_max_body_size 5m;`；
   或直接信任内网/穿透场景（demo 用途风险可接受）。
2. **访问控制**：服务无鉴权。建议 Nginx basic auth 或 Cloudflare Access（免费 50 用户）挡一道。
3. **只开必要端口**：8000 一个就够；SSH 用密钥禁密码。
4. 服务只跑 `--host 0.0.0.0 --port 8000`（脚本默认），不需要 root。

## 5. 验收清单（部署完逐条打勾）

```bash
# 1) 健康检查——五个关键字段必须是这些值
curl http://127.0.0.1:8000/health
#   rec_model_loaded: true
#   rec_model_source: .../trocr_model_nushu
#   cls_model_loaded: true   ← 不是 true 就检查 NVSHU_CHARCLS_CKPT
#   phonetic_lexicon_size: 396
#   device: cpu 或 cuda:true（GPU 机器）

# 2) 快速 profile 已生效
curl http://127.0.0.1:8000/rest/rec_profile   # rec_tta_enabled 应为 false

# 3) 冒烟测试（真实推理一张女书图）
python deploy/smoke_test.py http://127.0.0.1:8000
#   期望：HTTP 200，返回女书 Unicode（U+1Bxxx），不应出现 ASCII 乱码
```

**速度参考（今天台式机 CPU 实测）**：单字分类 1–2s；整行（含检测+TrOCR 回退）70–115s（慢配置）。
快速 profile + 分类主力走对后，单字秒级、短行数秒〔估〕。GPU 机器预计整行 <1s〔估〕。

## 6. 已知限制（演示前须知）

1. **检测环节弱**：无女书 YOLO 权重（COCO 通用 yolov8n 不认女书字形），切字靠投影/连通域，
   整行手写图容易切错→识别连锁出错。**演示建议**：单字图、印刷体、或预置样本（web_demo/）。
2. **翻译链已实测可用**（见 `verify_pdf598_e2e_report.md`）：字形→拼音 100% 自洽、
   拼音→汉字候选池字加权命中 68.7%。视觉层是短板，翻译层是强项——演示重心放在翻译链。
3. TrOCR 回退模型对女书域不熟（认 ASCII），确保分类模型加载成功再演示。
