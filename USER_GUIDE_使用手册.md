# NuShuPARS 使用手册

适用对象：从 GitHub 下载本仓库后要**实际跑起来**的使用者（含论文检查者）。
覆盖三种用法：离线命令行（零依赖）→ HTTP 服务（完整功能）→ 常见问题排查。
部署到服务器请另读 [`deploy/README_部署指南.md`](deploy/README_部署指南.md)；数据集版权口径见 [`dataset/README_数据卡_指针版.md`](dataset/README_数据卡_指针版.md)。

---

## 0. 选哪种用法

| 你想做的事 | 用什么 | 需要装什么 | 耗时 |
|---|---|---|---|
| 拼音/女书字形 → 汉字候选（论文核心链路） | **命令行 `translate_cli.py`** | 只要 Python 3.9+，**零第三方依赖** | 秒级 |
| 上传图片做识别+翻译，或给别人提供接口 | HTTP 服务（`server.py`） | torch + requirements-server.txt | 首次装环境约 10 分钟 |
| 只想核验论文指标 | `evidence/verify_pdf598_e2e.py` | Python 3.9+ | 数分钟 |

> 术语约定：**读音（reading）** = 江永方言拼音（如 `tciou siong`）；**女书字形** = 女书 Unicode 字符（如 𛆆𛉾）；**候选池** = 同一读音对应的全部汉字。

---

## 1. 命令行用法（translate_cli.py）

### 1.1 快速上手

```bash
# 读音 → 汉字候选
python translate_cli.py --reading "tciou siong"

# 女书字形 → 读音 → 汉字候选
python translate_cli.py --nushu "𛆆𛉾"

# JSON 格式输出（供脚本处理）
python translate_cli.py --reading "ang njie cya ciang miong tciang hau" --json
```

实测输出示例（`--reading "tciou siong"`，本机实跑）：

```
[tciou]  →  读音 tciou
    候选（30 字，字典顺序，非排序）：者 九 久 韭 酒 守 筹 抽 昼 州 周 洲 咒 交 旧 娇 绞 教 救 较 究 舅 纠 求 球 丘 粥 祝 角 觉
[siong]  →  读音 siong
    候选（9 字，字典顺序，非排序）：参 星 姓 性 簪 幸 犯 岁 鸭
```

### 1.2 参数表

| 参数 | 必选 | 说明 |
|---|---|---|
| `--reading TEXT` | 二选一 | 江永方言读音。空格或 `/` 分隔音节；可带调值数字（如 `tciou35`，会自动剥离调值） |
| `--nushu TEXT` | 二选一 | 女书 Unicode 字符，可一次给多个 |
| `--json` | 否 | 以 JSON 输出（便于程序消费） |

### 1.3 输出字段释义

`--json` 模式下每个音节/字返回一个对象：

| 字段 | 含义 |
|---|---|
| `input` / `nushu` | 你输入的音节或女书字 |
| `reading` | 归一后的读音（已剥调值） |
| `reading_raw` | 字典中的原始记音 |
| `unicode` | 女书字码位（如 `U+1B186`） |
| `candidates` | 汉字候选列表 |
| `pool_size` | 候选个数 |
| `oov` | true = 该字未收录于字典 |

### 1.4 重要口径（请务必理解）

1. **候选是字典顺序，不是排序结果。** 它回答"这个读音可能对应哪些汉字"，不回答"最可能是哪个"。语义排序（Top-1）在服务端才有（见 §2）。
2. **候选不带概率/置信度。** 候选列表本身无分值；任何"置信度百分比"都不存在于本层。
3. **读音是字典推导，非原书注音。** 女书字形 → 读音的映射来自《女书标准字字典》系词典（见 `NOTICE.md`）。

---

## 2. HTTP 服务用法（server.py）

### 2.1 启动

```bash
pip install -r requirements-server.txt          # torch 等，约 10 分钟
python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

启动后浏览器打开 `http://127.0.0.1:8000/docs` 有自动生成的交互式接口文档（FastAPI Swagger）。
启动脚本（Windows/Linux）在 `deploy/start_server.bat` / `deploy/start_server.sh`，含推荐的环境变量设置。

### 2.2 端点全表

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/health` | 健康检查：模型加载状态、设备、字典规模 |
| POST | `/rest/infer` | **图片识别+翻译**（multipart 上传 `file`） |
| POST | `/rest/translate` | **女书字符串翻译**（JSON `{"text": "𛅼𛋫"}`） |
| GET | `/rest/rec_profile` | 查看当前识别配置（已固化为最优参数） |
| POST | `/rest/set_charcls_ckpt` | 热切换字符分类模型 |
| POST | `/rest/set_model_dir` | 热切换 TrOCR 识别模型目录 |
| POST | `/rest/set_det_source` / `/rest/set_det_model` | 设定检测来源/检测模型 |
| POST | `/rest/set_rec_params` | 识别参数（**当前锁定**，见下注） |
| GET | `/` | 跳转到 `/docs` |

> 注：识别策略已锁定为稳定基线（`charcls_only_no_fallback`），`set_rec_params` 会返回 `locked: true` 并保持现有配置——这是有意设计（保证复现结果一致），不是接口坏了。

### 2.3 /rest/infer —— 图片识别+翻译（主接口）

请求（multipart/form-data，字段名 `file`）：

```python
import urllib.request, json, os
B = '----nvshu'
p = 'test.png'
data = open(p, 'rb').read()
body = (f'--{B}\r\nContent-Disposition: form-data; name="file"; filename="t.png"\r\n'
        f'Content-Type: image/png\r\n\r\n').encode() + data + f'\r\n--{B}--\r\n'.encode()
req = urllib.request.Request('http://127.0.0.1:8000/rest/infer', data=body,
                             headers={'Content-Type': f'multipart/form-data; boundary={B}'})
res = json.loads(urllib.request.urlopen(req, timeout=300).read().decode())
```

响应字段：

| 字段 | 含义 |
|---|---|
| `results[]` | 每个检测框一条记录（见下表） |
| `n_boxes` | 检测到的字块数 |
| `det_source` | 检测来源：`yolo` / `proj_fallback`（投影切分）/ `cc_fallback`（连通域）/ `fullimage`（全图兜底） |
| `rec_mode` | 整体识别模式：`char_cls`（字符分类，主路径）/ `trocr` / `fusion` |
| `fallback_count` | 退化处理的框数 |
| `full_text` / `full_unicode` | 整行识别结果及其码位 |
| `full_pinyin` | 整行读音 |
| `full_chinese` | 整行汉字候选（首选串联） |
| `full_phonetic` | 整行音韵分析 |
| `api_version` | 接口版本 |

`results[]` 单条字段：

| 字段 | 含义 |
|---|---|
| `index`, `box`, `expand_box` | 序号、检测框、外扩框坐标 |
| `text`, `text_raw`, `unicode` | 识别出的女书字（归一/原始/码位） |
| `pinyin` | 该字读音 |
| `chinese` | 首选汉字 |
| `candidates` | 该读音的全部汉字候选 |
| `candidate_pool` | 同读音候选池（音韵分析口径） |
| `phonetic` | 逐字音韵分析 |
| `confidence` | **视觉识别置信度**（softmax 概率，仅对"这个字形是什么字"有效） |
| `topk` | 识别模型的前 k 个候选（含各自 score） |
| `rec_mode` | 该框的识别模式 |
| `needs_review` | true = 置信度低于阈值，建议人工复核 |
| `crop_png_base64` | （仅 debug 模式）切图快照 |

### 2.4 /rest/translate —— 字符串翻译（无需图片）

```bash
curl -X POST http://127.0.0.1:8000/rest/translate \
     -H "Content-Type: application/json" \
     -d '{"text": "𛅼𛋫𛇼"}'
```

返回：`text`（归一）、`text_raw`、`unicode`、`pinyin`、`chinese`、`details`（逐字明细）、`phonetic`（音韵分析）。
这条链路与命令行 `--nushu` 等价，但多出音韵分析细节。

### 2.5 /health —— 状态检查

```bash
curl http://127.0.0.1:8000/health
```

关键字段：`device`/`cuda`（运行设备）、`cls_model_loaded` + `cls_model_arch`（分类模型，主识别路径）、`rec_model_loaded`（TrOCR，可选回退）、`phonetic_lexicon_size`（字典条目数）、`strategy_locked`（识别策略锁定状态）、`det_error`（检测模型加载问题）。
冒烟脚本：`python deploy/smoke_test.py http://127.0.0.1:8000`。

---

## 3. 典型工作流

**场景 A：批量翻译一批读音（论文复现/数据处理）**

```bash
while IFS= read -r line; do
  python translate_cli.py --reading "$line" --json >> out.jsonl
done < readings.txt
```

**场景 B：起服务给同事/检查者用**

```bash
bash deploy/start_server.sh            # 或 start_server.bat（Windows）
python deploy/post_start_tune.py http://127.0.0.1:8000   # 应用快速识别配置
python deploy/smoke_test.py http://127.0.0.1:8000        # 冒烟确认
# 对方浏览器打开 http://<你的IP>:8000/docs 即可交互试用
```

**场景 C：集成到自己的 Python 脚本（不起 HTTP）**

```python
import sys
from pathlib import Path
root = Path(r"路径\release_repo")          # 注意：必须是 Path 对象，不能是字符串
sys.path.insert(0, str(root))
from nvshu_phonetic import NvshuPhoneticLexicon
lex = NvshuPhoneticLexicon.from_project_root(root)
print(lex.hanzi_pool_for_reading("siong"))   # -> ['参', '星', '姓', '性', '簪', '幸', '犯', '岁', '鸭']
e = lex.entry_for_char("𛉾")
print(e.reading, e.raw_reading)              # -> 归一读音 + 字典原始记音
```

---

## 4. 置信度的诚实口径（重要）

| 环节 | 有没有分值 | 能否当"可信度百分比" |
|---|---|---|
| 视觉识别（字形 → Unicode） | ✅ `confidence` | **可以**（softmax 概率），低置信会标 `needs_review` |
| 汉字候选池（读音 → 汉字列表） | ❌ 无 | 候选是字典顺序，**没有概率** |
| 语义排序（服务端可选） | ✅ 分数 | **不可以**（是排序分不是概率，未校准） |

**因此：不要在任何界面把候选列表标成"置信度 92%"。** 需要展示可信度时，只用视觉层的 `confidence`，或只展示排序不标数字。论文报的 Top-1/Top-64 是命中率（排序口径），不是概率输出——两者不要混用。

---

## 5. 故障排查

| 现象 | 原因 | 解法 |
|---|---|---|
| `错误：未找到字典 githubdata/nvshudata/unicode_nushu-master/data.csv` | 仓库没下完整（大文件/目录缺失） | 重新 clone，确认 `githubdata/` 目录存在 |
| 服务启动报找不到 `models/trocr-nvshu-finetuned` 或尝试联网下载 | TrOCR 权重未随包（780MB，域外回退模型） | 无需处理即可用主链；确需完整视觉链时设 `NVSHU_REC_MODEL_DIR` 指向自备权重目录 |
| `/health` 里 `cls_model_loaded: false` | 分类模型未找到 | 确认 `models/nushu_unicode_best.pt`（45MB）在仓库内；或 `POST /rest/set_charcls_ckpt {"ckpt": "路径"}` 手动指定 |
| 识别结果是 ASCII 乱码（"23"、"s05a90ex7"） | 未加载分类模型而 TrOCR 回退兜底（TrOCR 是域外模型） | 加载分类模型（上一行）；乱码结果请勿采信 |
| `POST /rest/set_rec_params` 说 locked | 识别策略有意锁定为稳定基线 | 正常行为；配置已固化为最优，无需调参 |
| 多字图切框少/切乱 | 无女书专用检测权重（YOLO 是 COCO 通用模型） | 已知限制。演示/使用建议：单字图或已切好的字块效果最好（见 README"已知限制"） |
| 字形渲染图（如用 Noto 字体生成）识别不准 | 字体风格与训练字形有差异 | 正常现象；真实书写字形图片效果更好 |
| `pip install torch` 很慢/失败 | PyPI 直连不稳定 | 用国内镜像：`pip install -r requirements-server.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`；GPU 机器见 `deploy/install_cuda_torch.sh` |
| 候选里没有你要的那个字 | 该读音的字未收录（约 1/3 真字不在池内，见验证报告） | 这是字典覆盖率上限，不是 bug；可换输入形式或查 `evidence/verify_pdf598_e2e_report.md` 了解口径 |

---

## 6. FAQ

**Q：能把这个系统当"图片→女书→汉字"的全自动翻译器用吗？**
不建议。图片→字形的检测是已知短板（无女书专用检测权重）；字形→读音→汉字的翻译链是本系统验证过的部分。推荐输入是**读音（拼音）或已切好的女书字形**。

**Q：为什么候选没有"置信度"？**
见 §4。候选池是字典查表结果，没有概率模型支撑；给它标百分比是不诚实的。

**Q：能不能保证翻译正确？**
不能。实测口径（见 `evidence/verify_pdf598_e2e_report.md`，全部有产物可查）：读音→汉字候选池覆盖率 68.7%（字加权）；配语义排序后 Top-1 整句准确率 57.1%（N=28 封存测试集）。候选是"可能对应的汉字"，最终释读需人工判断。

**Q：数据集能随便用吗？**
不能。汉译全文受出版物版权约束未随包（指针版含页码+SHA-256 校验），使用前必读 `dataset/README_数据卡_指针版.md`，仅限研究用途。

**Q：模型/代码能商用吗？**
代码 MIT（见 LICENSE）。数据集 research-only。第三方组件（MIT 字典、OFL 字体等）见 `NOTICE.md`。

**Q：想复现论文的验证数字怎么办？**
`python evidence/verify_pdf598_e2e.py`——报告里的每个数字都能重跑核验（需 `reports/semantic_dataset/pdf598/` 数据，见证据包说明）。

---

## 7. 相关文档索引

| 文档 | 内容 |
|---|---|
| `README.md` | 仓库概览、快速开始、实测指标、已知限制 |
| `deploy/README_部署指南.md` | 服务器部署（阿里云/老笔记本双目标）、安全加固 |
| `dataset/README_数据卡_指针版.md` | 数据集构成、版权口径、7 条已知问题 |
| `dataset/AUDIT_REPORT.md` | 数据集公开前完整审计 |
| `evidence/verify_pdf598_e2e_report.md` | 端到端验证报告（含 bootstrap 置信区间） |
| `NOTICE.md` | 第三方组件与许可清单 |
| `CITATION.cff` | 引用格式 |
