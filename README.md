# NuShuPARS

**Intelligent Assisted Interpretation of Nüshu Poetry Based on Jiangyong Dialect Pronunciation**
（基于江永方言发音的女书诗歌智能辅助释读）

女书（Nüshu）是历史上由女性创造、传承使用的濒危文字。本项目实现一条可解释的释读链路：

```
女书字形 → 江永方言读音 → 汉字候选池 → 语义排序 → 释读
```

核心贡献在**读音层**：以江永方言发音为枢纽做女书字到汉字的候选映射，逐环节可解释、可核查。论文已被 **AIVRCH 2026**（2026 年人工智能、虚拟现实与文化遗产国际学术会议）录用，将提交 EI Compendex + Scopus 检索。

> **诚实声明（先读这个）**：本仓库是研究原型，不是成品翻译器。所有实测指标与已知限制见下文「实测指标」与「已知限制」两节——我们把上限写在明处。数据集为**指针版**（不含受版权保护的汉译全文），版权口径见 `dataset/README_数据卡_指针版.md`。

---

## 仓库内容

| 内容 | 说明 | 大小 |
|---|---|---|
| `USER_GUIDE_使用手册.md` | **完整使用手册**（CLI/API 详解、输出字段、故障排查、FAQ） | 小 |
| `docs/index.html` | **网页版翻译工具**（拼音→汉字 + 图片识别，单文件离线；`start_site.bat` 一键打开） | 427KB |
| `translate_cli.py` | **零依赖**离线翻译命令行（推荐入口） | 小 |
| `nvshu_phonetic.py` / `nvshu_paths.py` | 音韵词典与路径解析（纯标准库） | 小 |
| `server.py` | FastAPI 完整服务（检测→识别→翻译） | 51KB |
| `nvshu_semantic_bert_reranker.py` | 语义排序层（MLP 打分器） | 小 |
| `githubdata/nvshudata/unicode_nushu-master/` | 女书对照字典（396 字，MIT 许可） | 13KB |
| `models/nushu_unicode_best.pt` | 字符分类模型（视觉识别主力） | 45MB |
| `models/semantic_v1_final/`、`semantic_v2_final/` | 语义排序器权重 | <0.3MB |
| `fonts/NotoTraditionalNushu-Regular.ttf` | 女书字体（SIL OFL） | 0.3MB |
| `dataset/` | 数据集 v4.1 **指针版** + 完整审计报告 | <1MB |
| `evidence/` | PDF598 端到端验证报告与复验脚本 | 小 |
| `deploy/` | 服务器/笔记本部署套件与部署指南 | 小 |

**不随包分发**（附获取说明）：TrOCR 回退权重（~780MB，女书域外，仅供回退）、YOLO 检测权重（项目内没有女书微调版）、汉译全文数据集（版权原因，见数据卡）。

## 快速开始（零依赖，Python 3.9+）

> 详细用法（API 端点、输出字段释义、典型工作流、故障排查）见 **[`USER_GUIDE_使用手册.md`](USER_GUIDE_使用手册.md)**。

```bash
# 拼音 → 汉字候选
python translate_cli.py --reading "tciou siong"

# 女书字形 → 读音 → 汉字候选
python translate_cli.py --nushu "𛉾"

# JSON 输出
python translate_cli.py --reading "tciou siong" --json
```

实测输出（本机，字典顺序非排序）：

```
[tciou] → 读音 tciou   候选（30 字）：者 九 久 韭 酒 守 筹 抽 昼 州 …
[siong] → 读音 siong   候选（9 字）：参 星 姓 性 簪 幸 犯 岁 鸭
```

## 网页版（`docs/`，浏览器即用）

双击 `docs/start_site.bat`（或 `cd docs && python -m http.server 8080` 后访问 <http://127.0.0.1:8080/index.html>）：

- **翻译模块**（拼音/女书字符 → 汉字候选）：纯前端离线运行，词典与女书字体内嵌，输入不上传任何服务器；候选标注语料字频（常用度先验，非置信度）。
- **识别模块**（图片 → 女书字符 → 汉字候选）：需先启动本仓库的本地服务（见下一节），页面内填 `http://127.0.0.1:8000` 并「检测识别服务」即用；每字输出 **softmax 识别置信度**（真概率）。
- 页面底部有**模块与来源标注表**：识别 = 本项目自训分类权重（骨干 torchvision ResNet-18 + ImageNet 预训练初始化，训练图为开源女书字图合成；低置信回退 Microsoft TrOCR）；翻译 = 本项目音韵词典模块（词典数据源自 unicode_nushu，MIT）。

## 完整服务（可选，需 torch 环境）

```bash
pip install -r requirements-server.txt
python -m uvicorn server:app --host 127.0.0.1 --port 8000
curl http://127.0.0.1:8000/health
```

- 服务自带**字符分类**主力链路（`models/nushu_unicode_best.pt`，启动时自动探测），CPU 单字推理约 0.1 秒（快速 profile 下）。
- 无 TrOCR 权重时服务**仍可启动**（本仓库已打容错补丁），视觉回退不可用，不影响分类链与翻译链。
- 检测切字依赖 YOLO/投影/连通域三路兜底；无女书 YOLO 权重时整行切字不可靠（见已知限制）。
- 部署到云服务器或旧笔记本（含内网穿透方案）见 `deploy/README_部署指南.md`。

## 实测指标（全部有产物可查）

验证脚本与逐项报告在 `evidence/`，数据口径以报告为准：

| 层 | 指标 | 数值 | 口径说明 |
|---|---|---|---|
| 字形→读音 | 字典一致性 | 100% | ⚠️ 与数据集同源，是链路自洽性，**不是**独立泛化能力 |
| 读音→汉字候选池 | 字加权覆盖率 | **68.7%** | 约 1/3 的字真值不在候选池内，这是硬上限 |
| 候选池朴素 Top-1 | 行级全对 | 0% | 候选必须经语义排序才可用 |
| 语义排序 Top-1 | 整句准确率 | **57.1%**（N=28，sealed） | 论文 Table III 口径；字符准确率 77.3% |

## 数据集（v4.1 指针版）

`dataset/` 包含 6,726 行女书诗歌的**元数据、女书字形示意转写、江永方言读音、来源页码行号、汉译文本 SHA-256 指针**。汉译全文不随包发布——它是《中国女书集成》（清华大学出版社，1992）与《中国女书合集》正文的逐行抽取，整卷再传播有实质性版权风险（详见 `dataset/AUDIT_REPORT.md` §6，风险定级 P0）。持书研究者可按页码行号自行比对，用哈希校验一致性。

数据为**机器抽取 + 机器启发式标注，未经完整人工核验**（人工核对覆盖 ≤1.87%）；「女书拼音」为字典推导，非原书注音。完整已知问题（7 项）见数据卡，**引用或再分发前必读**。

## 已知限制

1. **检测切字弱**：无女书微调 YOLO 权重，手写整行图切字易错，连锁导致识别失败/乱码。单字、印刷体、预置样本不受影响。
2. **候选池覆盖 68.7%**：池兜不住的字排序救不回——这是翻译质量的结构性上限。
3. **排序分不是置信度**：语义排序器输出为无界排序分（实测校准误差 ECE≈0.35），不要当概率读。
4. **读音为字典推导**（dictionary_derived_not_source_attested）：与原书实证读音独立对照，字位一致率约 43%（严格）/ 63%（音系骨架）。
5. **TrOCR 回退对女书域不熟**（输出 ASCII），仅作兜底，不建议用于展示。

## 引用

```bibtex
@inproceedings{chen2026nushupars,
  title     = {NuShuPARS: Intelligent Assisted Interpretation of N{\"u}shu Poetry Based on Jiangyong Dialect Pronunciation},
  author    = {Chen, Lingyun and Li, Tong and Mao, Tengyue},
  booktitle = {Proceedings of the 2026 International Conference on Artificial Intelligence, Virtual Reality and Cultural Heritage (AIVRCH 2026)},
  year      = {2026},
  note      = {Accepted; to appear in EI Compendex / Scopus indexed proceedings}
}
```

（`CITATION.cff` 同步提供。引用数据集请另注明版本 `nushu-poetry-v4.1-pointer` 与 `dataset/SHA256SUMS.txt`。）

## 许可

- **代码**：MIT（见 `LICENSE`）
- **字典**（`githubdata/nvshudata/unicode_nushu-master/`）：MIT，© 2019 The Nushu Coder's Group on GitHub（字形遵循原许可）
- **字体**：SIL Open Font License 1.1（见 `fonts/OFL.txt`）
- **数据集**（`dataset/`）：仅供科研与文化遗产保护用途，禁止商业使用；内容版权归属原出版物著作权人
- **第三方依赖**（ultralytics 为 AGPL-3.0 的可选组件等）：见 `NOTICE.md`

使用本数据集与代码时请尊重女书的文化语境，避免猎奇化、娱乐化呈现。欢迎江永当地文化工作者与研究者提出修正意见。
