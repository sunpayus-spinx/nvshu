# NOTICE — 第三方组件与许可清单

本仓库（NuShuPARS）原创代码以 MIT 许可发布（见 `LICENSE`）。以下第三方组件保留各自许可，随包分发时须随附本清单及原始许可文本。

## 随包分发的第三方组件

| 组件 | 位置 | 许可 | 说明 |
|---|---|---|---|
| unicode_nushu 数据与构建代码（The Nushu Coder's Group on GitHub, 2019） | `githubdata/nvshudata/unicode_nushu-master/` | MIT（代码与数据）；女书字形遵循原许可 | 女书字符 ↔ 读音 ↔ 汉字对照字典（396 字），音韵链核心数据。原始许可文本见该目录 `LICENSE` |
| Noto Traditional Nushu 字体 | `fonts/NotoTraditionalNushu-Regular.ttf` | SIL Open Font License 1.1 | 仅用于渲染女书字样例（`deploy/test_char.png` 由本字体渲染生成）。许可全文见 `fonts/OFL.txt` |

## 运行期依赖（不随包分发，由用户环境安装）

| 依赖 | 许可 | 用途 | 备注 |
|---|---|---|---|
| PyTorch / torchvision | BSD-style | 模型推理 | `requirements-server.txt` |
| FastAPI / Uvicorn / python-multipart | MIT | Web 服务 | 同上 |
| Hugging Face Transformers | Apache-2.0 | TrOCR 回退（可选） | 同上 |
| Pillow / NumPy / OpenCV / PyYAML | MIT-CMU / BSD-3 / Apache-2.0 / MIT | 图像与配置处理 | 同上 |
| **Ultralytics（YOLO）** | **AGPL-3.0** | **仅可选的 YOLO 检测路径**（懒加载） | 见下方说明 |

### 关于 Ultralytics（AGPL-3.0）的特别说明

`server.py` 的 YOLO 检测路径（可选组件，懒加载；本仓库不随包分发 YOLO 权重）依赖
[Ultralytics](https://github.com/ultralytics/ultralytics)（AGPL-3.0）。

- **核心链路不涉及**：`translate_cli.py`、音韵词典、字符分类模型（纯 PyTorch 加载）、语义排序器均不导入 ultralytics。
- **若你分发**包含 YOLO 检测路径的衍生软件（例如打包为闭源服务对外提供），你的分发行为触发 AGPL-3.0 的相应义务（提供对应源码）。仅内部使用、不触发分发条款的场景不受此约束。
- 不希望承担 AGPL 义务的使用者：不安装 `ultralytics` 即可，服务会自动跳过 YOLO 路径、退化为投影/连通域切字。

## 数据集的独立条款

`dataset/` 下的数据**不在 MIT 许可范围内**：仅供科研与文化遗产保护用途、禁止商业使用；
其中汉译文本内容的著作权归属原出版物（《中国女书集成》清华大学出版社 1992；《中国女书合集》中华书局系），
本仓库仅发布指针与哈希。详见 `dataset/README_数据卡_指针版.md` 与 `dataset/AUDIT_REPORT.md`。

## 模型权重

`models/nushu_unicode_best.pt` 与 `models/semantic_*` 为本项目训练产出（在自有/开源女书字图数据上训练的权重），随代码同按 MIT 发布。
注意区分**权重**与**骨架**：分类模型的骨干架构为 torchvision ResNet-18（BSD-style，见上表），训练脚本默认以 ImageNet 预训练权重初始化（可 `--no-pretrained` 关闭）——架构与预训练初始化来自开源生态，**微调后的权重是本项目产出**。
训练数据的版权约束不因此转移至模型使用者，但请遵守数据卡中的使用限制精神。
