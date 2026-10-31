# CHANGELOG · v4.0 → v4.1 修复版（2026-10-30）

依据：`dataset_audit_20261030/女书诗歌数据集v4_公开前审计报告_20261030.md` 修复路线图
原则：**原始数据文件未做任何改动**（v4.0 各 CSV 保持原状）；全部修复以新增列 /
新增文件形式产出，逐项可追溯。构建脚本：`dataset_audit_20261030/build_release_v41.py`。

## 一、修复项逐条记录

| 路线图# | 问题 | 动作 | 修复前 | 修复后 |
|---|---|---|---|---|
| 2 | P1-3 人工核验表述 | README 落版「已知问题」7 条 + 伦理声明；明确「未经完整人工核验」 | 无数据卡 | `README_数据卡.md` |
| 3 | P1-2 示意列口径 | 列改名 + README 警示 | `女书unicode(示意)` | `nushu_unicode_schematic(程序示意_非原书字形)`，1,337 行加 `nushu_schematic_incomplete=yes` |
| 4 | P1-5 评测污染披露 | 逐行标记 + 清单随包 | 无标记 | `heldout_gold_overlap` 列 11 条记录 / `heldout_overlap_list.csv` |
| 5 | P1-4 训练派生集含密封行 | 剔除 heldout 相关行，产出净化版 | train_C 7,136 行 / train_B 7,556 行（含 47 id + 46 gold） | train_C 7,087 / train_B 7,507（各剔 49 行 = 47 id + 2 同文行）；泄漏留档 `train_derived_leak_list.csv` |
| 6 | P1-6 「页面一致」断言失真 | 全量加核验状态列；8 行真错改标 | 断言无校验 | `page_match_status` 全量；8 行改标「页面待复核(与OCR窗口不一致)」（行 238/474/1078/4257/4260/4292/4655/5076） |
| 7 | P2-1/P2-5/P1-1 结构化披露 | 加 5 列 | 无 | `dup_group_id`/`dup_type`（459 组/1,020 行：套语重唱 246、两书互证 198、疑似抽取双计 15）；`has_missing_glyph`/`missing_glyph_count`（89 行）；`reading_status`（66 行 source_attested_*） |
| 8 | P2-4 繁简混排 | 加归一列，原样保留 | 266 行简繁混排无归一视图 | `汉译诗句_simplified` 全量简体；`汉译诗句` 保持抽取原样 |
| 9 | P2-6 发布工程 | 校验和 + 引用格式 | 无 | `SHA256SUMS.txt` + README 引用节（bibtex 草案） |

## 二、新增列口径（10 列）

1. `汉译诗句_simplified` — hanziconv 简体归一
2. `page_match_status` — 与 OCR 中间产物窗口核验（口径见 README）
3. `确认依据_修复后` — 仅 8 行内容与原值不同
4. `dup_group_id` — 简体文本重复组编号（D0001–D0459）
5. `dup_type` — 套语重唱 / 两书互证 / 疑似抽取双计
6. `has_missing_glyph` — 行内含「口」占位（个别行「口」可能是正文实字，须人工甄别）
7. `missing_glyph_count` — 「口」计数
8. `nushu_schematic_incomplete` — 示意女书字数<汉字数（1,337 行；较审计 C3 的 1,268 多 69 行，系音节组数本就≠字数的行一并计入，口径差异已知）
9. `reading_status` — dictionary_derived_not_source_attested（6,660）/ source_attested_partial（61）/ row_match（4）/ mismatch（1）
10. `heldout_gold_overlap` — yes:<v2_id>（11 条记录，涉 10 个 gold 文本）

## 三、独立复算与审计互证

- `content_mismatch` 8 行与审计报告 §0-4 的 8 行**逐行一致**（238/474/1078/4257/4260/4292/4655/5076），独立重算互证。
- 重复归因 246/198/15 与审计 `B5_attrib` **完全一致**（归因规则：简体归一文本分组 → 跨书/同书相邻页(≤1)/同书远距）。
- 行 422/3824：PDF 目检反转行（CSV 正确、OCR 中间产物误读），`page_match_status=fuzzy_minor`，不计入错误。

## 四、未修复项（须人工决策，见审计报告 §7）

| # | 项 | 原因 |
|---|---|---|
| 1 | 版权口径（P0） | 授权/指针/限定三选一须导师/单位书面决策 |
| 10 | 300 行 PDF 逐字人工金标 | 需 2–3 人日人工 |
| 11 | 第二标注者 IAA（E4） | 需招募第二标注者 |
| 12 | ABC 标签重标定 | 需独立集；在弃用 A 率声明前不得引用 A 率 |

## 五、构建校验

- 构建脚本：`dataset_audit_20261030/build_release_v41.py`（幂等可重跑）
- 构建摘要：`build_summary.json`（本目录）
- 全包 SHA256：`SHA256SUMS.txt`（README/CHANGELOG 定稿后重新生成）
