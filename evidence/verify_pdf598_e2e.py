# -*- coding: utf-8 -*-
"""
PDF598 端到端验证：女书字形 -> 江永方言拼音 -> 汉字候选
====================================================
三层拆解（对照 NuShuPARS 链路）：
  A  音韵映射层：nushu_text 逐字 -> 词典 reading，对比真值 reading
  B1 候选生成层（真值拼音直喂）：gold reading -> 同音候选池，看 gold 汉字是否在池
  B2 候选生成层（字形端到端）  ：nushu char -> entry -> 候选池/首候选，对比 gold 汉字
口径分组：split(train/val/test) + sealed 对照组（论文 57.1% 的 28 行）
指标：token 级 acc / 整行 exact / 池命中率 / 朴素 Top-1(池首) + bootstrap 95% CI
"""
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(r"D:\code\nvshu")
sys.path.insert(0, str(ROOT))
from nvshu_phonetic import NvshuPhoneticLexicon, normalize_reading  # noqa: E402

DATA = ROOT / "reports/semantic_dataset/pdf598/semantic_lines.csv"
SEALED = ROOT / "paper_evidence/sealed_results/sealed_pdf598_final.json"
OUT_MD = ROOT / "verify_pdf598_e2e_report.md"
OUT_CSV = ROOT / "verify_pdf598_e2e_lines.csv"
SEED = 20261030
N_BOOT = 10000


def boot_ci(items, value_fn, n=N_BOOT, seed=SEED):
    """行级 bootstrap 95% CI。items: 行列表；value_fn: 行 -> [0/1] 或 [float]。"""
    vals = [value_fn(it) for it in items]
    if not vals:
        return (0.0, 0.0, 0.0)
    point = sum(vals) / len(vals)
    rng = random.Random(seed)
    stats = []
    m = len(vals)
    for _ in range(n):
        s = 0.0
        for _ in range(m):
            s += vals[rng.randrange(m)]
        stats.append(s / m)
    stats.sort()
    lo = stats[int(0.025 * n)]
    hi = stats[min(n - 1, int(0.975 * n))]
    return (point, lo, hi)


def fmt_ci(t):
    return f"{t[0]*100:.1f}% [{t[1]*100:.1f}, {t[2]*100:.1f}]"


def main():
    lex = NvshuPhoneticLexicon.from_project_root(ROOT)
    rows = list(csv.DictReader(open(DATA, encoding="utf-8-sig", newline="")))
    sealed = json.loads(SEALED.read_text(encoding="utf-8-sig"))
    sealed_ids = {d["id"]: d for d in sealed["details"]}

    per_line = []
    skipped = []
    for r in rows:
        nushu = (r.get("nushu_text") or "").strip()
        gold_h = (r.get("hanzi_text") or "").strip()
        gold_read_tokens = (r.get("reading") or "").strip().split()
        chars = list(nushu)
        if not chars or len(chars) != len(gold_read_tokens) or len(chars) != len(list(gold_h)):
            skipped.append((r["line_id"], len(chars), len(gold_read_tokens), len(list(gold_h))))
            continue

        a_hit = a_oov = 0
        b1_hit = b1_top1 = 0
        b2_hit = b2_top1 = b2_oov = 0
        pool_sizes = []
        for ch, gr, gh in zip(chars, gold_read_tokens, list(gold_h)):
            gr_norm = normalize_reading(gr)
            entry = lex.entry_for_char(ch)
            # A: char -> reading
            if entry is None:
                a_oov += 1
                sys_read = ""
            else:
                sys_read = entry.reading
            if sys_read and sys_read == gr_norm:
                a_hit += 1
            # B1: gold reading -> pool (isolates candidate generation)
            pool1 = lex.hanzi_pool_for_reading(gr)
            pool_sizes.append(len(pool1))
            if gh in pool1:
                b1_hit += 1
            if pool1 and pool1[0] == gh:
                b1_top1 += 1
            # B2: char -> entry pool (end-to-end phonetic chain)
            if entry is None:
                b2_oov += 1
                pool2 = []
            else:
                pool2 = list(entry.hanzi_candidates)
                for h in lex.hanzi_pool_for_reading(entry.reading):
                    if h not in pool2:
                        pool2.append(h)
            if gh in pool2:
                b2_hit += 1
            if pool2 and pool2[0] == gh:
                b2_top1 += 1

        n = len(chars)
        per_line.append({
            "line_id": r["line_id"],
            "split": r["split"],
            "n_chars": n,
            "a_hit": a_hit, "a_oov": a_oov,
            "b1_hit": b1_hit, "b1_top1": b1_top1,
            "b2_hit": b2_hit, "b2_top1": b2_top1, "b2_oov": b2_oov,
            "avg_pool": sum(pool_sizes) / n if n else 0.0,
            "a_line_exact": 1 if a_hit == n else 0,
            "b1_line_exact": 1 if b1_top1 == n else 0,
            "b2_line_exact": 1 if b2_top1 == n else 0,
            "b2_line_oracle": 1 if b2_hit == n else 0,
            "sealed_in": 1 if r["line_id"] in sealed_ids else 0,
            "gold_hanzi": gold_h,
            "sealed_top1": sealed_ids.get(r["line_id"], {}).get("top1", ""),
        })

    # aggregate
    def agg(items, name):
        if not items:
            return None
        n_tok = sum(it["n_chars"] for it in items)
        res = {"name": name, "n_lines": len(items), "n_tokens": n_tok}
        res["a_token_acc"] = boot_ci(items, lambda it: it["a_hit"] / it["n_chars"])
        res["a_oov_rate"] = boot_ci(items, lambda it: it["a_oov"] / it["n_chars"])
        res["a_line_exact"] = boot_ci(items, lambda it: it["a_line_exact"])
        res["b1_pool_hit"] = boot_ci(items, lambda it: it["b1_hit"] / it["n_chars"])
        res["b1_top1_acc"] = boot_ci(items, lambda it: it["b1_top1"] / it["n_chars"])
        res["b2_pool_hit"] = boot_ci(items, lambda it: it["b2_hit"] / it["n_chars"])
        res["b2_top1_acc"] = boot_ci(items, lambda it: it["b2_top1"] / it["n_chars"])
        res["b2_oov_rate"] = boot_ci(items, lambda it: it["b2_oov"] / it["n_chars"])
        res["b2_line_exact"] = boot_ci(items, lambda it: it["b2_line_exact"])
        res["b2_line_oracle"] = boot_ci(items, lambda it: it["b2_line_oracle"])
        res["avg_pool"] = sum(it["avg_pool"] for it in items) / len(items)
        return res

    groups = [
        agg([x for x in per_line if x["split"] == "test"], "test (60 行主口径)"),
        agg([x for x in per_line if x["split"] == "val"], "val"),
        agg([x for x in per_line if x["split"] == "train"], "train (参考)"),
        agg([x for x in per_line if x["sealed_in"] == 1], "sealed 对照组 (论文 57.1% 的 28 行)"),
        agg(per_line, "全部"),
    ]

    # write per-line csv
    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_line[0].keys()))
        w.writeheader()
        w.writerows(per_line)

    # write report
    L = []
    L.append("# PDF598 端到端验证报告（女书字形 → 江永方言拼音 → 汉字）")
    L.append("")
    L.append("日期：2026-10-30　|　数据：`reports/semantic_dataset/pdf598/semantic_lines.csv`"
             f"（{len(rows)} 行，对齐后 {len(per_line)} 行）　|　词典：NvshuPhoneticLexicon（396 字）")
    L.append("")
    L.append("## 验证设计")
    L.append("")
    L.append("| 层 | 输入 | 输出 | 回答的问题 |")
    L.append("|---|---|---|---|")
    L.append("| A 音韵映射 | 女书字形 Unicode | 江永方言拼音 | 字形→拼音对不对？ |")
    L.append("| B1 候选生成（真值拼音） | 真值拼音 | 汉字候选池 | 拼音→汉字的候选池兜不兜得住真值？ |")
    L.append("| B2 候选生成（端到端） | 女书字形 Unicode | 汉字候选池/首候选 | 字形→拼音→汉字整条链可用吗？ |")
    L.append("")
    L.append(f"- 指标均为行级 bootstrap 95% CI（{N_BOOT} 次重采样，seed={SEED}）")
    L.append("- 朴素 Top-1 = 候选池首候选（无语义排序层；封存实验的排序模型不在本次验证范围）")
    L.append(f"- 跳过 {len(skipped)} 行（字数/读音/汉字长度不一致）")
    L.append("")
    L.append("## 总表")
    L.append("")
    for g in groups:
        if not g:
            continue
        L.append(f"### {g['name']}（{g['n_lines']} 行 / {g['n_tokens']} 字）")
        L.append("")
        L.append("| 指标 | 数值 [95% CI] |")
        L.append("|---|---|")
        L.append(f"| A 拼音逐字正确率 | {fmt_ci(g['a_token_acc'])} |")
        L.append(f"| A 词典 OOV 率 | {fmt_ci(g['a_oov_rate'])} |")
        L.append(f"| A 整行全对率 | {fmt_ci(g['a_line_exact'])} |")
        L.append(f"| B1 真值拼音→池命中率 | {fmt_ci(g['b1_pool_hit'])} |")
        L.append(f"| B1 朴素 Top-1 逐字 | {fmt_ci(g['b1_top1_acc'])} |")
        L.append(f"| B2 端到端池命中率（oracle 上限） | {fmt_ci(g['b2_pool_hit'])} |")
        L.append(f"| B2 端到端朴素 Top-1 逐字 | {fmt_ci(g['b2_top1_acc'])} |")
        L.append(f"| B2 整行 Top-1 全对率 | {fmt_ci(g['b2_line_exact'])} |")
        L.append(f"| B2 整行池全命中率 | {fmt_ci(g['b2_line_oracle'])} |")
        L.append(f"| B2 平均候选池大小 | {g['avg_pool']:.1f} |")
        L.append("")
    L.append("## 与论文封存结果（sealed_pdf598_final.json）对照")
    L.append("")
    L.append(f"- 封存：test_size={sealed['test_size']}，Top-1 exact = {sealed['top1_exact_pct']:.1f}%"
             f"（{sealed['top1_exact']}/{sealed['test_size']}），Top-1 字符准确率 = {sealed['top1_char_acc']:.1f}%")
    L.append("- 口径说明：封存实验输入为**真值读音**（读音→汉字，含语义排序）；本报告 B1 与之最接近但**无排序层**"
             "（朴素池首），B2 输入为**字形**（多一层音韵映射误差）。")
    L.append("- sealed 对照组一行两看：CSV 逐行真值重算（本表）与封存 JSON 预测（下样例）互为印证。")
    L.append("")
    L.append("## sealed 对照组逐行样例（前 8 行）")
    L.append("")
    L.append("| line_id | 真值汉字 | 封存 Top-1 预测 |")
    L.append("|---|---|---|")
    shown = 0
    for it in per_line:
        if it["sealed_in"] and shown < 8:
            L.append(f"| {it['line_id']} | {it['gold_hanzi']} | {it['sealed_top1']} |")
            shown += 1
    L.append("")
    L.append("## 失败模式速览（test 集）")
    L.append("")
    test_items = [x for x in per_line if x["split"] == "test"]
    worst_a = sorted(test_items, key=lambda it: it["a_hit"] / it["n_chars"])[:5]
    worst_b = sorted(test_items, key=lambda it: it["b2_top1"] / it["n_chars"])[:5]
    L.append("拼音层（A）最差 5 行：")
    L.append("")
    L.append("| line_id | 字数 | 拼音正确 | OOV |")
    L.append("|---|---|---|---|")
    for it in worst_a:
        L.append(f"| {it['line_id']} | {it['n_chars']} | {it['a_hit']}/{it['n_chars']} | {it['a_oov']} |")
    L.append("")
    L.append("端到端（B2 朴素 Top-1）最差 5 行：")
    L.append("")
    L.append("| line_id | 字数 | Top-1 正确 | 池命中 | 真值汉字 |")
    L.append("|---|---|---|---|---|")
    for it in worst_b:
        L.append(f"| {it['line_id']} | {it['n_chars']} | {it['b2_top1']}/{it['n_chars']} | "
                 f"{it['b2_hit']}/{it['n_chars']} | {it['gold_hanzi']} |")
    L.append("")
    L.append("## 结论")
    L.append("")
    L.append("（见交付说明——结论以本表数字为准，逐行明细在 verify_pdf598_e2e_lines.csv）")
    L.append("")
    if skipped:
        L.append("## 跳过的行（长度不对齐）")
        L.append("")
        for s in skipped[:10]:
            L.append(f"- {s[0]}: chars={s[1]} readings={s[2]} hanzi={s[3]}")
        L.append("")
    OUT_MD.write_text("\n".join(L), encoding="utf-8")

    # console summary
    print("=== PDF598 E2E VERIFY ===")
    for g in groups:
        if not g:
            continue
        print(f"\n## {g['name']} ({g['n_lines']} lines / {g['n_tokens']} chars)")
        print(f"  A  reading acc      : {fmt_ci(g['a_token_acc'])}")
        print(f"  A  oov rate         : {fmt_ci(g['a_oov_rate'])}")
        print(f"  B1 pool hit         : {fmt_ci(g['b1_pool_hit'])}")
        print(f"  B2 pool hit(oracle) : {fmt_ci(g['b2_pool_hit'])}")
        print(f"  B2 top1 acc         : {fmt_ci(g['b2_top1_acc'])}")
        print(f"  B2 line exact       : {fmt_ci(g['b2_line_exact'])}")
    print(f"\nreport -> {OUT_MD}")
    print(f"lines  -> {OUT_CSV}")
    print(f"skipped: {len(skipped)}")


if __name__ == "__main__":
    main()
