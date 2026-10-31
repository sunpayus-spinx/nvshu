#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NuShuPARS 离线翻译命令行（零第三方依赖，Python 3.9+）。

链路：女书字形 → 江永方言读音 → 汉字候选池
     （拼音输入则从读音一步开始）

用法示例：
  python translate_cli.py --reading "tciou siong"
  python translate_cli.py --nushu "𛆆𛉾"
  python translate_cli.py --reading "ang njie cya ciang miong tciang hau" --json

说明：
  - 候选来自《女书标准字字典》系词典（见 githubdata/ 目录与 NOTICE.md），
    按字典顺序给出，不是排序结果，也不带概率。
  - 带调值的读音（如 tciou35）会自动剥离调值。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nvshu_phonetic import NvshuPhoneticLexicon, normalize_reading  # noqa: E402


def build_lexicon() -> NvshuPhoneticLexicon:
    root = Path(__file__).resolve().parent
    lex = NvshuPhoneticLexicon.from_project_root(root)
    if not lex.by_char:
        print("错误：未找到字典 githubdata/nvshudata/unicode_nushu-master/data.csv", file=sys.stderr)
        sys.exit(2)
    return lex


def translate_reading(lex: NvshuPhoneticLexicon, text: str) -> list[dict]:
    syllables = [s for s in text.replace("/", " ").split() if s]
    out = []
    for syl in syllables:
        norm = normalize_reading(syl)
        pool = lex.hanzi_pool_for_reading(syl)
        out.append({"input": syl, "reading": norm, "candidates": pool, "pool_size": len(pool)})
    return out


def translate_nushu(lex: NvshuPhoneticLexicon, text: str) -> list[dict]:
    out = []
    for ch in text.strip():
        if ch.isspace():
            continue
        entry = lex.entry_for_char(ch)
        if entry is None:
            out.append({"nushu": ch, "reading": None, "candidates": [], "pool_size": 0, "oov": True})
            continue
        pool = lex.hanzi_pool_for_reading(entry.reading)
        out.append({
            "nushu": ch,
            "unicode": f"U+{ord(ch):X}",
            "reading_raw": entry.raw_reading,
            "reading": entry.reading,
            "candidates": pool,
            "pool_size": len(pool),
            "oov": False,
        })
    return out


def pretty(rows: list[dict], mode: str) -> None:
    for r in rows:
        if mode == "nushu":
            head = f"{r['nushu']} ({r.get('unicode', '?')})"
            if r.get("oov"):
                print(f"{head}  →  未收录于字典")
                continue
            print(f"{head}  →  读音 {r['reading']}  (原始记音 {r['reading_raw']})")
        else:
            print(f"[{r['input']}]  →  读音 {r['reading']}")
        if r["pool_size"] == 0:
            print("    候选：（空 —— 该读音在字典中无对应汉字）")
        else:
            print(f"    候选（{r['pool_size']} 字，字典顺序，非排序）：{' '.join(r['candidates'])}")
    print()
    print("注：候选为字典顺序，不是排序结果；读音为字典推导，非原书注音。")


def main() -> None:
    ap = argparse.ArgumentParser(description="女书字形/读音 → 汉字候选（离线字典链）")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--reading", help="江永方言读音，空格/斜杠分隔音节，可带调值数字")
    g.add_argument("--nushu", help="女书 Unicode 字符（可多个）")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    args = ap.parse_args()

    lex = build_lexicon()
    if args.reading is not None:
        rows = translate_reading(lex, args.reading)
        mode = "reading"
    else:
        rows = translate_nushu(lex, args.nushu)
        mode = "nushu"

    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        pretty(rows, mode)


if __name__ == "__main__":
    main()
