"""Nvshu phonetic lexicon and candidate-pool helpers.

This module keeps translation separate from visual recognition:

image -> Nvshu character -> phonetic reading -> Hanzi candidate pool.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def normalize_reading(reading: str) -> str:
    """Normalize a dialect reading for grouping: lowercase letters only."""
    if not reading:
        return ""
    reading = reading.strip().lower().replace(" ", "")
    return "".join(ch for ch in reading if "a" <= ch <= "z")


def split_hanzi_candidates(value: str) -> list[str]:
    """Split the compact candidate string used by the data CSV into characters."""
    return [ch for ch in (value or "").strip() if ch and not ch.isspace()]


def char_unicode(ch: str) -> str:
    if not ch:
        return ""
    if len(ch) == 2 and 0xD800 <= ord(ch[0]) <= 0xDBFF and 0xDC00 <= ord(ch[1]) <= 0xDFFF:
        cp = 0x10000 + ((ord(ch[0]) - 0xD800) << 10) + (ord(ch[1]) - 0xDC00)
        return f"U+{cp:04X}"
    if len(ch) == 1:
        return f"U+{ord(ch):04X}"
    return " ".join(f"U+{ord(part):04X}" for part in ch)


def looks_like_nushu(ch: str) -> bool:
    if not ch:
        return False
    if len(ch) == 2 and 0xD800 <= ord(ch[0]) <= 0xDBFF and 0xDC00 <= ord(ch[1]) <= 0xDFFF:
        cp = 0x10000 + ((ord(ch[0]) - 0xD800) << 10) + (ord(ch[1]) - 0xDC00)
    elif len(ch) == 1:
        cp = ord(ch)
    else:
        return False
    return 0x1B170 <= cp <= 0x1B2FF


@dataclass(frozen=True)
class PhoneticEntry:
    nushu: str
    unicode: str
    raw_reading: str
    reading: str
    hanzi_candidates: tuple[str, ...]
    source_index: str = ""


class NvshuPhoneticLexicon:
    def __init__(self, entries: Iterable[PhoneticEntry] = ()):
        self.by_char: dict[str, PhoneticEntry] = {}
        self.by_reading: dict[str, list[PhoneticEntry]] = {}
        self.by_raw_reading: dict[str, list[PhoneticEntry]] = {}

        for entry in entries:
            if not entry.nushu:
                continue
            self.by_char[entry.nushu] = entry
            if entry.reading:
                self.by_reading.setdefault(entry.reading, []).append(entry)
            if entry.raw_reading:
                self.by_raw_reading.setdefault(entry.raw_reading, []).append(entry)

    @classmethod
    def from_csv(cls, csv_path: Path) -> "NvshuPhoneticLexicon":
        if not csv_path.exists():
            return cls()

        entries: list[PhoneticEntry] = []
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                nushu = (row.get("女书字符") or "").strip()
                if not nushu:
                    continue
                raw_reading = (row.get("江永方言代表发音") or "").strip()
                entries.append(
                    PhoneticEntry(
                        nushu=nushu,
                        unicode=char_unicode(nushu),
                        raw_reading=raw_reading,
                        reading=normalize_reading(raw_reading),
                        hanzi_candidates=tuple(split_hanzi_candidates(row.get("对应汉字") or "")),
                        source_index=(row.get("《字帖》序") or "").strip(),
                    )
                )
        return cls(entries)

    @classmethod
    def from_project_root(cls, root: Path) -> "NvshuPhoneticLexicon":
        candidates = [
            root / "githubdata" / "nvshudata" / "unicode_nushu-master" / "data.csv",
            root / "nvshu_synthetic_data" / "unicode_nushu-master" / "data.csv",
        ]
        for csv_path in candidates:
            if csv_path.exists():
                return cls.from_csv(csv_path)
        return cls()

    def entry_for_char(self, ch: str) -> PhoneticEntry | None:
        return self.by_char.get(ch)

    def hanzi_pool_for_reading(self, reading: str) -> list[str]:
        normalized = normalize_reading(reading)
        seen: set[str] = set()
        out: list[str] = []
        for entry in self.by_reading.get(normalized, []):
            for hanzi in entry.hanzi_candidates:
                if hanzi not in seen:
                    seen.add(hanzi)
                    out.append(hanzi)
        return out

    def token_detail(self, ch: str, *, topk: list[dict] | None = None) -> dict:
        entry = self.entry_for_char(ch)
        if entry is None:
            return {
                "nushu": ch,
                "unicode": char_unicode(ch) if ch else "",
                "raw_reading": "",
                "reading": "",
                "hanzi_candidates": [ch] if ch else [],
                "same_reading_pool": [ch] if ch else [],
                "source_index": "",
                "visual_topk": topk or [],
            }

        return {
            "nushu": entry.nushu,
            "unicode": entry.unicode,
            "raw_reading": entry.raw_reading,
            "reading": entry.reading,
            "hanzi_candidates": list(entry.hanzi_candidates),
            "same_reading_pool": self.hanzi_pool_for_reading(entry.raw_reading),
            "source_index": entry.source_index,
            "visual_topk": topk or [],
        }

    def analyze_text(self, text: str, *, topk_by_index: dict[int, list[dict]] | None = None) -> dict:
        topk_by_index = topk_by_index or {}
        tokens = [
            self.token_detail(ch, topk=topk_by_index.get(i, []))
            for i, ch in enumerate(text or "")
        ]
        return {
            "nushu": text or "",
            "raw_reading": [t["raw_reading"] for t in tokens],
            "reading": [t["reading"] for t in tokens],
            "tokens": tokens,
        }
