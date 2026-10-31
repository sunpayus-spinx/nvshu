"""
Project path helpers for nvshu.

Goal: make the project runnable after moving folders/drives (no hard-coded E:/code/nvshu).
All paths should be resolved relative to the project root unless explicitly absolute.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional, Union


PathLike = Union[str, os.PathLike]


def project_root() -> Path:
    """
    Return project root directory.

    Priority:
    1) NVSHU_ROOT environment variable (absolute or relative)
    2) This file's parent directory (repo root)
    """
    env = os.environ.get("NVSHU_ROOT", "").strip()
    if env:
        p = Path(env).expanduser()
        return p.resolve()
    return Path(__file__).resolve().parent


def resolve_path(p: Optional[PathLike], *, root: Optional[Path] = None) -> Optional[Path]:
    """Resolve a possibly-relative path against project root."""
    if p is None:
        return None
    s = str(p).strip()
    if not s:
        return None
    pp = Path(s).expanduser()
    if pp.is_absolute():
        return pp
    rr = (root or project_root()).resolve()
    return (rr / pp).resolve()


def resolve_paths(values: Iterable[PathLike], *, root: Optional[Path] = None) -> list[Path]:
    rr = root or project_root()
    out: list[Path] = []
    for v in values:
        rp = resolve_path(v, root=rr)
        if rp is not None:
            out.append(rp)
    return out

