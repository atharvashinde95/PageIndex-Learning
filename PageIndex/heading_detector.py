from __future__ import annotations
import re
from typing import List, Dict, Tuple

TOC_TAG = "<<TOC_BLOCK>>"

LINE_START = re.compile(
    r"(?m)^\s*(?P<struct>\d{1,3}(?:\.\d{1,3}){0,6})\.\s+(?P<title>[^\n]+)$"
)
BULLET = re.compile(
    r"(?m)^\s*[-•]\s*(?P<struct>\d{1,3}(?:\.\d{1,3}){0,6})\.\s+(?P<title>[^\n]+)$"
)
INLINE = re.compile(
    r"(?m)^\s*(?P<struct>\d{1,3}(?:\.\d{1,3}){1,6})\.\s+(?P<title>[A-Z][^\n]*)$"
)

def _valid(s, depth=7):
    p = s.split(".")
    return len(p) <= depth and all(x.isdigit() for x in p)

def _norm(t):
    t = t.strip()
    t = re.split(r"[•\n]", t)[0].strip()
    t = re.sub(r"\s+", " ", t)
    return t[:150]

def detect_headings(pages: List[Tuple[int, str]]):
    seen = {}
    out = []

    for p, text in pages:
        if TOC_TAG in text:
            continue

        for m in LINE_START.finditer(text):
            s, t = m.group("struct"), m.group("title")
            if _valid(s) and s not in seen:
                seen[s] = p
                out.append({"structure": s, "title": _norm(t), "start_index": p})

        for m in BULLET.finditer(text):
            s, t = m.group("struct"), m.group("title")
            if _valid(s) and s not in seen:
                seen[s] = p
                out.append({"structure": s, "title": _norm(t), "start_index": p})

        for m in INLINE.finditer(text):
            s, t = m.group("struct"), m.group("title")
            if _valid(s) and s not in seen:
                seen[s] = p
                out.append({"structure": s, "title": _norm(t), "start_index": p})

    def key(h):
        return (h["start_index"], *(int(x) for x in h["structure"].split(".")))

    out.sort(key=key)
    return out
