from __future__ import annotations
import re
from typing import List, Dict, Tuple

TOC_TAG = "<<TOC_BLOCK>>"

# FIX A: Use [ \t]+ instead of \s+ after the dot.
# \s+ matches newlines, so "2.\nUtilization of wheat..." (a numbered list item)
# was read as structure "2" with activity text as its title, poisoning seen{}.
LINE_START = re.compile(
    r"(?m)^\s*(?P<struct>\d{1,3}(?:\.\d{1,3}){0,6})\.[ \t]+(?P<title>[^\n]+)$"
)
BULLET = re.compile(
    r"(?m)^\s*[-•][ \t]*(?P<struct>\d{1,3}(?:\.\d{1,3}){0,6})\.[ \t]+(?P<title>[^\n]+)$"
)
INLINE = re.compile(
    r"(?m)^\s*(?P<struct>\d{1,3}(?:\.\d{1,3}){1,6})\.[ \t]+(?P<title>[A-Z][^\n]*)$"
)

# FIX B: Some PDFs format sub-section headings WITHOUT a trailing dot,
# e.g. "2.3  Targeted outputs..." — requires depth>=2 (dot in struct) to
# avoid matching bare list items like "1  Some text".
NO_DOT = re.compile(
    r"(?m)^\s*(?P<struct>\d{1,3}(?:\.\d{1,3}){1,6})[ \t]+(?P<title>[A-Z][^\n]+)$"
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

        # NO_DOT runs last — only fills structs not already found above
        for m in NO_DOT.finditer(text):
            s, t = m.group("struct"), m.group("title")
            if _valid(s) and s not in seen:
                seen[s] = p
                out.append({"structure": s, "title": _norm(t), "start_index": p})

        # FIX C: Some PDFs split the heading across two lines: "1.\n" then "Executive summary"
        # None of the above patterns catch this split format.
        lines = text.split("\n")
        BARE_NUM = re.compile(r"^\s*(?P<struct>\d{1,3}(?:\.\d{1,3}){0,6})\.\s*$")
        for i, line in enumerate(lines):
            bm = BARE_NUM.match(line)
            if not bm:
                continue
            s = bm.group("struct")
            if not _valid(s) or s in seen:
                continue
            for j in range(i + 1, min(i + 3, len(lines))):
                t = lines[j].strip()
                if not t:
                    continue
                # Reject list-item sentence fragments: real titles are short and Title Case
                words = t.split()
                if len(words) <= 8 and words[0][0].isupper() and not t.endswith(","):
                    seen[s] = p
                    out.append({"structure": s, "title": _norm(t), "start_index": p})
                break

    def key(h):
        return (h["start_index"], *(int(x) for x in h["structure"].split(".")))

    out.sort(key=key)
    return out
