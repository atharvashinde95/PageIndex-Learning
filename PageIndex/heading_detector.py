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

# ---------------------------------------------------------------
# BUG FIX: Detect whether a page is a TOC page.
# The original code marked a page with <<TOC_BLOCK>> only when the
# TOC block was found on page index 1 (page 2).  But the heading
# regexes still ran on that same page, so every TOC entry like
# "2.1. Problem definition" was registered as a real heading with
# start_index = 2 (the TOC page).  Those phantom nodes then get the
# TOC page text instead of their real section text.
#
# Fix: skip ALL heading detection on any page that contains a TOC block.
# ---------------------------------------------------------------

def _is_toc_page(text: str) -> bool:
    """Return True if this page is (or contains) a table-of-contents block."""
    return TOC_TAG in text


def detect_headings(pages: List[Tuple[int, str]]):
    seen = {}
    out = []

    # ---- Pass 1: collect headings, skipping TOC pages ----
    for p, text in pages:
        # Skip pages that were tagged as TOC by the PDF parser
        if _is_toc_page(text):
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
