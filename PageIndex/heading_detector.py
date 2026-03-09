from __future__ import annotations
import re
from typing import List, Dict, Tuple

TOC_TAG = "<<TOC_BLOCK>>"

# BUG FIX A: Use [ \t]+ instead of \s+ after the dot.
# \s+ matches newlines, so "1.\nSome activity text" was incorrectly matched as
# structure "1" with title "Some activity text" (numbered list items poisoning the tree).
# [ \t]+ only matches spaces/tabs on the same line, rejecting those false positives.
LINE_START = re.compile(
    r"(?m)^\s*(?P<struct>\d{1,3}(?:\.\d{1,3}){0,6})\.[ \t]+(?P<title>[^\n]+)$"
)
BULLET = re.compile(
    r"(?m)^\s*[-•][ \t]*(?P<struct>\d{1,3}(?:\.\d{1,3}){0,6})\.[ \t]+(?P<title>[^\n]+)$"
)
INLINE = re.compile(
    r"(?m)^\s*(?P<struct>\d{1,3}(?:\.\d{1,3}){1,6})\.[ \t]+(?P<title>[A-Z][^\n]*)$"
)

# BUG FIX B: Some PDFs format headings WITHOUT a trailing dot, e.g. "2.3  Targeted outputs..."
# The patterns above all require a dot after the structure number, so these are silently missed.
# This pattern catches sub-section headings (depth >= 2) without a dot, requiring:
#   - at least one dot in the structure (so "1  some text" is rejected — that's a list item)
#   - title starts with uppercase (filters out sentence fragments)
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

        # NO_DOT runs last so it only fills in structures not already captured above
        for m in NO_DOT.finditer(text):
            s, t = m.group("struct"), m.group("title")
            if _valid(s) and s not in seen:
                seen[s] = p
                out.append({"structure": s, "title": _norm(t), "start_index": p})

        # BUG FIX C: Some PDFs (especially those exported with line-breaks in headings)
        # put the structure number and title on SEPARATE lines, e.g.:
        #   "1.\n"
        #   "Executive summary"
        # None of the above patterns catch this. Detect it by finding a bare "N." line
        # and treating the very next non-empty line as its title.
        lines = text.split("\n")
        BARE_NUM = re.compile(r"^\s*(?P<struct>\d{1,3}(?:\.\d{1,3}){0,6})\.\s*$")
        for i, line in enumerate(lines):
            bm = BARE_NUM.match(line)
            if not bm:
                continue
            s = bm.group("struct")
            if not _valid(s) or s in seen:
                continue
            # Find next non-empty line as the title
            for j in range(i + 1, min(i + 3, len(lines))):
                t = lines[j].strip()
                if not t:
                    continue
                # Reject list-item sentence fragments:
                # real section titles are short (<=8 words) and don't read as
                # mid-sentence continuations (no leading lowercase, no commas in first word)
                words = t.split()
                is_title_like = (
                    len(words) <= 8
                    and words[0][0].isupper()
                    and not t.endswith(",")
                )
                if is_title_like:
                    seen[s] = p
                    out.append({"structure": s, "title": _norm(t), "start_index": p})
                break

    def key(h):
        return (h["start_index"], *(int(x) for x in h["structure"].split(".")))

    out.sort(key=key)
    return out
