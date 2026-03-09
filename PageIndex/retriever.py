import re
import fitz
from pathlib import Path
from typing import List, Tuple

# Matches lines like "2.1.  Problem definition  ...... 5" in TOC
TOC_BLOCK = re.compile(r"(?m)^\s*\d{1,3}(?:\.\d{1,3}){0,4}\.\s+[A-Za-z].+$")

# Minimum number of TOC-like lines on a page to consider it a TOC page
TOC_LINE_THRESHOLD = 3


def extract_pages(pdf_path: str) -> List[Tuple[int, str]]:
    doc = fitz.open(pdf_path)
    pages = []

    for i, page in enumerate(doc):
        blocks = page.get_text("blocks")
        blocks.sort(key=lambda b: (b[1], b[0]))

        lines = []
        is_toc_page = False

        for b in blocks:
            text = (b[4] or "").strip()
            if not text:
                continue
            split = text.split("\n")

            # BUG FIX: detect TOC blocks on ANY page (not just page index 1).
            # Original code only tagged page i==1 with <<TOC_BLOCK>>, but the
            # heading regexes still ran and picked up TOC entries.  Now we
            # tag the whole page as a TOC page so heading_detector skips it.
            toc_hits = [l for l in split if TOC_BLOCK.match(l)]
            if len(toc_hits) >= TOC_LINE_THRESHOLD:
                is_toc_page = True
                # Don't add TOC block lines – they would confuse the heading detector
                continue

            for l in split:
                if l not in lines:
                    lines.append(l)

        # BUG FIX: Prepend the TOC marker so heading_detector.py can reliably
        # detect and skip the whole page, regardless of page number.
        if is_toc_page:
            page_text = "<<TOC_BLOCK>>\n" + "\n".join(lines)
        else:
            page_text = "\n".join(lines)

        pages.append((i + 1, page_text))

    doc.close()
    return pages


def get_page_text(pages, s, e):
    """Return concatenated text for pages in range [s, e] (inclusive).
    Strips the TOC marker from any page that has it so it doesn't leak
    into section text.
    """
    parts = []
    for pn, text in pages:
        if s <= pn <= e:
            # Strip internal TOC marker if somehow present
            clean = text.replace("<<TOC_BLOCK>>", "").strip()
            if clean:
                parts.append(clean)
    return "\n\n".join(parts)
