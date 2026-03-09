import re
import fitz
from pathlib import Path
from typing import List, Tuple

TOC_BLOCK = re.compile(r"(?m)^\s*\d{1,3}(?:\.\d{1,3}){0,4}\.\s+[A-Za-z].+$")


def extract_pages(pdf_path: str) -> List[Tuple[int, str]]:
    doc = fitz.open(pdf_path)
    pages = []

    for i, page in enumerate(doc):
        blocks = page.get_text("blocks")
        blocks.sort(key=lambda b: (b[1], b[0]))

        # FIX: Pre-scan ALL lines across ALL blocks on this page first.
        # The original code checked TOC hits per-block with a threshold of 3.
        # When each TOC entry is its own single-line PDF block (very common),
        # no individual block ever reaches 3 hits, so <<TOC_BLOCK>> is never
        # added and heading_detector processes the TOC page as real content,
        # creating ghost nodes pointing to the TOC page.
        # Solution: count TOC-like lines across the ENTIRE page before processing.
        all_page_lines = []
        for b in blocks:
            text = (b[4] or "").strip()
            if text:
                all_page_lines.extend(text.split("\n"))

        page_toc_hits = [l for l in all_page_lines if TOC_BLOCK.match(l)]
        is_toc_page = len(page_toc_hits) >= 3

        lines = []
        if is_toc_page:
            # Mark the entire page as TOC so heading_detector skips it completely.
            # Removed the "if i == 1" guard — TOC can appear on any page.
            lines.append("<<TOC_BLOCK>>")
        else:
            for b in blocks:
                text = (b[4] or "").strip()
                if not text:
                    continue
                for l in text.split("\n"):
                    if l not in lines:
                        lines.append(l)

        pages.append((i + 1, "\n".join(lines)))

    doc.close()
    return pages


def get_page_text(pages, s, e):
    return "\n\n".join(text for pn, text in pages if s <= pn <= e)
