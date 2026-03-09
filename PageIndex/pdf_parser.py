import re
import fitz
from pathlib import Path
from typing import List, Tuple

TOC_BLOCK = re.compile(r"(?m)^\s*\d{1,3}(?:\.\d{1,3}){0,4}\.\s+[A-Za-z].+$")


def extract_pages(pdf_path: str) -> List[Tuple[int, str]]:
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    pages = []

    # FIX 1: TOC pages only appear early in a document.
    # Limit TOC detection to the first 30% of pages (minimum 3).
    # The old code used "if i == 1" (only page 2), missing TOCs on other early pages.
    # The previous fix removed that guard entirely — but then late content pages
    # that happen to have 3+ section headings (e.g. page 11 with 2.5, 2.6, 2.6.1)
    # were wrongly tagged as TOC, stripping those sections entirely.
    toc_window = max(3, int(total_pages * 0.30))

    for i, page in enumerate(doc):
        page_num = i + 1  # 1-indexed
        blocks = page.get_text("blocks")
        blocks.sort(key=lambda b: (b[1], b[0]))

        # FIX 2: Pre-scan ALL lines across ALL blocks on the page before deciding.
        # The old code checked per-block with threshold >= 3. When each TOC entry
        # is its own PDF block (very common), no single block reaches 3 hits, so
        # <<TOC_BLOCK>> was never written and the TOC page was indexed as content.
        all_page_lines = []
        for b in blocks:
            text = (b[4] or "").strip()
            if text:
                all_page_lines.extend(text.split("\n"))

        page_toc_hits = [l for l in all_page_lines if TOC_BLOCK.match(l)]
        is_toc_page = (page_num <= toc_window) and (len(page_toc_hits) >= 3)

        lines = []
        if is_toc_page:
            lines.append("<<TOC_BLOCK>>")
        else:
            for b in blocks:
                text = (b[4] or "").strip()
                if not text:
                    continue
                for l in text.split("\n"):
                    if l not in lines:
                        lines.append(l)

        pages.append((page_num, "\n".join(lines)))

    doc.close()
    return pages


def get_page_text(pages, s, e):
    return "\n\n".join(text for pn, text in pages if s <= pn <= e)
