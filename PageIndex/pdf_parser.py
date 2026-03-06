"""
core/pdf_parser.py
------------------
Extracts text from every PDF page using PyMuPDF (fitz).
Each page is wrapped with <physical_index_X> tags — exactly the
same convention used by the original PageIndex source code.

Page numbers are 1-based throughout this project.
"""

import logging
from pathlib import Path
from typing import List, Tuple

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


def extract_pages(pdf_path: str) -> List[Tuple[int, str]]:
    """
    Open a PDF and extract the text of every page.

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        List of (page_number, page_text) tuples.  Page numbers are 1-based.
        Pages with no extractable text return an empty string.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(path))
    pages: List[Tuple[int, str]] = []

    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        pages.append((i + 1, text))          # 1-based page number

    doc.close()
    logger.info("Extracted %d pages from '%s'", len(pages), path.name)
    return pages


def build_tagged_text(pages: List[Tuple[int, str]]) -> str:
    """
    Wrap each page in <physical_index_X> open/close tags.

    This is the same format the original PageIndex uses so that the LLM
    can reliably locate which physical page a section starts on.

    Example output for page 3:
        <physical_index_3>
        ... page text ...
        <physical_index_3>

    Args:
        pages: List of (page_number, page_text) from extract_pages().

    Returns:
        Single string with all pages tagged and concatenated.
    """
    parts = []
    for page_num, text in pages:
        tag = f"<physical_index_{page_num}>"
        parts.append(f"{tag}\n{text}\n{tag}\n")
    return "\n".join(parts)


def build_tagged_chunk(pages: List[Tuple[int, str]], start: int, end: int) -> str:
    """
    Build a tagged text string for a page range [start, end] inclusive.
    Both start and end are 1-based page numbers.
    """
    subset = [(pn, pt) for pn, pt in pages if start <= pn <= end]
    return build_tagged_text(subset)


def get_page_text(pages: List[Tuple[int, str]], start: int, end: int) -> str:
    """
    Return the raw (un-tagged) concatenated text for pages [start, end].
    Used when feeding content to the answer-generation LLM call.
    """
    parts = [pt for pn, pt in pages if start <= pn <= end]
    return "\n\n".join(parts)
