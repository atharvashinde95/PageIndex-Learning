import json
import logging
from typing import Any, Dict, List, Tuple

from core.heading_detector import detect_headings
from core.llm_client import call_llm
from core.pdf_parser import get_page_text

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2
GENERATOR = "clause-indexer"


# -----------------------------------------------------
# Helper: section depth
# -----------------------------------------------------
def _depth(s):
    return len(s.split("."))


# -----------------------------------------------------
# Helper: get parent "2.3.1" -> "2.3"
# -----------------------------------------------------
def _parent(s):
    p = s.split(".")
    return ".".join(p[:-1]) if len(p) > 1 else ""


# -----------------------------------------------------
# Set end_page boundaries for sections
# -----------------------------------------------------
def _assign_end(sections, total_pages):
    n = len(sections)
    starts = [s["start_index"] for s in sections]
    depths = [_depth(s["structure"]) for s in sections]

    for i in range(n):
        si = starts[i]
        di = depths[i]
        end = total_pages

        for j in range(i + 1, n):
            if depths[j] <= di:
                end = max(si, starts[j] - 1)
                break

        sections[i]["end_index"] = end


# -----------------------------------------------------
# Convert flat list into nested tree
# -----------------------------------------------------
def _tree(sections):
    index = {s["structure"]: s for s in sections}
    for s in sections:
        s.setdefault("children", [])

    roots = []

    for s in sections:
        p = _parent(s["structure"])

        while p and p not in index:
            p = _parent(p)

        if p:
            index[p]["children"].append(s)
        else:
            roots.append(s)

    return roots


# -----------------------------------------------------
# Assign sequential node IDs (fixes your Node ID issue)
# -----------------------------------------------------
def _ids(tree):
    counter = [0]

    def dfs(nodes):
        for n in nodes:
            n["node_id"] = str(counter[0]).zfill(4)
            counter[0] += 1
            dfs(n["children"])

    dfs(tree)


# -----------------------------------------------------
# LLM Summaries (section-level)
# -----------------------------------------------------
def _summarize_section(title, raw_text, structure):
    # Trim raw text so the prompt stays within a safe context window.
    # 6000 chars ≈ ~1500 tokens of input, comfortably within any model limit.
    trimmed = raw_text[:6000]

    prompt = f"""You are building a search index. Your job is to write a rich, dense summary
of the document section below so that a retrieval system can match user queries to it.

Rules:
1. Use ONLY information from the provided text — do NOT invent anything.
2. Write 5-7 bullet points that together cover the FULL content of the section.
3. Every bullet must be a complete, informative sentence (not a vague label).
4. Explicitly include: key topics, named entities (organizations, countries, people),
   acronyms with their expansions, specific numbers/metrics, and technical terms.
   These are the exact words users will search for.
5. If a bullet covers a sub-topic, name it clearly (e.g. "Regarding drought stress: ...").

Section Title: {title}
Section Number: {structure}

Text:
{trimmed}

Summary (bullet points only, no preamble):
"""
    try:
        # Raised from 200 → 500 tokens so summaries are complete, not truncated.
        return call_llm(prompt, max_tokens=500)
    except Exception:
        return ""


# -----------------------------------------------------
# Document-level description
# -----------------------------------------------------
def _doc_description(pages):
    full = "\n".join([text for _, text in pages])[:8000]

    prompt = f"""
Provide a concise 5–8 line high-level summary of the entire document.
Use ONLY the content provided.

Text:
{full}

Summary:
"""
    try:
        return call_llm(prompt, max_tokens=250)
    except Exception:
        return ""


# -----------------------------------------------------
# Extract text + generate summaries recursively
# -----------------------------------------------------
def _own_text(node: Dict, pages) -> str:
    """Return only the text that belongs to this node itself,
    EXCLUDING pages already covered by its children.

    Without this, a parent node like section 2 (pages 6-11) gets summarised
    with ALL of its children's content, producing a bloated summary that
    the LLM truncates badly. Each node should summarise only what it owns.
    """
    s, e = node["start_index"], node["end_index"]
    children = node.get("children", [])

    if not children:
        # Leaf node: owns all its pages
        return get_page_text(pages, s, e)

    # Collect page numbers claimed by any child
    child_pages: set = set()
    for c in children:
        cs, ce = c.get("start_index", s), c.get("end_index", e)
        child_pages.update(range(cs, ce + 1))

    # Keep only pages NOT covered by a child
    own_parts = [
        text for pn, text in pages
        if s <= pn <= e and pn not in child_pages
    ]
    return "\n\n".join(own_parts).strip()


def _text_and_summaries(tree, pages, add_summaries):
    for n in tree:
        s, e = n["start_index"], n["end_index"]

        # Full text for the node (used by retriever for answer generation)
        raw = get_page_text(pages, s, e)
        n["text"] = raw

        # Summary uses only the node's OWN text (not children's pages)
        # so each node's summary is focused and accurate for retrieval scoring.
        if add_summaries:
            own = _own_text(n, pages)
            # Fall back to full text if node has no exclusive pages
            # (e.g. a parent whose children cover all its pages)
            summarise_with = own if len(own) > 100 else raw
            n["summary"] = _summarize_section(
                n.get("title", ""),
                summarise_with,
                n.get("structure", "")
            )
        else:
            n["summary"] = ""

        _text_and_summaries(n["children"], pages, add_summaries)


# -----------------------------------------------------
# MAIN INDEX BUILDER
# -----------------------------------------------------
def build_index(pages, add_summaries=True, add_text=True, add_description=True):

    # Step 1: Detect numbered headings
    heads = detect_headings(pages)
    if not heads:
        raise RuntimeError("No headings detected")

    heads.sort(
        key=lambda h: (
            h["start_index"],
            *(int(x) for x in h["structure"].split(".")),
        )
    )

    # Step 2: Assign end-page
    _assign_end(heads, len(pages))

    # Step 3: Convert to hierarchical tree
    tree = _tree(heads)

    # Step 4: Assign Node IDs  (FIXED — This was missing)
    _ids(tree)

    # Step 5: Extract text + generate summaries
    if add_text:
        _text_and_summaries(tree, pages, add_summaries)

    # Step 6: Document description
    desc = _doc_description(pages) if add_description else ""

    return {
        "schema_version": SCHEMA_VERSION,
        "generator": GENERATOR,
        "doc_description": desc,
        "total_pages": len(pages),
        "children": tree,
    }
