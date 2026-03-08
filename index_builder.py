import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import tiktoken

from core.llm_client import call_llm
from core.pdf_parser import build_tagged_chunk, build_tagged_text, get_page_text

logger = logging.getLogger(__name__)

TOC_CHECK_PAGES = 20
MAX_PAGES_PER_NODE = 10
MAX_TOKENS_PER_NODE = 20000
CHUNK_SIZE_PAGES = 15
ENCODING_NAME = "cl100k_base"

# ── Token counter ──────────────────────────────────────────────────────────────
_enc = tiktoken.get_encoding(ENCODING_NAME)
def count_tokens(text: str) -> int:
    return len(_enc.encode(text))


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 – TOC Detection
# ═══════════════════════════════════════════════════════════════════════════════
def detect_toc(pages: List[Tuple[int, str]], check_pages: int = TOC_CHECK_PAGES) -> Optional[str]:
    """
    Ask the LLM whether the first `check_pages` pages contain a Table of Contents.
    Returns the raw TOC text if found, else None.
    """
    tagged = build_tagged_chunk(pages, start=1, end=min(check_pages, len(pages)))
    prompt = f"""You are a document analysis expert.
        Below are the first pages of a PDF document, each wrapped in <physical_index_X> tags.
        Your task:
        1. Determine if the document contains an explicit Table of Contents (TOC) page.
        2. If YES, extract and return the FULL TOC text exactly as it appears.
        3. If NO, reply with exactly: NO_TOC
        A TOC typically lists section titles paired with page numbers, or just section titles
        in a structured list. Do NOT include the tagged page markers in your output.
        Document pages:
        {tagged}
        Reply with either the full TOC text, or NO_TOC."""
    result = call_llm(prompt, max_tokens=2048)
    if result.strip().upper().startswith("NO_TOC"):
        logger.info("TOC detection: no TOC found")
        return None
    logger.info("TOC detection: TOC found (%d chars)", len(result))
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2a – TOC with page numbers
# ═══════════════════════════════════════════════════════════════════════════════
def toc_transformer(toc_text: str) -> List[Dict]:
    """
    Convert raw TOC text into a structured JSON list.
    Each entry: { "title": str, "structure": str, "toc_page": int | null }
    "structure" uses the numeric hierarchy system: "1", "1.1", "1.1.2", etc.
    "toc_page" is the page number printed in the TOC (may differ from physical page).
    
    NOTE: This only captures entries explicitly listed in the TOC.
    Use scan_body_for_subsections() afterwards to find unlisted subsections.
    """
    prompt = f"""You are an expert at parsing document structure.
Transform the Table of Contents below into a JSON array.

Strict rules:
- The TOC may be a flat, single-level list for slides; if so, produce sequential numeric "structure"
  values starting from "1" with no gaps: "1", "2", "3", ...
- Preserve the original visual order of items as they appear.
- Each item must have:
  "title" : section title as a string
  "structure" : numeric hierarchy string e.g. "1", "1.1", "2", "2.3.1"
  "toc_page" : integer page number if visible in the TOC, else null
- Return ONLY the JSON array. No explanation, no markdown fences.

Table of Contents:
{toc_text}"""
    raw = call_llm(prompt, max_tokens=4096)
    raw = _strip_json_fences(raw)
    try:
        data = json.loads(raw)
        logger.info("toc_transformer: parsed %d entries", len(data))
        return data
    except json.JSONDecodeError as exc:
        logger.warning("toc_transformer JSON parse failed: %s — retrying", exc)
        return _retry_json_parse(prompt, raw)


def toc_index_extractor(
    toc_entries: List[Dict],
    pages: List[Tuple[int, str]],
    check_pages: int = TOC_CHECK_PAGES,
) -> List[Dict]:
    """
    Map each TOC entry to its physical page number using the <physical_index_X> tags.
    Adds "start_index" to each entry. Derives "end_index" from the next entry.
    """
    tagged = build_tagged_chunk(pages, start=1, end=min(check_pages + 20, len(pages)))
    prompt = f"""You are given a list of document sections and the tagged pages of a PDF.
Each page in the document is marked with <physical_index_X> tags.
Your job: for each section title, find the physical page number where that section STARTS.
Return a JSON array in the same order as the input.
Each object must have:
 "title" : same title string
 "structure" : same structure string
 "start_index" : integer physical page number where the section starts
Return ONLY the JSON array.
Sections:
{json.dumps([{"title": e["title"], "structure": e["structure"]} for e in toc_entries], indent=2)}
Document pages:
{tagged}"""
    raw = call_llm(prompt, max_tokens=4096)
    raw = _strip_json_fences(raw)
    try:
        mapped = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("toc_index_extractor JSON parse failed: %s — retrying", exc)
        mapped = _retry_json_parse(prompt, raw)

    # Derive end_index: each section ends one page before the next sibling starts
    _assign_end_indices(mapped, total_pages=len(pages))
    logger.info("toc_index_extractor: mapped %d sections", len(mapped))
    return mapped


# ═══════════════════════════════════════════════════════════════════════════════
# NEW: PHASE 2d – Scan body for subsections missing from TOC
# ═══════════════════════════════════════════════════════════════════════════════
def scan_body_for_subsections(
    top_level_sections: List[Dict],
    pages: List[Tuple[int, str]],
    chunk_size: int = CHUNK_SIZE_PAGES,
) -> List[Dict]:
    """
    After extracting top-level sections from TOC, scan the document body to
    find ALL subsections (e.g. 2.1, 2.1.1, 2.2, 3.1 ... 6.2.8) that the TOC
    may have omitted.

    Strategy:
    - For each top-level section, scan its page range in chunks
    - Ask the LLM to identify all numbered subsection headings within those pages
    - Merge subsections back into the flat list

    Args:
        top_level_sections: Output of toc_index_extractor() — flat list with start/end.
        pages: Full page list.

    Returns:
        Expanded flat list including all discovered subsections, sorted by start_index.
    """
    total = len(pages)
    all_sections = list(top_level_sections)  # copy

    for parent in top_level_sections:
        parent_struct = parent.get("structure", "")
        start = int(parent.get("start_index", 1))
        end = int(parent.get("end_index", total))

        # Skip trivially small sections (1 page — unlikely to have subsections)
        if end - start < 1:
            continue

        logger.info(
            "scan_body_for_subsections: scanning '%s' pages %d–%d",
            parent.get("title"), start, end,
        )

        discovered: List[Dict] = []

        for chunk_start in range(start, end + 1, chunk_size):
            chunk_end = min(chunk_start + chunk_size - 1, end)
            tagged = build_tagged_chunk(pages, chunk_start, chunk_end)

            prompt = f"""You are reading pages {chunk_start}–{chunk_end} of a document.
Each page is wrapped in <physical_index_X> tags.

The parent section is: "{parent.get('title')}" with structure ID "{parent_struct}".

Your task: Find ALL numbered subsection headings within these pages.
Subsection headings look like: "2.1 - Title", "2.1.1 - Title", "3.2 Title", "6.2.3 - Title" etc.
They always start with a numeric identifier like X.Y or X.Y.Z.

For each subsection found, return its physical start page (from the <physical_index_X> tags).

Rules:
- Only include subsections that are DIRECT or NESTED children of "{parent_struct}"
  e.g. if parent is "2", include "2.1", "2.1.1", "2.2", "2.2.1", "2.2.2" etc.
- Do NOT include the parent section itself.
- Do NOT include sections from other top-level chapters.
- If no subsections are found, return an empty array [].
- Return ONLY a JSON array. Each object:
  "title"       : subsection heading text (without the numeric prefix)
  "structure"   : full numeric structure ID e.g. "2.1", "2.1.1"
  "start_index" : integer physical page number

Return ONLY the JSON array. No explanation, no markdown fences.

Document pages:
{tagged}"""

            raw = call_llm(prompt, max_tokens=2048)
            raw = _strip_json_fences(raw)
            try:
                found = json.loads(raw)
                if isinstance(found, list):
                    # Deduplicate by structure
                    existing_structs = {s.get("structure") for s in discovered}
                    for item in found:
                        if item.get("structure") and item["structure"] not in existing_structs:
                            discovered.append(item)
                            existing_structs.add(item["structure"])
            except json.JSONDecodeError:
                logger.warning(
                    "scan_body_for_subsections: JSON parse failed for chunk %d–%d",
                    chunk_start, chunk_end,
                )

        if discovered:
            logger.info(
                "scan_body_for_subsections: found %d subsections under '%s'",
                len(discovered), parent.get("title"),
            )
            all_sections.extend(discovered)

    # Sort everything by start_index before returning
    valid = [s for s in all_sections if int(s.get("start_index", 0)) > 0]
    valid.sort(key=lambda s: int(s["start_index"]))

    logger.info(
        "scan_body_for_subsections: total sections after merge = %d (was %d)",
        len(valid), len(top_level_sections),
    )
    return valid


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2b – TOC without page numbers
# ═══════════════════════════════════════════════════════════════════════════════
def add_page_number_to_toc(
    toc_entries: List[Dict],
    pages: List[Tuple[int, str]],
    chunk_size: int = CHUNK_SIZE_PAGES,
) -> List[Dict]:
    """
    When the TOC has no page numbers, scan the document in chunks and ask
    the LLM to match each section title to its physical start page.
    """
    total = len(pages)
    unresolved = {e["structure"]: e["title"] for e in toc_entries}
    resolved: Dict[str, int] = {}

    for chunk_start in range(1, total + 1, chunk_size):
        chunk_end = min(chunk_start + chunk_size - 1, total)
        tagged = build_tagged_chunk(pages, chunk_start, chunk_end)
        still_needed = {s: t for s, t in unresolved.items() if s not in resolved}
        if not still_needed:
            break

        prompt = f"""You are reading pages {chunk_start}–{chunk_end} of a PDF document.
Each page is wrapped in <physical_index_X> tags.
For each section title below, check if it STARTS on one of these pages.
If a title starts in this range, record its physical page number.
If it does not appear in this range, skip it.
Return a JSON object mapping structure_id → start_page_number.
Only include sections found in this chunk. Return ONLY the JSON object.
Sections to find:
{json.dumps(still_needed, indent=2)}
Document pages:
{tagged}"""
        raw = call_llm(prompt, max_tokens=2048)
        raw = _strip_json_fences(raw)
        try:
            found = json.loads(raw)
            resolved.update({str(k): int(v) for k, v in found.items()})
        except (json.JSONDecodeError, ValueError):
            logger.warning("add_page_number_to_toc: parse failed for chunk %d–%d", chunk_start, chunk_end)

    result = []
    for entry in toc_entries:
        sid = entry["structure"]
        result.append({
            "title": entry["title"],
            "structure": sid,
            "start_index": resolved.get(sid, 0),
        })

    _assign_end_indices(result, total_pages=len(pages))
    logger.info("add_page_number_to_toc: resolved %d/%d entries", len(resolved), len(toc_entries))
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2c – No TOC: generate structure from body text
# ═══════════════════════════════════════════════════════════════════════════════
def generate_toc_from_content(
    pages: List[Tuple[int, str]],
    chunk_size: int = CHUNK_SIZE_PAGES,
) -> List[Dict]:
    """
    When no TOC exists, build the hierarchy from scratch by scanning the full document.
    Finds ALL section headings including subsections like 2.1, 2.1.1, 3.2.4 etc.
    """
    total = len(pages)
    accumulated: List[Dict] = []

    for chunk_start in range(1, total + 1, chunk_size):
        chunk_end = min(chunk_start + chunk_size - 1, total)
        tagged = build_tagged_chunk(pages, chunk_start, chunk_end)
        is_first_chunk = (chunk_start == 1)

        if is_first_chunk:
            prompt = f"""You are an expert in extracting hierarchical tree structure from documents.
Your task: generate a structured table of contents for the document pages below.
Rules:
- Find ALL section headings including subsections (e.g. 1, 1.1, 1.1.1, 2, 2.1, 2.1.2 etc.)
- Use the numeric structure system: "1", "1.1", "1.2", "2", "2.1", etc.
- For each section, provide the physical start page using the <physical_index_X> tags.
- Return a JSON array. Each object:
  "title" : section title string (without the number prefix)
  "structure" : numeric hierarchy string
  "start_index" : integer physical page number where this section starts
- Return ONLY the JSON array. No explanation, no markdown fences.
Document pages:
{tagged}"""
        else:
            prompt = f"""You are continuing to extract the hierarchical tree structure of a document.
Below are the NEXT pages of the document (pages {chunk_start}–{chunk_end}).
Continue extending the structure — do NOT repeat sections already found.
Find ALL section headings including subsections (e.g. 2.1, 2.1.1, 3.2, etc.)
Existing structure so far:
{json.dumps(accumulated[-20:], indent=2)}
Rules:
- Continue the same numbering scheme.
- Return ONLY the NEW sections found in this chunk as a JSON array.
- Each object: "title", "structure", "start_index"
- Return ONLY the JSON array.
Document pages:
{tagged}"""

        raw = call_llm(prompt, max_tokens=4096)
        raw = _strip_json_fences(raw)
        try:
            chunk_entries = json.loads(raw)
            if isinstance(chunk_entries, list):
                accumulated.extend(chunk_entries)
        except json.JSONDecodeError:
            logger.warning("generate_toc_from_content: JSON parse failed for chunk %d–%d", chunk_start, chunk_end)

    _assign_end_indices(accumulated, total_pages=len(pages))
    logger.info("generate_toc_from_content: generated %d sections", len(accumulated))
    return accumulated


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3 – Verification
# ═══════════════════════════════════════════════════════════════════════════════
def verify_sections(
    sections: List[Dict],
    pages: List[Tuple[int, str]],
) -> Tuple[List[Dict], List[Dict]]:
    """
    For each section, verify its title appears in the text of its start_index page.
    Returns (correct, incorrect) lists.
    """
    page_map = {pn: pt for pn, pt in pages}
    correct, incorrect = [], []

    for sec in sections:
        title = sec.get("title", "")
        start = sec.get("start_index", 0)

        # ── FIX: guard against invalid start_index ──
        if not start or start <= 0 or start > len(pages):
            logger.warning("verify_sections: invalid start_index %s for '%s'", start, title)
            incorrect.append(sec)
            continue

        page_text = page_map.get(start, "")

        # Case-insensitive substring check (lenient — first 6 words of title)
        probe = " ".join(title.split()[:6]).lower()
        if probe and probe in page_text.lower():
            correct.append(sec)
        else:
            incorrect.append(sec)

    accuracy = len(correct) / len(sections) if sections else 1.0
    logger.info("verify_sections: accuracy=%.1f%% (%d/%d correct)",
                accuracy * 100, len(correct), len(sections))
    return correct, incorrect


def fix_incorrect_sections(
    incorrect: List[Dict],
    pages: List[Tuple[int, str]],
    max_retries: int = 3,
) -> List[Dict]:
    """
    Re-ask the LLM to correct the page assignments for sections that failed verification.
    """
    for attempt in range(1, max_retries + 1):
        if not incorrect:
            break

        titles_needed = [{"title": s["title"], "structure": s["structure"]} for s in incorrect]
        tagged = build_tagged_text(pages)

        prompt = f"""The following sections were assigned wrong start pages.
Re-examine the full document and find the CORRECT physical start page for each.
Return a JSON array. Each object:
 "title" : same title
 "structure" : same structure
 "start_index" : corrected integer page number
Return ONLY the JSON array.
Incorrect sections:
{json.dumps(titles_needed, indent=2)}
Document (all pages):
{tagged[:12000]}"""

        raw = call_llm(prompt, max_tokens=2048)
        raw = _strip_json_fences(raw)
        try:
            fixed = json.loads(raw)
            logger.info("fix_incorrect_sections attempt %d: got %d fixes", attempt, len(fixed))
            incorrect = []
            return fixed
        except json.JSONDecodeError:
            logger.warning("fix_incorrect_sections attempt %d: JSON parse failed", attempt)

    return incorrect


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4 – Recursive split for oversized nodes
# ═══════════════════════════════════════════════════════════════════════════════
def split_large_nodes(
    sections: List[Dict],
    pages: List[Tuple[int, str]],
    max_pages: int = MAX_PAGES_PER_NODE,
    max_tokens: int = MAX_TOKENS_PER_NODE,
) -> List[Dict]:
    """
    If a node's page range exceeds both max_pages AND max_tokens,
    recursively re-run generate_toc_from_content on just those pages
    and replace the node with its children.
    """
    result = []
    for sec in sections:
        start = sec.get("start_index", 1)
        end = sec.get("end_index", start)

        # ── FIX: guard against inverted ranges ──
        if end < start:
            logger.warning("split_large_nodes: fixing inverted range for '%s' (%d > %d)", sec.get("title"), start, end)
            end = start
            sec["end_index"] = end

        page_span = end - start + 1
        node_text = get_page_text(pages, start, end)
        token_count = count_tokens(node_text)

        if page_span > max_pages and token_count > max_tokens:
            logger.info(
                "split_large_nodes: splitting '%s' (%d pages, %d tokens)",
                sec["title"], page_span, token_count,
            )
            sub_pages = [(pn, pt) for pn, pt in pages if start <= pn <= end]
            sub_sections = generate_toc_from_content(sub_pages, chunk_size=max_pages)

            parent_struct = sec["structure"]
            for i, sub in enumerate(sub_sections, start=1):
                sub["structure"] = f"{parent_struct}.{i}"

            result.extend(split_large_nodes(sub_sections, pages, max_pages, max_tokens))
        else:
            result.append(sec)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5 – post_processing: flat list → nested tree + assign node_id
# ═══════════════════════════════════════════════════════════════════════════════
def list_to_tree(sections: List[Dict]) -> List[Dict]:
    """
    Convert a flat list of sections (with "structure" like "1", "1.1", "2.3.1")
    into a nested tree using a stack-based algorithm.
    """
    root_nodes: List[Dict] = []
    stack: List[Tuple[Dict, int]] = []

    for sec in sections:
        sec.setdefault("children", [])
        depth = len(sec["structure"].split("."))

        while stack and stack[-1][1] >= depth:
            stack.pop()

        if stack:
            stack[-1][0]["children"].append(sec)
        else:
            root_nodes.append(sec)

        stack.append((sec, depth))

    return root_nodes


def assign_node_ids(tree: List[Dict], counter: List[int] = None) -> None:
    """
    Depth-first traversal assigning zero-padded 4-digit node IDs.
    Mutates the tree in-place.
    """
    if counter is None:
        counter = [0]
    for node in tree:
        node["node_id"] = str(counter[0]).zfill(4)
        counter[0] += 1
        assign_node_ids(node.get("children", []), counter)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 6 – Enrichment
# ═══════════════════════════════════════════════════════════════════════════════
def add_node_text(tree: List[Dict], pages: List[Tuple[int, str]]) -> None:
    """Attach raw page text to each node using its start_index → end_index range."""
    for node in tree:
        start = node.get("start_index", 0)
        end = node.get("end_index", start)
        if start and end and start <= end:
            node["text"] = get_page_text(pages, start, end)
        else:
            node["text"] = ""
        add_node_text(node.get("children", []), pages)


def add_node_summary(tree: List[Dict]) -> None:
    """Generate a short LLM summary for each node from its text."""
    for node in tree:
        text = node.get("text", "")
        if text:
            prompt = f"""Summarize the following document section in 2–3 sentences.
Be specific: mention key topics, numbers, or entities. Be concise.
Section title: {node.get('title', '')}
Section text:
{text[:3000]}
Return only the summary text."""
            node["summary"] = call_llm(prompt, max_tokens=256)
        else:
            node["summary"] = ""
        add_node_summary(node.get("children", []))


def add_doc_description(root_tree: List[Dict], pages: List[Tuple[int, str]]) -> str:
    """Generate a top-level description of the entire document."""
    first_pages_text = get_page_text(pages, 1, min(5, len(pages)))
    top_titles = [n.get("title", "") for n in root_tree[:10]]
    prompt = f"""You are reading a document. Based on the first few pages and the top-level
section titles, write a 2–3 sentence description of what this document is about.
Top-level sections:
{json.dumps(top_titles, indent=2)}
First pages of the document:
{first_pages_text[:2000]}
Return only the description text."""
    return call_llm(prompt, max_tokens=256)


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════
def build_index(
    pages: List[Tuple[int, str]],
    toc_check_pages: int = TOC_CHECK_PAGES,
    max_pages_per_node: int = MAX_PAGES_PER_NODE,
    max_tokens_per_node: int = MAX_TOKENS_PER_NODE,
    add_summaries: bool = True,
    add_text: bool = True,
    add_description: bool = True,
) -> Dict[str, Any]:
    """
    Full PageIndex-equivalent pipeline. Returns the complete tree as a dict.

    Steps:
     1  Detect TOC
     2  Extract top-level sections (3 strategies with fallback)
     2d [NEW] Scan document body for ALL subsections missing from TOC
     3  Verify + fix page assignments
     4  Split oversized nodes
     5  Build nested tree + assign node_ids
     6  Enrich: text, summaries, doc description

    Key fix: PDFs often have a TOC that only lists chapter-level headings (1, 2, 3 ...),
    while the body contains many subsections (2.1, 2.1.1, 3.2 etc.) that the TOC omits.
    scan_body_for_subsections() fills this gap by scanning each chapter's page range.
    """
    total_pages = len(pages)
    logger.info("build_index: starting on %d pages", total_pages)

    # ── Step 1: TOC Detection ──────────────────────────────────────────────────
    toc_text = detect_toc(pages, check_pages=toc_check_pages)

    # ── Step 2: Section Extraction ─────────────────────────────────────────────
    if toc_text:
        toc_entries = toc_transformer(toc_text)
        has_page_numbers = any(e.get("toc_page") for e in toc_entries)
        if has_page_numbers:
            logger.info("build_index: strategy = TOC with page numbers")
            sections = toc_index_extractor(toc_entries, pages, check_pages=toc_check_pages)
        else:
            logger.info("build_index: strategy = TOC without page numbers")
            sections = add_page_number_to_toc(toc_entries, pages)

        # ── Step 2d [NEW]: Scan body for subsections not in TOC ───────────────
        # This is critical: most TOCs only list top-level chapters, but the body
        # contains many subsections (2.1, 2.2.1, 3.4.2, etc.) that we need for
        # accurate Q&A retrieval.
        logger.info("build_index: scanning body for subsections missing from TOC")
        sections = scan_body_for_subsections(sections, pages)

    else:
        logger.info("build_index: strategy = no TOC → generate from content")
        # generate_toc_from_content already scans for all subsections
        sections = generate_toc_from_content(pages)

    # ── Universal fix: filter and sort by start_index ─────────────────────────
    sections = [s for s in sections if int(s.get("start_index", 0)) > 0]
    sections.sort(key=lambda s: int(s["start_index"]))

    # ── Deduplicate by structure ID (keep first occurrence) ───────────────────
    seen_structures: set = set()
    deduped: List[Dict] = []
    for s in sections:
        struct = s.get("structure", "")
        if struct and struct not in seen_structures:
            deduped.append(s)
            seen_structures.add(struct)
    sections = deduped
    logger.info("build_index: %d sections after dedup", len(sections))

    # ── Step 3: Verification + Fix ─────────────────────────────────────────────
    correct, incorrect = verify_sections(sections, pages)
    if incorrect:
        logger.info("build_index: %d sections failed verification, attempting fix", len(incorrect))
        fixed = fix_incorrect_sections(incorrect, pages)
        sections = correct + fixed

    # Re-sort by start_index after any fixes
    sections = [s for s in sections if int(s.get("start_index", 0)) > 0]
    sections.sort(key=lambda s: int(s["start_index"]))

    # ── Step 4: Split Oversized Nodes ──────────────────────────────────────────
    sections = split_large_nodes(sections, pages, max_pages_per_node, max_tokens_per_node)

    # Re-sort again after splits
    sections = [s for s in sections if int(s.get("start_index", 0)) > 0]
    sections.sort(key=lambda s: int(s["start_index"]))

    # ── FIX: Enforce clean, non-overlapping end boundaries ────────────────────
    # Original code had a bug where end_index could be less than start_index.
    # We now compute end_index correctly by looking at the next section's start.
    for i in range(len(sections)):
        start_i = int(sections[i]["start_index"])
        if i + 1 < len(sections):
            next_start = int(sections[i + 1]["start_index"])
            # end is one page before next section starts, but never before its own start
            sections[i]["end_index"] = max(start_i, next_start - 1)
        else:
            sections[i]["end_index"] = total_pages

    # ── Step 5: Build Nested Tree ──────────────────────────────────────────────
    tree = list_to_tree(sections)
    assign_node_ids(tree)

    # ── Step 6: Enrichment ─────────────────────────────────────────────────────
    if add_text:
        add_node_text(tree, pages)
    if add_summaries:
        add_node_summary(tree)

    doc_description = ""
    if add_description:
        doc_description = add_doc_description(tree, pages)

    result = {
        "doc_description": doc_description,
        "total_pages": total_pages,
        "children": tree,
    }

    # Log final tree stats
    def _count(nodes):
        c = len(nodes)
        for n in nodes:
            c += _count(n.get("children", []))
        return c

    logger.info(
        "build_index: complete — %d top-level nodes, %d total nodes",
        len(tree), _count(tree),
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════
def _strip_json_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers if present."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _retry_json_parse(original_prompt: str, bad_output: str) -> Any:
    """Ask the LLM to fix its own malformed JSON."""
    fix_prompt = f"""The following text was supposed to be valid JSON but is not.
Fix it and return ONLY the corrected JSON. No explanation.
Broken output:
{bad_output}"""
    raw = call_llm(fix_prompt, max_tokens=4096)
    raw = _strip_json_fences(raw)
    return json.loads(raw)


def _assign_end_indices(sections: List[Dict], total_pages: int) -> None:
    """
    Mutate sections in-place to add end_index.
    Each section ends one page before the next section starts.
    The last section ends at total_pages.

    FIX: Ensures end_index is never less than start_index.
    """
    for i, sec in enumerate(sections):
        start_i = int(sec.get("start_index", 1))
        if i + 1 < len(sections):
            next_start = int(sections[i + 1].get("start_index", total_pages))
            # end must be >= start (guard against bad LLM output)
            sec["end_index"] = max(start_i, int(next_start) - 1)
        else:
            sec["end_index"] = total_pages
