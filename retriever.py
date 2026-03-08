import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from core.llm_client import call_llm
from core.pdf_parser import get_page_text

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
#  Node helpers
# ═════════════════════════════════════════════════════════════════════════════

def _node_summary_block(nodes: List[Dict]) -> str:
    """
    Format a list of nodes as a compact summary block for the LLM.
    Only includes node_id, title, structure, start/end pages, and summary.
    """
    lines = []
    for node in nodes:
        line = (
            f"[node_id={node.get('node_id','?')}] "
            f"[structure={node.get('structure','?')}] "
            f"[pages {node.get('start_index','?')}–{node.get('end_index','?')}] "
            f"{node.get('title','Untitled')}"
        )
        summary = node.get("summary", "")
        if summary:
            line += f"\n  Summary: {summary}"
        lines.append(line)
    return "\n\n".join(lines)


def _find_node_by_id(tree: List[Dict], target_id: str) -> Optional[Dict]:
    """Depth-first search for a node by its node_id."""
    for node in tree:
        if node.get("node_id") == target_id:
            return node
        found = _find_node_by_id(node.get("children", []), target_id)
        if found:
            return found
    return None


def _flatten_tree(nodes: List[Dict]) -> List[Dict]:
    """Return a flat list of ALL nodes in the tree (DFS order)."""
    result = []
    for node in nodes:
        result.append(node)
        result.extend(_flatten_tree(node.get("children", [])))
    return result


# ═════════════════════════════════════════════════════════════════════════════
#  Core navigation step
# ═════════════════════════════════════════════════════════════════════════════

def _navigate_step(query: str, candidates: List[Dict]) -> Optional[str]:
    """
    Given a query and a list of candidate nodes at the same level,
    ask the LLM which node_id is most likely to contain the answer.

    Returns the chosen node_id, or None if no relevant node found.
    """
    summary_block = _node_summary_block(candidates)

    prompt = f"""You are navigating a document tree to find the answer to a query.

Query: "{query}"

Below are document sections at the current level. Each has a node_id, title, page range, and summary.

{summary_block}

Your task:
- Identify which single section is MOST LIKELY to contain the answer to the query.
- Reply with ONLY the node_id of that section (e.g. "0003").
- If NONE of these sections seem relevant, reply with: NONE

Reply with only the node_id or NONE. No explanation."""

    response = call_llm(prompt, max_tokens=16).strip().strip('"').strip("'")
    logger.debug("navigate_step: chose node_id='%s'", response)

    if response.upper() == "NONE":
        return None
    return response


# ═════════════════════════════════════════════════════════════════════════════
#  Agentic tree search — with fallback full-tree scan
# ═════════════════════════════════════════════════════════════════════════════

def tree_search(
    query: str,
    tree_root: Dict,
    max_depth: int = 8,
) -> Tuple[Optional[Dict], List[str]]:
    """
    Navigate the PageIndex tree to find the most relevant leaf node.

    FIX: Added fallback full-tree keyword scan when tree navigation returns NONE.
    This handles cases where the top-level node summaries don't mention the answer
    but a deep subsection does.

    Args:
        query:      The user's question.
        tree_root:  The full index dict (output of build_index).
        max_depth:  Safety cap on recursion depth.

    Returns:
        (best_node, navigation_path)
        best_node       — the leaf Dict that contains the answer
        navigation_path — list of titles showing the navigation trail
    """
    current_candidates = tree_root.get("children", [])
    navigation_path: List[str] = []
    best_node: Optional[Dict] = None

    for depth in range(max_depth):
        if not current_candidates:
            break

        chosen_id = _navigate_step(query, current_candidates)

        if chosen_id is None:
            logger.info("tree_search: LLM said NONE at depth %d", depth)
            break

        chosen_node = next(
            (n for n in current_candidates if n.get("node_id") == chosen_id),
            None,
        )

        if chosen_node is None:
            logger.warning("tree_search: node_id '%s' not found in candidates", chosen_id)
            break

        navigation_path.append(chosen_node.get("title", chosen_id))
        best_node = chosen_node

        children = chosen_node.get("children", [])
        if not children:
            logger.info("tree_search: reached leaf '%s' after %d steps", chosen_node.get("title"), depth + 1)
            break

        current_candidates = children

    # ── FIX: Fallback — if we got NONE at root level, do a full-tree LLM scan ──
    # This handles the case where top-level summaries don't hint at the answer
    # but a specific subsection (e.g. "3.5 Severity Levels") contains it.
    if best_node is None:
        logger.info("tree_search: primary navigation failed, trying full-tree fallback scan")
        best_node, navigation_path = _full_tree_fallback(query, tree_root)

    return best_node, navigation_path


def _full_tree_fallback(
    query: str,
    tree_root: Dict,
) -> Tuple[Optional[Dict], List[str]]:
    """
    When tree navigation fails, flatten ALL nodes and ask the LLM to pick
    the best one directly. Handles up to 50 nodes at once.

    Returns (best_node, navigation_path).
    """
    all_nodes = _flatten_tree(tree_root.get("children", []))

    if not all_nodes:
        return None, []

    # Batch into groups of 50 to stay within token limits
    BATCH = 50
    best_node = None
    best_path: List[str] = []

    for i in range(0, len(all_nodes), BATCH):
        batch = all_nodes[i:i + BATCH]
        summary_block = _node_summary_block(batch)

        prompt = f"""You are searching a document index for the answer to a query.

Query: "{query}"

Below are ALL document sections (with their summaries). Find the ONE most likely to contain the answer.

{summary_block}

Reply with ONLY the node_id of the best section, or NONE if nothing is relevant.
No explanation."""

        response = call_llm(prompt, max_tokens=16).strip().strip('"').strip("'")

        if response.upper() != "NONE":
            found = next((n for n in batch if n.get("node_id") == response), None)
            if found:
                best_node = found
                best_path = [found.get("title", response)]
                break

    if best_node:
        logger.info("_full_tree_fallback: selected node '%s'", best_node.get("title"))
    else:
        logger.warning("_full_tree_fallback: no relevant node found")

    return best_node, best_path


# ═════════════════════════════════════════════════════════════════════════════
#  Answer generation
# ═════════════════════════════════════════════════════════════════════════════

def generate_answer(
    query: str,
    node: Dict,
    pages: List[Tuple[int, str]],
) -> Dict[str, Any]:
    """
    Given the retrieved node, extract page text and generate a grounded answer.

    FIX: Also includes text from child nodes so that subsection content is
    available even if the parent node was selected.

    Returns a dict with:
        answer        — the LLM-generated answer string
        section       — title of the retrieved section
        structure     — structure ID (e.g. "2.1")
        start_page    — start page number
        end_page      — end page number
        node_id       — node_id
    """
    start = node.get("start_index", 0)
    end   = node.get("end_index", start)
    title = node.get("title", "Unknown Section")

    # Prefer pre-stored node text; fall back to live extraction
    context = node.get("text") or get_page_text(pages, start, end)

    # ── FIX: If node has children, append their text too ──────────────────────
    # This ensures that when a parent section is selected, the answer generation
    # has access to all the subsection content within it.
    child_texts = _collect_child_texts(node)
    if child_texts:
        context = context + "\n\n" + "\n\n".join(child_texts)

    # Truncate to avoid token overflow (keep ~6000 chars)
    context = context[:8000]

    prompt = f"""You are an expert document analyst. Answer the question below
using ONLY the provided document section. Be precise and factual.
If the answer is not found in the section, say: "Not found in this section."

Document section: "{title}" (pages {start}–{end})

Section content:
{context}

Question: {query}

Answer:"""

    answer_text = call_llm(prompt, max_tokens=1024)

    return {
        "answer":    answer_text,
        "section":   title,
        "structure": node.get("structure", ""),
        "start_page": start,
        "end_page":   end,
        "node_id":   node.get("node_id", ""),
    }


def _collect_child_texts(node: Dict, max_chars: int = 3000) -> List[str]:
    """Recursively collect text from child nodes (up to max_chars total)."""
    texts = []
    total = 0
    for child in node.get("children", []):
        text = child.get("text", "")
        if text and total < max_chars:
            texts.append(text[:max_chars - total])
            total += len(text)
        if total >= max_chars:
            break
    return texts


# ═════════════════════════════════════════════════════════════════════════════
#  Top-level retrieve() — single entry point
# ═════════════════════════════════════════════════════════════════════════════

def retrieve(
    query: str,
    tree_root: Dict,
    pages: List[Tuple[int, str]],
) -> Dict[str, Any]:
    """
    Full retrieval pipeline:
      1. Agentic tree navigation → find best node (with fallback)
      2. Extract context from that node's page range
      3. Generate grounded answer with citation

    Args:
        query:      User question.
        tree_root:  Output of build_index().
        pages:      List of (page_number, page_text) from extract_pages().

    Returns dict with: answer, section, structure, start_page, end_page,
                       node_id, navigation_path
    """
    best_node, navigation_path = tree_search(query, tree_root)

    if best_node is None:
        return {
            "answer":         "Could not locate a relevant section for this query.",
            "section":        "N/A",
            "structure":      "N/A",
            "start_page":     0,
            "end_page":       0,
            "node_id":        "N/A",
            "navigation_path": [],
        }

    result = generate_answer(query, best_node, pages)
    result["navigation_path"] = navigation_path
    return result
