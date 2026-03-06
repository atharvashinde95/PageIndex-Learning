"""
core/retriever.py
-----------------
Agentic tree-search retrieval — mirrors the PageIndex retrieval notebook.

The LLM navigates the tree by reasoning over node summaries at each level,
drilling down from root → branch → leaf until it reaches the most relevant section.
No vector similarity. No embeddings. Pure reasoning.
"""

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


# ═════════════════════════════════════════════════════════════════════════════
#  Core navigation step
# ═════════════════════════════════════════════════════════════════════════════

def _navigate_step(query: str, candidates: List[Dict]) -> Optional[str]:
    """
    Given a query and a list of candidate nodes at the same level,
    ask the LLM which node_id is most likely to contain the answer.

    Returns the chosen node_id, or None if the answer is already found
    at this level (LLM says ANSWER_HERE).
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
#  Agentic tree search
# ═════════════════════════════════════════════════════════════════════════════

def tree_search(
    query: str,
    tree_root: Dict,
    max_depth: int = 8,
) -> Tuple[Optional[Dict], List[str]]:
    """
    Navigate the PageIndex tree to find the most relevant leaf node.

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
            # Leaf node reached
            logger.info("tree_search: reached leaf '%s' after %d steps", chosen_node.get("title"), depth + 1)
            break

        current_candidates = children

    return best_node, navigation_path


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

    prompt = f"""You are an expert document analyst. Answer the question below
using ONLY the provided document section. Be precise and factual.
If the answer is not found in the section, say: "Not found in this section."

Document section: "{title}" (pages {start}–{end})

Section content:
{context[:6000]}

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
      1. Agentic tree navigation → find best node
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
