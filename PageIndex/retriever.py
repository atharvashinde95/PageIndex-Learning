import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from core.llm_client import call_llm
from core.pdf_parser import get_page_text

logger = logging.getLogger(__name__)

STOPWORDS = {
    "the","a","an","of","and","or","to","in","on","for","by","is",
    "are","be","with","as","that","this","it","at","from","was",
    "were","but","not","if","then","than","so","such","into",
    "their","its","his","her","they","them","we","our","you","your"
}

CLAUSE_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){1,6}\b", re.I)

# simple pattern to detect definition-like questions
DEFQ_RE = re.compile(
    r"\b(?:what\s+is|who\s+is|define|explain|what\s+does\s+[A-Za-z0-9\-&/ ]+\s+stand\s+for|stands?\s+for)\b",
    re.I
)

# ------------------------------
# Token Helpers
# ------------------------------
def _norm(s: Any) -> str:
    return str(s or "").strip()

def _tokenize(s: str) -> List[str]:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\.]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return [t for t in s.split() if t not in STOPWORDS]

# ------------------------------
# Flatten Tree
# ------------------------------
def _flatten(nodes: List[Dict]) -> List[Dict]:
    out = []
    for n in nodes:
        out.append(n)
        out.extend(_flatten(n.get("children", [])))
    return out

# ------------------------------
# Build navigation path
# ------------------------------
def _path_to_node(children: List[Dict], target: Dict) -> List[str]:
    path = []
    def dfs(nodes, trail):
        for n in nodes:
            new = trail + [n.get("title", n.get("node_id"))]
            if n is target:
                path.extend(new)
                return True
            if dfs(n.get("children", []), new):
                return True
        return False
    dfs(children, [])
    return path

# ------------------------------
# Document Preparation (scoring)
# ------------------------------
def _prepare_docs(nodes: List[Dict]) -> List[Dict[str, Any]]:
    docs = []
    df = {}

    for n in nodes:
        t_title = set(_tokenize(_norm(n.get("title"))))
        t_sum   = set(_tokenize(_norm(n.get("summary"))))
        t_txt   = set(_tokenize(_norm((n.get("text") or "")[:2000])))

        doc = {
            "node": n,
            "title_tokens": t_title,
            "summary_tokens": t_sum,
            "text_tokens": t_txt,
        }
        docs.append(doc)

        merged = t_title | t_sum | t_txt
        for tok in merged:
            df[tok] = df.get(tok, 0) + 1

    for d in docs:
        d["_df"] = df
        d["_N"]  = len(docs)

    return docs

def _idf(tok: str, df: Dict[str, int], N: int) -> float:
    return 1.0 + (N / max(1, df.get(tok, 1))) ** 0.35

def _score_node(doc: Dict[str, Any], query_tokens: List[str], clause_nums: List[str]):
    node = doc["node"]
    struct = _norm(node.get("structure"))
    score = 0.0

    # Clause match boost
    for c in clause_nums:
        if struct == c:
            score += 300
        elif struct.startswith(c + "."):
            score += 150

    df = doc["_df"]
    N  = doc["_N"]

    for tok in query_tokens:
        idf = _idf(tok, df, N)
        if tok in doc["title_tokens"]:
            score += 4.0 * idf
        if tok in doc["summary_tokens"]:
            score += 3.0 * idf
        if tok in doc["text_tokens"]:
            score += 1.6 * idf

    depth = len(struct.split("."))
    score += max(0, 8 - depth) * 0.1

    # BUG FIX: Penalise nodes whose text is suspiciously short (likely TOC-sourced)
    text_len = len((node.get("text") or "").strip())
    if text_len < 300:
        score *= 0.1

    return score

# BUG FIX: Return TOP-N nodes, not just the single best, so multi-section
# questions (like "tell me about the WANA region") can draw from all
# relevant sections.
def _pick_top_nodes(query: str, tree_root: Dict, top_n: int = 3) -> List[Tuple[Dict, List[str]]]:
    nodes = _flatten(tree_root.get("children", []))
    if not nodes:
        return []
    docs = _prepare_docs(nodes)

    q_tokens    = _tokenize(query)
    clause_nums = CLAUSE_RE.findall(query)

    scored = [(doc["node"], _score_node(doc, q_tokens, clause_nums)) for doc in docs]
    scored.sort(key=lambda x: x[1], reverse=True)

    # BUG FIX: Lower the minimum threshold – the old 2.2 was too aggressive.
    # A threshold of 0.5 still filters out total noise but keeps valid nodes.
    MIN_SCORE = 0.5
    results = []
    for node, score in scored[:top_n]:
        if score < MIN_SCORE:
            break
        path = _path_to_node(tree_root.get("children", []), node)
        results.append((node, path))

    return results

# Keep old single-best helper for callers that need it
def _pick_best_node(query: str, tree_root: Dict) -> Tuple[Optional[Dict], List[str]]:
    top = _pick_top_nodes(query, tree_root, top_n=1)
    if not top:
        return None, []
    return top[0]

# ------------------------------
# Strict extractive fallback
# ------------------------------
def _extractive_fallback(query: str, context: str, limit=700):
    if not context:
        return "Not found in this section."
    q = set(_tokenize(query))
    if not q:
        return context[:limit]

    sentences = re.split(r"(?<=[.!?])\s+", context)
    scored = []
    for s in sentences:
        overlap = len(q & set(_tokenize(s)))
        if overlap:
            scored.append((overlap, s))
    if not scored:
        return "Not found in this section."
    scored.sort(key=lambda x: x[0], reverse=True)
    return " ".join([s for _, s in scored[:3]])[:limit]

# ------------------------------
# Utilities for concept inference
# ------------------------------
TERM_CAPTURE = re.compile(
    r"\b(?:what\s+is|who\s+is|define|explain|what\s+does\s+|stands?\s+for\s+)([A-Za-z0-9\-&/ \(\)]+)\??",
    re.I
)

def _extract_concept_term(question: str) -> Optional[str]:
    if not DEFQ_RE.search(question or ""):
        return None
    m = TERM_CAPTURE.search(question or "")
    if m:
        term = m.group(1).strip()
        term = re.sub(r"^(the|a|an)\s+", "", term, flags=re.I).strip(" ?.")
        return term[:80] if term else None
    tokens = [t for t in _tokenize(question) if len(t) > 2]
    return " ".join(tokens[-4:]) if tokens else None

def _find_acronym_expansion(acronym: str, pages) -> Optional[str]:
    if not acronym:
        return None
    acr = re.escape(acronym.strip("() "))
    pat1 = re.compile(rf"([A-Z][A-Za-z0-9&/\-, ]{{3,120}})\s*\(\s*{acr}\s*\)", re.M)
    pat2 = re.compile(rf"\b{acr}\s*\(\s*([^)]+)\)", re.M)

    for _, text in pages:
        if not text:
            continue
        m1 = pat1.search(text)
        if m1:
            full = m1.group(1).strip()
            if 3 <= len(full.split()) <= 20:
                return full
        m2 = pat2.search(text)
        if m2:
            full = m2.group(1).strip()
            if 3 <= len(full.split()) <= 20:
                return full
    return None

def _collect_evidence_for_term(term: str, tree_root: Dict, pages, char_limit=4000) -> str:
    term_ci = term.lower()
    buf, used = [], 0

    def hit(node: Dict) -> bool:
        t = " ".join([
            _norm(node.get("title")),
            _norm(node.get("summary")),
            _norm(node.get("text") or ""),
        ]).lower()
        return term_ci in t

    def dfs(nodes):
        nonlocal used
        for n in nodes:
            if used >= char_limit:
                return
            if hit(n):
                s, e = n.get("start_index"), n.get("end_index")
                raw = get_page_text(pages, s, e)
                chunk = raw[: max(0, char_limit - used)]
                if chunk:
                    buf.append(chunk)
                    used += len(chunk)
            dfs(n.get("children", []))

    dfs(tree_root.get("children", []))
    return ("\n\n---\n\n").join(buf)

def _infer_from_evidence(question: str, evidence: str) -> str:
    prompt = f"""
You are a careful document assistant.

You MAY infer concise concept explanations using ONLY the EVIDENCE below.
- Do NOT use outside knowledge.
- If the evidence is insufficient to answer, reply exactly:
  Not found in this document.

EVIDENCE:
{evidence}

Question: {question}

Answer (one or two sentences, concise):
"""
    try:
        ans = call_llm(prompt, max_tokens=180)
        if not ans or "[ERROR" in ans:
            raise RuntimeError
        return ans.strip()
    except Exception:
        return "Not found in this document."

# ------------------------------
# BUG FIX: Multi-node answer generator
# Combines context from top-N relevant nodes so broad questions
# (e.g. "tell me about WANA region") get a complete answer.
# ------------------------------
def generate_answer_multi(query: str, nodes_with_paths: List[Tuple[Dict, List[str]]], pages) -> Dict[str, Any]:
    """
    Build an answer by passing all relevant sections to the LLM at once.
    Falls back to the single-node path if only one node is available.
    """
    if not nodes_with_paths:
        return {
            "answer": "No relevant clause or section found.",
            "section": "—",
            "structure": "—",
            "start_page": "—",
            "end_page": "—",
            "node_id": "—",
            "navigation_path": [],
        }

    # Gather combined context from all top nodes
    combined_parts = []
    for node, _path in nodes_with_paths:
        s, e = node.get("start_index"), node.get("end_index")
        title = node.get("title", "")
        raw   = get_page_text(pages, s, e)
        summary_text = node.get("summary", "")
        section_text = (summary_text + "\n\n" + raw).strip()
        combined_parts.append(f"[Section {node.get('structure')} – {title}]\n{section_text}")

    combined = "\n\n========\n\n".join(combined_parts)

    # Use the best (top-scored) node for metadata
    best_node, best_path = nodes_with_paths[0]

    prompt = f"""
You are a document assistant. Answer ONLY using the content below.
Provide a thorough, well-structured answer. If the answer cannot be found in the content,
reply exactly: "Not found in this document."

Content:
{combined[:8000]}

Question: {query}
Answer:
"""
    try:
        ans = call_llm(prompt, max_tokens=1000)
        if not ans or "[ERROR" in ans:
            raise RuntimeError
        ans = ans.strip()
    except Exception:
        # Extractive fallback across all combined text
        ans = _extractive_fallback(query, combined)

    if ans and "Not found in this document." not in ans and "Not found in this section." not in ans:
        return {
            "answer": ans,
            "section": best_node.get("title", "—"),
            "structure": best_node.get("structure", "—"),
            "start_page": best_node.get("start_index", "—"),
            "end_page": best_node.get("end_index", "—"),
            "node_id": best_node.get("node_id", "—"),
            "navigation_path": best_path,
        }

    # Last resort: concept inference
    term = _extract_concept_term(query)
    if term:
        expansion = None
        if re.fullmatch(r"[A-Z0-9\-]{2,10}", term):
            expansion = _find_acronym_expansion(term, pages)

        if expansion:
            inferred = f"{term} — {expansion}."
        else:
            evidence = _collect_evidence_for_term(term, {"children": [n for n,_ in nodes_with_paths]}, pages, char_limit=3000)
            if len(evidence) < 400:
                all_text = "\n\n---\n\n".join([t for _, t in pages])[:4000]
                evidence = (evidence + "\n\n---\n\n" + all_text).strip()
            inferred = _infer_from_evidence(query, evidence)

        if inferred and "Not found in this document." not in inferred:
            return {
                "answer": inferred,
                "section": best_node.get("title", "—"),
                "structure": best_node.get("structure", "—"),
                "start_page": best_node.get("start_index", "—"),
                "end_page": best_node.get("end_index", "—"),
                "node_id": best_node.get("node_id", "—"),
                "navigation_path": best_path,
            }

    fallback_msg = "Not found in this document." if DEFQ_RE.search(query or "") else "Not found in this section."
    return {
        "answer": fallback_msg,
        "section": best_node.get("title", "—"),
        "structure": best_node.get("structure", "—"),
        "start_page": best_node.get("start_index", "—"),
        "end_page": best_node.get("end_index", "—"),
        "node_id": best_node.get("node_id", "—"),
        "navigation_path": best_path,
    }


# Keep legacy single-node generator for any callers that use it directly
def generate_answer(query: str, node: Dict, pages) -> Dict[str, Any]:
    if node is None:
        return {
            "answer": "No relevant clause or section found.",
            "section": "—",
            "structure": "—",
            "start_page": "—",
            "end_page": "—",
            "node_id": "—",
            "navigation_path": [],
        }
    return generate_answer_multi(query, [(node, [])], pages)


# ------------------------------
# Main entry point
# ------------------------------
def retrieve(query: str, tree_root: Dict, pages):
    """
    BUG FIX: Now retrieves top-3 relevant nodes and synthesises an answer
    from all of them, instead of only passing the single best node to the LLM.
    This fixes broad/multi-section questions like "tell me about WANA region".
    """
    top_nodes = _pick_top_nodes(query, tree_root, top_n=3)

    if not top_nodes:
        return {
            "answer": "No relevant clause or section found.",
            "section": "—",
            "structure": "—",
            "start_page": "—",
            "end_page": "—",
            "node_id": "—",
            "navigation_path": [],
        }

    result = generate_answer_multi(query, top_nodes, pages)
    return result
