"""
app.py
------
Streamlit frontend for the PageIndex-equivalent RAG system.
Keeps UI simple and clean. All heavy logic lives in core/.
"""

import logging
import sys
from pathlib import Path

import streamlit as st

# ── ensure project root is on sys.path ──────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from core.index_builder import build_index
from core.index_store import index_exists, load_index, save_index
from core.pdf_parser import extract_pages
from core.retriever import retrieve

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)

# ─────────────────────────────────────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PageIndex RAG",
    page_icon="📑",
    layout="wide",
)

st.title("📑 PageIndex RAG")
st.caption("Vectorless · Reasoning-based · No chunking · Powered by Capgemini Generative Engine")

# ─────────────────────────────────────────────────────────────────────────────
#  Sidebar — Upload + Index
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("1 · Document")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file:
        # Save to input_docs/
        input_dir = Path("input_docs")
        input_dir.mkdir(exist_ok=True)
        pdf_path = input_dir / uploaded_file.name
        pdf_path.write_bytes(uploaded_file.read())

        st.success(f"Saved: {uploaded_file.name}")

        already_indexed = index_exists(uploaded_file.name)
        if already_indexed:
            st.info("Index already exists. Skipping re-index.")

        st.divider()
        st.header("2 · Index Options")

        add_summaries   = st.checkbox("Add node summaries",      value=True)
        add_text        = st.checkbox("Store page text in nodes", value=True)
        add_description = st.checkbox("Generate doc description", value=True)

        st.divider()

        if st.button("🔨 Build Index", use_container_width=True, disabled=already_indexed):
            with st.spinner("Building index — this may take a few minutes…"):
                try:
                    pages = extract_pages(str(pdf_path))
                    st.session_state["pages"] = pages

                    tree = build_index(
                        pages,
                        add_summaries=add_summaries,
                        add_text=add_text,
                        add_description=add_description,
                    )
                    save_index(tree, uploaded_file.name)
                    st.session_state["tree"] = tree
                    st.session_state["pdf_name"] = uploaded_file.name
                    st.success("Index built and saved ✅")
                except Exception as exc:
                    st.error(f"Indexing failed: {exc}")
                    logging.exception("Indexing error")

        if already_indexed and "tree" not in st.session_state:
            if st.button("📂 Load Existing Index", use_container_width=True):
                tree = load_index(uploaded_file.name)
                pages = extract_pages(str(pdf_path))
                st.session_state["tree"]     = tree
                st.session_state["pages"]    = pages
                st.session_state["pdf_name"] = uploaded_file.name
                st.success("Index loaded ✅")

# ─────────────────────────────────────────────────────────────────────────────
#  Main area — Tree viewer + Query
# ─────────────────────────────────────────────────────────────────────────────

tree  = st.session_state.get("tree")
pages = st.session_state.get("pages")

if not tree:
    st.info("Upload a PDF and build (or load) its index using the sidebar to get started.")
    st.stop()

# ── Doc description ──────────────────────────────────────────────────────────
doc_desc = tree.get("doc_description", "")
if doc_desc:
    st.info(f"**Document:** {doc_desc}")

col_tree, col_query = st.columns([1, 1], gap="large")

# ── Left: Tree viewer ─────────────────────────────────────────────────────────
with col_tree:
    st.subheader("📂 Index Tree")

    def render_tree(nodes, indent=0):
        for node in nodes:
            prefix = "─" * indent
            title  = node.get("title", "Untitled")
            struct = node.get("structure", "")
            start  = node.get("start_index", "?")
            end    = node.get("end_index", "?")
            nid    = node.get("node_id", "")
            label  = f"{prefix} **{struct}** {title} `p.{start}–{end}` `#{nid}`"

            with st.expander(label, expanded=(indent == 0)):
                summary = node.get("summary", "")
                if summary:
                    st.caption(summary)
                children = node.get("children", [])
                if children:
                    render_tree(children, indent + 2)

    render_tree(tree.get("children", []))

# ── Right: Query + Answer ─────────────────────────────────────────────────────
with col_query:
    st.subheader("🔍 Ask a Question")

    query = st.text_area(
        "Enter your question:",
        placeholder="e.g. What was the total revenue in 2023?",
        height=100,
    )

    if st.button("▶ Get Answer", use_container_width=True) and query.strip():
        with st.spinner("Navigating document tree…"):
            try:
                result = retrieve(query=query.strip(), tree_root=tree, pages=pages)

                st.success("Answer found")

                st.markdown("### Answer")
                st.write(result["answer"])

                st.divider()
                st.markdown("### Citation")
                c1, c2, c3 = st.columns(3)
                c1.metric("Section", result["structure"] or "—")
                c2.metric("Pages",   f"{result['start_page']} – {result['end_page']}")
                c3.metric("Node ID", result["node_id"])
                st.caption(f"**Section title:** {result['section']}")

                st.divider()
                st.markdown("### Navigation Path")
                path = result.get("navigation_path", [])
                if path:
                    st.write(" → ".join(path))
                else:
                    st.write("—")

            except Exception as exc:
                st.error(f"Retrieval failed: {exc}")
                logging.exception("Retrieval error")
