# PageIndex RAG — Self-Hosted Implementation

A complete, self-hosted replication of the PageIndex pipeline — no PageIndex API, no vector DB, no chunking.
Runs entirely against your Capgemini Generative Engine LLM endpoint.

---

## Project Structure

```
pageindex_rag/
├── app.py                   # Streamlit UI
├── requirements.txt
├── .env.example             # Copy to .env and fill in your keys
│
├── core/
│   ├── llm_client.py        # Capgemini LLM wrapper (all LLM calls go here)
│   ├── pdf_parser.py        # PDF → pages with <physical_index_X> tags
│   ├── index_builder.py     # Full PageIndex pipeline (all 6 phases)
│   ├── retriever.py         # Agentic tree-search + answer generation
│   └── index_store.py       # Save / load tree JSON
│
├── input_docs/              # Place your PDFs here (or upload via UI)
├── results/                 # Generated index JSON files saved here
└── logs/
```

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env and add your CAPGEMINI_API_KEY

# 3. Run
streamlit run app.py
```

---

## What Each Core Module Does

| Module | PageIndex equivalent | Description |
|--------|----------------------|-------------|
| `pdf_parser.py` | `get_page_tokens()` | Extracts pages, wraps each in `<physical_index_X>` tags |
| `index_builder.py` | `page_index()` + `meta_processor()` | All 6 phases of index generation |
| `retriever.py` | Cookbook notebooks | Agentic tree navigation + answer generation |
| `index_store.py` | Local filesystem | Persist/load the JSON tree |
| `llm_client.py` | `ChatGPT_API()` | Single LLM call wrapper |

---

## Index Generation Phases (inside index_builder.py)

1. **TOC Detection** — scans first 20 pages, asks LLM if a TOC exists
2. **Section Extraction** — 3 strategies with auto-fallback:
   - TOC with page numbers → `toc_transformer` + `toc_index_extractor`
   - TOC without page numbers → `toc_transformer` + `add_page_number_to_toc`
   - No TOC → `generate_toc_from_content` (chunk-by-chunk inference)
3. **Verification** — checks every title appears on its assigned page; retries up to 3×
4. **Recursive Splitting** — nodes > 10 pages AND > 20k tokens are split further
5. **Tree Assembly** — flat section list → nested JSON tree via stack algorithm
6. **Enrichment** — node text, LLM summaries, doc description added

---

## Retrieval Flow (inside retriever.py)

1. LLM reads top-level node summaries → picks most relevant branch
2. Drills into children → picks most relevant sub-section
3. Repeats until leaf node reached
4. Extracts page text from `start_index` → `end_index`
5. LLM generates grounded answer with section + page citation

---

## .env Variables

```
CAPGEMINI_API_KEY=...
CAPGEMINI_BASE_URL=https://openai.generative.engine.capgemini.com/v1
MODEL_ID=amazon.nova-lite-v1:0
```
