# PageIndex-Learning
Our pageindex_rag code — 8 guardrails built in
Here's every guardrail, exactly where it sits:
1. JSON parse failure → auto-retry with LLM self-correction
python# index_builder.py: _retry_json_parse()
except json.JSONDecodeError:
    return _retry_json_parse(prompt, raw)  # asks LLM to fix its own broken JSON
2. Section verification — every title checked against its assigned page
python# index_builder.py: verify_sections()
probe = " ".join(title.split()[:6]).lower()
if probe and probe in page_text.lower():
    correct.append(sec)
else:
    incorrect.append(sec)   # flagged for fixing
3. Wrong page assignments → LLM fix with 3 retries
python# index_builder.py: fix_incorrect_sections()
for attempt in range(1, max_retries + 1):   # retries up to 3×
    ...
# Fallback: return incorrect sections as-is rather than crash
4. Oversized nodes → recursive splitting
python# index_builder.py: split_large_nodes()
if page_span > max_pages and token_count > max_tokens:
    # re-run generate_toc_from_content on just those pages
5. Prompt token overflow → hard truncation
python# index_builder.py line 363
{tagged[:12000]}   # never overflows the LLM context window
6. Invalid node_id during navigation → graceful stop
python# retriever.py: tree_search()
if chosen_node is None:
    logger.warning("node_id '%s' not found", chosen_id)
    break   # stops navigation, returns best_node found so far
7. No relevant node found → clean fallback message
python# retriever.py: retrieve()
if best_node is None:
    return {"answer": "Could not locate a relevant section for this query.", ...}
8. Missing env variables → fail fast with clear message
python# llm_client.py
if not api_key or not base_url:
    raise EnvironmentError("CAPGEMINI_API_KEY and CAPGEMINI_BASE_URL must be set...")
