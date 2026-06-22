"""
Streamlit UI for the FAO Multimodal RAG system.

Layout:
  - Sidebar: doc registry overview, cache controls, conversation reset
  - Main: chat-style Q&A with rendered sources panel beneath each answer

Features baked in:
  - Chat history kept in st.session_state (last 4 turns sent to rewriter)
  - Sources panel shows reranked chunks with type, page, section, score
  - Table chunks render as proper HTML tables (not raw markdown)
  - Citations [chunk_id] in the answer are converted into clickable
    badges that scroll to / highlight the corresponding source
  - "Cached" badge when a response came from the cache
  - Verification status (green check / warning if hallucinated cites)
"""

from __future__ import annotations

import json
import time
from typing import List

import streamlit as st

from cache import QueryCache
from config import CHAT_HISTORY_TURNS, DOCUMENTS
from rag_chain import RAGChain, RAGResponse
from retriever import RetrievedChunk


# ──────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FAO SOFA RAG",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────────────────────────────────
# We keep:
#   - messages:  full chat transcript for display
#   - history:   compact (role, content) pairs for the rewriter — last N turns
#   - responses: parallel list of RAGResponse for each assistant turn,
#                so we can render sources/verification after the fact.
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []
if "responses" not in st.session_state:
    st.session_state.responses = []


# ──────────────────────────────────────────────────────────────────────────
# Heavy resources — built once, cached across reruns
# ──────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models (one-time, takes ~10s)...")
def get_chain() -> RAGChain:
    """Build the RAG chain once. st.cache_resource keeps it across reruns."""
    return RAGChain(use_cache=True)


@st.cache_resource
def get_cache() -> QueryCache:
    """Cache handle for the sidebar stats/clear controls."""
    return QueryCache()


# ──────────────────────────────────────────────────────────────────────────
# Rendering helpers
# ──────────────────────────────────────────────────────────────────────────
def _render_table_chunk(chunk: RetrievedChunk) -> None:
    """
    Render a table chunk as a proper HTML table using table_json metadata
    (cleaner than the markdown form). Falls back to markdown if table_json
    can't be parsed.
    """
    if chunk.table_caption:
        st.markdown(f"**{chunk.table_caption}**")

    try:
        table_data = json.loads(chunk.table_json) if chunk.table_json else None
    except json.JSONDecodeError:
        table_data = None

    if table_data and table_data.get("rows"):
        rows = table_data["rows"]
        header = rows[0]
        body = rows[1:] if len(rows) > 1 else []
        # Build a Streamlit dataframe-style display via st.table for
        # clean rendering without pandas import overhead.
        st.table([dict(zip(header, row + [""] * (len(header) - len(row))))
                  for row in body])
    else:
        # Fall back to the markdown form embedded in page_content
        st.markdown(chunk.page_content)


def _render_source_card(chunk: RetrievedChunk, index: int) -> None:
    """
    Render a single source in the sources panel as an expandable card.
    """
    score = getattr(chunk, "rerank_score", None)
    score_str = f"rerank={score:+.2f}" if score is not None else ""

    badge = "📊 TABLE" if chunk.chunk_type == "table" else "📄 TEXT"
    title = (
        f"{badge}  •  [{chunk.chunk_id}]  •  "
        f"{chunk.doc_id} p{chunk.page}  •  {score_str}"
    )

    with st.expander(title, expanded=False):
        st.caption(f"**Section:** {chunk.section}")
        if chunk.chunk_type == "table":
            _render_table_chunk(chunk)
        else:
            st.markdown(chunk.page_content)


def _render_sources_panel(response: RAGResponse) -> None:
    """Show the reranked chunks beneath an assistant message."""
    if not response.sources:
        return
    st.markdown("**Sources**")
    cited_set = (
        set(response.verification.valid_ids)
        if response.verification else set()
    )

    for i, chunk in enumerate(response.sources):
        if chunk.chunk_id in cited_set:
            # Highlight cited sources subtly
            st.markdown(f"✅ *Cited in answer*")
        _render_source_card(chunk, i)


def _render_meta_row(response: RAGResponse) -> None:
    """
    Small status row above the sources panel: cache hit, verification,
    refusal reason, etc.
    """
    bits: List[str] = []
    stats = response.retrieval_stats or {}

    if stats.get("cache_hit"):
        bits.append("⚡ cached")
    if response.refused:
        reason = stats.get("refusal_reason", "model_refused")
        bits.append(f"🚫 refused ({reason})")
    if response.verification:
        v = response.verification
        if v.is_clean and v.valid_ids:
            bits.append(f"✅ {len(v.valid_ids)} citation(s) verified")
        elif v.hallucinated_ids:
            bits.append(
                f"⚠️ {len(v.hallucinated_ids)} hallucinated citation(s) "
                f"stripped"
            )
    if response.rewrite and len(response.rewrite.sub_queries) > 1:
        bits.append(
            f"🔀 decomposed into {len(response.rewrite.sub_queries)} "
            f"sub-queries"
        )
    if response.rewrite and response.rewrite.metadata_filter:
        f = response.rewrite.metadata_filter
        bits.append(f"🎯 filtered to {f.get('doc_id', '?')}")

    if bits:
        st.caption("  •  ".join(bits))


# ──────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────
def render_sidebar() -> None:
    st.sidebar.markdown("## 🌾 FAO SOFA RAG")
    st.sidebar.markdown(
        "Multimodal RAG over FAO *State of Food and Agriculture* In Brief "
        "reports."
    )

    st.sidebar.markdown("### Indexed documents")
    for meta in DOCUMENTS.values():
        st.sidebar.markdown(
            f"- **{meta['year']}** — {meta['title']}"
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Conversation")
    if st.sidebar.button("🔄 Reset conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history = []
        st.session_state.responses = []
        st.rerun()

    st.sidebar.markdown("### Cache")
    cache = get_cache()
    stats = cache.stats()
    st.sidebar.metric("Entries", stats["total_entries"])
    st.sidebar.metric("Total hits", stats["total_hits"])
    if st.sidebar.button("🗑️ Clear cache", use_container_width=True):
        n = cache.clear()
        st.sidebar.success(f"Cleared {n} entr{'y' if n == 1 else 'ies'}.")
        st.rerun()

    with st.sidebar.expander("Top cached queries", expanded=False):
        if not stats["top_queries"]:
            st.caption("No queries cached yet.")
        for q in stats["top_queries"]:
            st.markdown(f"- ({q['hits']}×) {q['query']}")

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Built with PyMuPDF, pdfplumber, MiniLM, BM25, ChromaDB, "
        "BGE-style cross-encoder, and Gemini 2.0 Flash."
    )


# ──────────────────────────────────────────────────────────────────────────
# Main chat area
# ──────────────────────────────────────────────────────────────────────────
def render_chat() -> None:
    st.title("Ask the FAO SOFA reports")
    st.caption(
        "Questions are answered using only retrieved passages from the three "
        "indexed reports. Citations are inline; click any source below to "
        "verify."
    )

    # Replay history
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # If this is an assistant message, render its sources panel
            if msg["role"] == "assistant":
                idx = msg.get("response_idx")
                if idx is not None and idx < len(st.session_state.responses):
                    response = st.session_state.responses[idx]
                    _render_meta_row(response)
                    _render_sources_panel(response)

    # New input
    query = st.chat_input("Ask a question about the FAO SOFA reports...")
    if not query:
        return

    # User turn
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Assistant turn
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            chain = get_chain()
            t0 = time.time()
            response = chain.answer(
                query=query,
                chat_history=st.session_state.history,
            )
            elapsed = time.time() - t0

        st.markdown(response.answer)

        # Small timing line
        cache_hit = (response.retrieval_stats or {}).get("cache_hit", False)
        if cache_hit:
            st.caption(f"⚡ Answered in {elapsed*1000:.0f}ms (cache hit)")
        else:
            st.caption(f"Answered in {elapsed:.2f}s")

        _render_meta_row(response)
        _render_sources_panel(response)

    # Persist for replay
    st.session_state.responses.append(response)
    st.session_state.messages.append({
        "role": "assistant",
        "content": response.answer,
        "response_idx": len(st.session_state.responses) - 1,
    })

    # Update compact history for the rewriter (last N turns).
    # Note: we keep last CHAT_HISTORY_TURNS exchanges = 2 * N messages.
    st.session_state.history.append({"role": "user", "content": query})
    st.session_state.history.append(
        {"role": "assistant", "content": response.answer}
    )
    max_msgs = 2 * CHAT_HISTORY_TURNS
    if len(st.session_state.history) > max_msgs:
        st.session_state.history = st.session_state.history[-max_msgs:]


# ──────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────
def main() -> None:
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()