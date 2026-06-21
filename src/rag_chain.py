"""
RAG chain — end-to-end orchestration.

Flow:
  user query + chat_history
    → rewriter            (history-aware rewrite + decomposition + filter)
    → retriever × N       (one call per sub-query, with shared filter)
    → merge + dedupe      (keep best RRF score per chunk_id)
    → reranker            (cross-encoder on merged candidates → top FINAL_K)
    → refusal check       (reranker floor + grounded prompt)
    → Gemini generation   (grounded answer with inline [chunk_id] citations)
    → RAGResponse

Stateless: chat history is passed in per call. The caller (Streamlit UI,
eval harness) owns conversation state.

Public API:
    RAGChain().answer(query, chat_history=None) -> RAGResponse
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from config import (
    FINAL_K,
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_MODEL,
    GEMINI_TEMPERATURE,
    GOOGLE_API_KEY,
    REFUSAL_RERANKER_FLOOR,
)
from query_rewriter import QueryRewriter, RewriteResult
from reranker import Reranker
from retriever import RetrievedChunk, Retriever


# ──────────────────────────────────────────────────────────────────────────
# Response dataclass
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class RAGResponse:
    """
    Everything the caller needs from one query.

    answer            Generated answer text with inline [chunk_id] citation
                      markers. For refusals, a canned message explaining
                      why no answer could be produced.
    sources           The chunks passed to the LLM. The UI shows these in
                      the sources panel. Even on refusal, this may be
                      populated (chunks were retrieved but didn't meet the
                      floor) — useful for showing the user what was found.
    rewrite           Rewriter output (for debugging and transparency).
    retrieval_stats   Per-sub-query counts, top reranker score, refusal
                      reason. Useful for the eval harness.
    refused           True if we returned a canned refusal instead of
                      generating. The citation verifier downstream can
                      skip verification when this is True.
    """
    answer: str
    sources: List[RetrievedChunk] = field(default_factory=list)
    rewrite: Optional[RewriteResult] = None
    retrieval_stats: dict = field(default_factory=dict)
    refused: bool = False


# ──────────────────────────────────────────────────────────────────────────
# Grounded generation prompt
# ──────────────────────────────────────────────────────────────────────────
_GENERATION_SYSTEM_PROMPT = """You are a careful research assistant answering questions about FAO State of Food and Agriculture reports.

You will be given the user's question and a numbered list of CHUNKS from the reports. Each chunk has an ID like [sofa_2021_p018_table_0] and contains text or a table.

Strict rules:

1. Use ONLY information from the provided chunks. Do not use outside knowledge, common sense estimates, or anything that isn't in the chunks.

2. Cite every factual claim with the chunk ID that supports it, inline, in square brackets. For example: "Around 3 billion people cannot afford a healthy diet [sofa_2021_p018_table_0]." Multiple chunks can support one claim — list them all: "...as discussed in the foreword [sofa_2021_p005_text_0][sofa_2021_p006_text_1]."

3. If the chunks do not contain enough information to answer the question, say exactly: "I don't have information about this in the documents." Do not guess, do not partially answer with caveats — just refuse cleanly.

4. Never fabricate numbers, page references, or chunk IDs. If a number isn't in the chunks, don't include it. If you cite a chunk ID, it must appear in the provided list.

5. Keep answers concise and grounded. Lead with the direct answer to the user's question. Add nuance from the chunks only if it's relevant to the question.

6. When a table chunk is provided and the question is about specific numbers, quote the relevant cells exactly and cite the table chunk.

Examples of correct behavior:

Question: "How many people cannot afford a healthy diet?"
Chunks include [sofa_2021_p018_table_0] with TABLE 2 showing 3,000.5 million worldwide.
Good answer: "Globally, around 3 billion people cannot afford a healthy diet — specifically 3,000.5 million people in 2019, or 41.9% of the world's population [sofa_2021_p018_table_0]."

Question: "What is the carbon footprint of rice production?"
Chunks do not contain carbon footprint figures.
Good answer: "I don't have information about this in the documents."

Question: "Compare hidden costs in low-income vs high-income countries."
Chunks contain hidden-cost breakdowns by income group.
Good answer: "In low-income countries, hidden costs of agrifood systems represent a higher share of GDP than in high-income countries... [sofa_2023_p017_text_1]. Specifically, the ratio is 12% in lower-middle-income countries [sofa_2023_p017_text_1] while high-income countries show a lower proportion [sofa_2023_p015_text_0]."
"""


_REFUSAL_MESSAGE = "I don't have information about this in the documents."


# ──────────────────────────────────────────────────────────────────────────
# Chain
# ──────────────────────────────────────────────────────────────────────────
class RAGChain:
    """
    Builds rewriter, retriever-factory, reranker, and LLM once. The
    retriever is constructed per-call because metadata_filter varies by
    query (a single Retriever instance is filter-locked at construction).

    The retriever's BM25 rebuild + Chroma open are cheap (~50ms total) so
    per-call construction is fine. The expensive things (embedding model,
    reranker model, Gemini client) are reused across calls via this class's
    own instance.
    """

    def __init__(self):
        if not GOOGLE_API_KEY:
            raise RuntimeError(
                "GOOGLE_API_KEY not set in .env — RAG chain cannot start."
            )
        self._rewriter = QueryRewriter()
        self._reranker = Reranker()
        self._llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            temperature=GEMINI_TEMPERATURE,
            max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
            google_api_key=GOOGLE_API_KEY,
        )

    # ──────────────────────────────────────────────────────────────────
    # Public entry point
    # ──────────────────────────────────────────────────────────────────
    def answer(
        self,
        query: str,
        chat_history: Optional[List[dict]] = None,
    ) -> RAGResponse:
        """
        Run the full pipeline for one user query.
        """
        query = (query or "").strip()
        if not query:
            return RAGResponse(
                answer="Please ask a question.",
                refused=True,
                retrieval_stats={"refusal_reason": "empty_query"},
            )

        # ── 1. Rewrite ──
        rewrite = self._rewriter.rewrite(query, chat_history)

        # ── 2. Retrieve per sub-query, merge ──
        merged_candidates = self._retrieve_and_merge(
            sub_queries=rewrite.sub_queries,
            metadata_filter=rewrite.metadata_filter,
        )

        if not merged_candidates:
            return RAGResponse(
                answer=_REFUSAL_MESSAGE,
                sources=[],
                rewrite=rewrite,
                refused=True,
                retrieval_stats={
                    "refusal_reason": "no_candidates",
                    "n_sub_queries": len(rewrite.sub_queries),
                    "n_candidates": 0,
                },
            )

        # ── 3. Rerank ──
        # Use rewritten_query (not sub_queries) as the reranker's query —
        # the reranker scores against the user's overall information need,
        # while sub_queries' job was to broaden recall during retrieval.
        reranked = self._reranker.rerank(
            query=rewrite.rewritten_query,
            candidates=merged_candidates,
            top_k=FINAL_K,
        )

        top_score = reranked[0].rerank_score if reranked else float("-inf")

        # ── 4. Refusal check (hard floor) ──
        if top_score < REFUSAL_RERANKER_FLOOR:
            return RAGResponse(
                answer=_REFUSAL_MESSAGE,
                sources=reranked,  # show user what we found, even if weak
                rewrite=rewrite,
                refused=True,
                retrieval_stats={
                    "refusal_reason": "low_reranker_score",
                    "top_reranker_score": top_score,
                    "floor": REFUSAL_RERANKER_FLOOR,
                    "n_sub_queries": len(rewrite.sub_queries),
                    "n_candidates": len(merged_candidates),
                },
            )

        # ── 5. Generate (grounded with refusal allowed in-prompt) ──
        answer_text = self._generate(
            rewritten_query=rewrite.rewritten_query,
            chunks=reranked,
            chat_history=chat_history,
        )

        # If Gemini self-refused (returned the exact refusal message), mark
        # the response as refused so the citation verifier downstream
        # doesn't waste work extracting citations from a refusal.
        refused = answer_text.strip() == _REFUSAL_MESSAGE

        return RAGResponse(
            answer=answer_text,
            sources=reranked,
            rewrite=rewrite,
            refused=refused,
            retrieval_stats={
                "top_reranker_score": top_score,
                "n_sub_queries": len(rewrite.sub_queries),
                "n_candidates": len(merged_candidates),
                "n_final_chunks": len(reranked),
            },
        )

    # ──────────────────────────────────────────────────────────────────
    # Retrieval + merge
    # ──────────────────────────────────────────────────────────────────
    def _retrieve_and_merge(
        self,
        sub_queries: List[str],
        metadata_filter: Optional[dict],
    ) -> List[RetrievedChunk]:
        """
        Run retrieval per sub-query and merge by chunk_id, keeping the
        best (highest) rrf_score per chunk. The reranker will re-score
        everything so we don't need to be clever about ranks here — just
        get the union of candidates to the reranker.
        """
        retriever = Retriever(metadata_filter=metadata_filter)
        merged: Dict[str, RetrievedChunk] = {}

        for sq in sub_queries:
            candidates = retriever.retrieve(sq, k=20)
            for c in candidates:
                existing = merged.get(c.chunk_id)
                if existing is None or c.rrf_score > existing.rrf_score:
                    merged[c.chunk_id] = c

        # Return as a list, ordered by RRF score desc. Order matters only
        # for the smoke-test log — the reranker will re-sort anyway.
        return sorted(merged.values(), key=lambda x: x.rrf_score, reverse=True)

    # ──────────────────────────────────────────────────────────────────
    # Generation
    # ──────────────────────────────────────────────────────────────────
    def _generate(
        self,
        rewritten_query: str,
        chunks: List[RetrievedChunk],
        chat_history: Optional[List[dict]],
    ) -> str:
        """Build the grounded-generation prompt and call Gemini."""
        # Format chunks as a numbered list. We use chunk_id as the citation
        # token because that's stable, unique, and matches what the prompt
        # examples show. chunk_type and section_path included as context
        # so the LLM can pick table chunks when the question demands them.
        chunk_block = self._format_chunks(chunks)

        history_block = self._format_history(chat_history)

        user_block = (
            f"{history_block}"
            f"Question: {rewritten_query}\n\n"
            f"CHUNKS:\n{chunk_block}\n\n"
            f"Answer the question using only the chunks above. Cite each "
            f"claim inline with the chunk ID in square brackets. If the "
            f"chunks don't contain enough information, refuse using the "
            f"exact message specified in the rules."
        )

        try:
            response = self._llm.invoke([
                SystemMessage(content=_GENERATION_SYSTEM_PROMPT),
                HumanMessage(content=user_block),
            ])
            return response.content.strip()
        except Exception as e:
            # Network/quota failures here are user-facing — we can't degrade
            # to a fast path because there's no fast path for generation.
            # Return a clear error message; the eval harness and UI can
            # surface this distinctly from a refusal.
            return (
                f"I encountered an error while generating the answer "
                f"({type(e).__name__}). Please try again."
            )

    @staticmethod
    def _format_chunks(chunks: List[RetrievedChunk]) -> str:
        """
        Format chunks for the prompt. We include chunk_id, type, doc/page
        breadcrumb, and the page_content. Section is included because it
        helps the LLM understand cross-chunk context (e.g., "this is from
        the FOREWORD vs. the SUMMARY").
        """
        parts = []
        for i, c in enumerate(chunks, start=1):
            header = (
                f"[{c.chunk_id}] (type={c.chunk_type}, "
                f"{c.doc_id} p{c.page}, section: {c.section})"
            )
            parts.append(f"{header}\n{c.page_content}\n")
        return "\n".join(parts)

    @staticmethod
    def _format_history(chat_history: Optional[List[dict]]) -> str:
        """
        Format chat history into a compact prompt block.

        We give Gemini enough context to keep tone and continuity, but the
        rewriter has already resolved coreferences into rewritten_query, so
        the history here is mostly for tone-matching and avoiding
        contradiction with prior answers.
        """
        if not chat_history:
            return ""
        lines = ["Recent conversation (most recent last):"]
        for turn in chat_history:
            role = turn.get("role", "user").capitalize()
            content = turn.get("content", "").strip()
            if len(content) > 400:
                content = content[:400] + "..."
            lines.append(f"  {role}: {content}")
        lines.append("")
        return "\n".join(lines) + "\n"


# ──────────────────────────────────────────────────────────────────────────
# Test entry point
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    test_cases = [
        {
            "label": "Factual table-grounded question",
            "query": "How many people cannot afford a healthy diet "
                     "according to the 2021 report?",
            "history": None,
        },
        {
            "label": "Cross-document comparative (decomposition)",
            "query": "Compare hidden costs in low-income vs high-income "
                     "countries",
            "history": None,
        },
        {
            "label": "Out-of-scope (should refuse)",
            "query": "What is the carbon footprint of rice production in "
                     "Vietnam?",
            "history": None,
        },
        {
            "label": "Coreference with history",
            "query": "what about for kilocalories?",
            "history": [
                {"role": "user",
                 "content": "What is the DSFI for protein?"},
                {"role": "assistant",
                 "content": "The DSFI for protein measures the diversity "
                            "of dietary protein sourcing pathways "
                            "[sofa_2021_p013_text_0]."},
            ],
        },
        {
            "label": "Latest-report reference",
            "query": "What does the latest report say about land "
                     "degradation policies?",
            "history": None,
        },
    ]

    if len(sys.argv) > 1:
        custom_query = " ".join(sys.argv[1:])
        test_cases = [{"label": "Custom", "query": custom_query,
                       "history": None}]

    print("\n" + "=" * 70)
    print("RAG chain smoke test")
    print("=" * 70)

    chain = RAGChain()

    for case in test_cases:
        print(f"\n{'─' * 70}")
        print(f"{case['label']}")
        print(f"Query:   {case['query']}")
        if case["history"]:
            print(f"History: {len(case['history'])} turn(s)")
        print(f"{'─' * 70}")

        response = chain.answer(case["query"], case["history"])

        print(f"\nRewritten:  {response.rewrite.rewritten_query if response.rewrite else '(none)'}")
        if response.rewrite and len(response.rewrite.sub_queries) > 1:
            print(f"Sub-queries:")
            for sq in response.rewrite.sub_queries:
                print(f"  - {sq}")
        if response.rewrite and response.rewrite.metadata_filter:
            print(f"Filter:     {response.rewrite.metadata_filter}")
        print(f"Refused:    {response.refused}")
        print(f"Stats:      {response.retrieval_stats}")
        print(f"\nAnswer:\n{response.answer}\n")
        print(f"Sources ({len(response.sources)}):")
        for s in response.sources:
            score = getattr(s, "rerank_score", None)
            score_str = f"{score:+.3f}" if score is not None else "n/a"
            print(f"  [{s.chunk_id}]  rerank={score_str}  "
                  f"{s.doc_id} p{s.page} | {s.chunk_type}")