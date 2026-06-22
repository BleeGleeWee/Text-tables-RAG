"""
Retriever — hybrid dense + sparse retrieval with Reciprocal Rank Fusion.

Pipeline:
  1. Dense retrieval (Chroma + MiniLM embeddings) → top DENSE_K
  2. Sparse retrieval (BM25 over page_content tokens) → top SPARSE_K
  3. Fuse via RRF → top 20 candidates for the reranker
  4. SQLite docstore lookup → hydrate full chunk records

Metadata filter is opt-in: callers pass {"doc_id": "sofa_2023"} or
{"doc_year": 2023} to restrict retrieval to a subset. We don't auto-detect
year/doc here — that's query_rewriter.py's job in the next build step.

Public API:
    Retriever(metadata_filter=None).retrieve(query, k=20) -> List[RetrievedChunk]
"""

from __future__ import annotations

import pickle
import sqlite3
from dataclasses import dataclass
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

from config import (
    BM25_PATH,
    CHROMA_COLLECTION,
    CHROMA_DIR,
    DENSE_K,
    DOCSTORE_PATH,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL,
    RRF_K,
    SPARSE_K,
)
from ingest import _tokenize  # reuse the exact tokenizer ingest used


# ──────────────────────────────────────────────────────────────────────────
# Data class
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class RetrievedChunk:
    """
    A chunk returned from retrieval, hydrated from the SQLite docstore.

    Mirrors the Chunk dataclass but adds retrieval-time scores. The
    `rrf_score` is the fused score; `dense_rank` and `sparse_rank` are the
    1-indexed ranks the chunk had in each retriever (or None if it wasn't
    in that retriever's top-K).
    """
    chunk_id: str
    doc_id: str
    doc_title: str
    doc_year: int
    page: int
    section: str
    chunk_type: str
    page_content: str
    table_caption: str
    table_json: str

    rrf_score: float
    dense_rank: Optional[int]
    sparse_rank: Optional[int]


# ──────────────────────────────────────────────────────────────────────────
# Retriever
# ──────────────────────────────────────────────────────────────────────────
class Retriever:
    """
    Hybrid retriever. Builds dense (Chroma) and sparse (BM25) indices on
    construction so a single instance can serve many queries cheaply.

    The BM25 index is rebuilt from the persisted token corpus at init time
    (not loaded from a pickled BM25Okapi). This guarantees consistency with
    whatever was ingested and avoids stale-index bugs if rank_bm25's internal
    state ever changes between versions.
    """

    def __init__(self, metadata_filter: Optional[dict] = None):
        self.metadata_filter = metadata_filter

        # ── Dense side ──
        self._embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )
        self._chroma = Chroma(
            collection_name=CHROMA_COLLECTION,
            embedding_function=self._embeddings,
            persist_directory=str(CHROMA_DIR),
        )

        # ── Sparse side ──
        if not BM25_PATH.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {BM25_PATH}. Run ingest.py first."
            )
        with open(BM25_PATH, "rb") as f:
            state = pickle.load(f)
        self._bm25_chunk_ids: List[str] = state["chunk_ids"]
        self._bm25_doc_ids: List[str] = state["doc_ids"]
        self._bm25 = BM25Okapi(state["corpus_tokens"])

        # ── Docstore connection ──
        # SQLite connections are per-thread; we lazily create one when
        # needed instead of holding one open across the instance lifetime.

    # ──────────────────────────────────────────────────────────────────
    # Dense
    # ──────────────────────────────────────────────────────────────────
    def _dense_search(self, query: str, k: int) -> List[str]:
        """Return Chroma's top-k chunk_ids for the query."""
        # similarity_search returns Document objects with our metadata.
        # We requested embeddings normalized at ingest, so similarity is
        # cosine and higher = better. Chroma already returns them sorted.
        results = self._chroma.similarity_search(
            query=query,
            k=k,
            filter=self.metadata_filter,  # Chroma accepts None as "no filter"
        )
        return [r.metadata["chunk_id"] for r in results]

    # ──────────────────────────────────────────────────────────────────
    # Sparse
    # ──────────────────────────────────────────────────────────────────
    def _sparse_search(self, query: str, k: int) -> List[str]:
        """
        Return BM25's top-k chunk_ids for the query.

        BM25 doesn't support metadata filtering natively, so we apply the
        filter post-hoc: score all chunks, then drop those that don't match
        the filter, then take the top-k. For a 211-chunk corpus this is
        free; for 10k+ corpora we'd want a smarter approach.
        """
        query_tokens = _tokenize(query)
        scores = self._bm25.get_scores(query_tokens)

        # Pair (score, chunk_id, doc_id), filter, sort by score desc.
        candidates = list(zip(scores, self._bm25_chunk_ids, self._bm25_doc_ids))

        if self.metadata_filter:
            candidates = [
                c for c in candidates
                if self._matches_filter(c[1], c[2])
            ]

        candidates.sort(key=lambda t: t[0], reverse=True)
        # Drop chunks with score 0 — these are non-matches BM25 returned
        # purely to fill the array, no actual term overlap.
        candidates = [c for c in candidates if c[0] > 0]
        return [cid for _, cid, _ in candidates[:k]]

    def _matches_filter(self, chunk_id: str, doc_id: str) -> bool:
        """
        Check if a BM25 candidate matches the metadata filter.

        The BM25 state only has chunk_id and doc_id directly. For doc_year
        filters we'd need to look up the chunk in the docstore, but in
        practice metadata_filter is almost always {"doc_id": ...} — the
        rewriter will resolve year -> doc_id before calling us. Support
        doc_id natively; defer richer filters to the rewriter.
        """
        f = self.metadata_filter or {}
        if "doc_id" in f and doc_id != f["doc_id"]:
            return False
        # doc_year not supported here — rewriter should convert to doc_id.
        return True

    # ──────────────────────────────────────────────────────────────────
    # Fusion
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _rrf_fuse(
        dense_ids: List[str],
        sparse_ids: List[str],
        k_rrf: int = RRF_K,
    ) -> List[tuple[str, float, Optional[int], Optional[int]]]:
        """
        Reciprocal Rank Fusion.

        score(doc) = sum over retrievers of 1 / (k_rrf + rank_i)

        Returns a list of (chunk_id, rrf_score, dense_rank, sparse_rank)
        sorted by score desc. k_rrf=60 is the standard value from the
        original RRF paper — smooths out the contribution of low-ranked
        hits so a doc that ranks #1 in one retriever doesn't completely
        dominate one that ranks #5 in both.
        """
        dense_rank = {cid: i + 1 for i, cid in enumerate(dense_ids)}
        sparse_rank = {cid: i + 1 for i, cid in enumerate(sparse_ids)}

        all_ids = set(dense_rank) | set(sparse_rank)
        scored = []
        for cid in all_ids:
            score = 0.0
            d_rank = dense_rank.get(cid)
            s_rank = sparse_rank.get(cid)
            if d_rank is not None:
                score += 1.0 / (k_rrf + d_rank)
            if s_rank is not None:
                score += 1.0 / (k_rrf + s_rank)
            scored.append((cid, score, d_rank, s_rank))

        scored.sort(key=lambda t: t[1], reverse=True)
        return scored

    # ──────────────────────────────────────────────────────────────────
    # Docstore hydration
    # ──────────────────────────────────────────────────────────────────
    def _hydrate(
        self,
        scored: List[tuple[str, float, Optional[int], Optional[int]]],
    ) -> List[RetrievedChunk]:
        """
        Look up each chunk_id in SQLite and build RetrievedChunk objects.
        Preserves the order from `scored` (which is RRF-ranked).
        """
        if not scored:
            return []

        chunk_ids = [cid for cid, _, _, _ in scored]
        placeholders = ",".join("?" * len(chunk_ids))

        conn = sqlite3.connect(str(DOCSTORE_PATH))
        try:
            rows = conn.execute(f"""
                SELECT chunk_id, doc_id, doc_title, doc_year, page, section,
                       chunk_type, page_content, table_caption, table_json
                FROM chunks
                WHERE chunk_id IN ({placeholders})
            """, chunk_ids).fetchall()
        finally:
            conn.close()

        by_id = {r[0]: r for r in rows}

        out: List[RetrievedChunk] = []
        for cid, score, d_rank, s_rank in scored:
            row = by_id.get(cid)
            if row is None:
                # Shouldn't happen in steady state — would mean Chroma/BM25
                # contain a chunk_id that's not in the docstore. Skip
                # silently to keep retrieval robust to partial-ingest bugs.
                continue
            out.append(RetrievedChunk(
                chunk_id=row[0],
                doc_id=row[1],
                doc_title=row[2],
                doc_year=row[3],
                page=row[4],
                section=row[5],
                chunk_type=row[6],
                page_content=row[7],
                table_caption=row[8],
                table_json=row[9],
                rrf_score=score,
                dense_rank=d_rank,
                sparse_rank=s_rank,
            ))
        return out

    # ──────────────────────────────────────────────────────────────────
    # Public entry point
    # ──────────────────────────────────────────────────────────────────
    def retrieve(self, query: str, k: int = 20) -> List[RetrievedChunk]:
        """
        Run hybrid retrieval and return the top-k fused candidates,
        hydrated from the docstore.

        Typical usage:
            chunks = Retriever().retrieve("hidden costs of agrifood systems")
            # → top 20 candidates, passed to reranker next.
        """
        dense_ids = self._dense_search(query, DENSE_K)
        sparse_ids = self._sparse_search(query, SPARSE_K)
        scored = self._rrf_fuse(dense_ids, sparse_ids)
        scored = scored[:k]
        return self._hydrate(scored)


# ──────────────────────────────────────────────────────────────────────────
# Test entry point
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    # Test queries covering different retrieval patterns:
    #   - exact-term match (BM25 should excel)
    #   - semantic paraphrase (dense should excel)
    #   - table-grounded (table chunks should surface)
    #   - cross-doc (no filter)
    #   - filtered to one doc
    test_queries = [
        ("Exact-term (BM25 should shine)",
         "PPFI primary production flexibility index", None),
        ("Semantic paraphrase (dense should shine)",
         "how can countries make their food systems handle disruptions better",
         None),
        ("Table-grounded query (should surface TABLE 2)",
         "how many people cannot afford a healthy diet by income group", None),
        ("Filtered to SOFA 2023",
         "what are hidden costs of agrifood systems",
         {"doc_id": "sofa_2023"}),
        ("Filtered to SOFA 2025",
         "land degradation policy interventions",
         {"doc_id": "sofa_2025"}),
    ]

    # Optional CLI override: python src/retriever.py "your query"
    if len(sys.argv) > 1:
        custom_query = " ".join(sys.argv[1:])
        test_queries = [("Custom query", custom_query, None)]

    print("\n" + "=" * 70)
    print("Retriever smoke test")
    print("=" * 70)

    # Build retrievers (one per filter — small cost, easier to read)
    for label, query, mfilter in test_queries:
        print(f"\n{'─' * 70}")
        print(f"{label}")
        print(f"Query:  {query}")
        if mfilter:
            print(f"Filter: {mfilter}")
        print(f"{'─' * 70}")

        retriever = Retriever(metadata_filter=mfilter)
        results = retriever.retrieve(query, k=5)

        if not results:
            print("  (no results)")
            continue

        for i, r in enumerate(results, start=1):
            d = f"d#{r.dense_rank}" if r.dense_rank else "d#-"
            s = f"s#{r.sparse_rank}" if r.sparse_rank else "s#-"
            print(f"  {i}. [{r.chunk_id}]  rrf={r.rrf_score:.4f}  {d}/{s}")
            print(f"     {r.doc_id} p{r.page} | {r.chunk_type} | "
                  f"{r.section[:60]}")
            preview = r.page_content[:140].replace("\n", " ")
            print(f"     \"{preview}...\"")