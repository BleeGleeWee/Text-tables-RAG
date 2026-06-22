"""

Query/response cache — SQLite-backed, keyed on rewritten query.



Why this exists:

  During development and demos, the same queries get re-run dozens of

  times. Each one costs a Gemini call (free tier: 15 RPM, 1500/day). The

  cache skips retrieval + rerank + generation when we've already answered

  the same canonical question before.



Cache key:

  SHA-256 hash of the rewritten query. We cache AFTER rewriting so that

  "what about for kilocalories?" (with DSFI history) and "What is the DSFI

  for kilocalories?" (without history) both resolve to the same cached

  entry — they're the same question, normalized.



What we store:

  The full RAGResponse serialized as JSON, including sources, rewrite

  result, verification, and stats. Cache hits look identical to fresh

  responses from the caller's perspective.



Lifecycle:

  No TTL. Entries live forever until explicitly cleared via clear() or

  clear_for_doc(doc_id). The underlying FAO reports don't change between

  ingests, so eviction would just cost rebuild time.



Public API:

    QueryCache().get(rewritten_query) -> Optional[RAGResponse]

    QueryCache().set(rewritten_query, response)

    QueryCache().clear()

    QueryCache().clear_for_doc(doc_id)

    QueryCache().stats() -> dict

"""



from __future__ import annotations



import dataclasses

import hashlib

import json

import sqlite3

import time

from typing import Optional, TYPE_CHECKING



from config import CACHE_PATH

from citation_verifier import VerificationResult

from query_rewriter import RewriteResult

from retriever import RetrievedChunk



if TYPE_CHECKING:

    from rag_chain import RAGResponse





# ──────────────────────────────────────────────────────────────────────────

# Hashing

# ──────────────────────────────────────────────────────────────────────────

def _hash_query(rewritten_query: str) -> str:

    """

    SHA-256 of the lowercased, stripped rewritten query.



    We lowercase + strip so trivial casing/whitespace differences don't

    create separate cache entries. The rewriter already produces fairly

    canonical output, but this normalization is a cheap safety net.

    """

    normalized = (rewritten_query or "").strip().lower()

    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()





# ──────────────────────────────────────────────────────────────────────────

# Serialization

# ──────────────────────────────────────────────────────────────────────────

def _serialize_response(response: "RAGResponse") -> str:

    """

    Convert a RAGResponse to a JSON string for storage.



    RAGResponse contains nested dataclasses (RetrievedChunk, RewriteResult,

    VerificationResult). asdict() walks them recursively into plain dicts.

    The rerank_score that's dynamically attached to RetrievedChunk in

    reranker.py isn't a dataclass field, so we capture it manually.

    """

    payload = {

        "answer": response.answer,

        "sources": [

            {**dataclasses.asdict(s),

             "rerank_score": getattr(s, "rerank_score", None)}

            for s in response.sources

        ],

        "rewrite": dataclasses.asdict(response.rewrite) if response.rewrite else None,

        "retrieval_stats": response.retrieval_stats,

        "refused": response.refused,

        "verification": (

            dataclasses.asdict(response.verification)

            if response.verification else None

        ),

    }

    return json.dumps(payload, ensure_ascii=False)





def _deserialize_response(blob: str) -> "RAGResponse":

    """

    Inverse of _serialize_response. Reconstructs the nested dataclasses.



    Local import of RAGResponse avoids circular imports — cache is imported

    by rag_chain, so cache can't import rag_chain at module load time.

    """

    from rag_chain import RAGResponse



    data = json.loads(blob)



    # Rebuild sources

    sources = []

    for s in data.get("sources", []):

        rerank_score = s.pop("rerank_score", None)

        chunk = RetrievedChunk(**s)

        if rerank_score is not None:

            chunk.rerank_score = rerank_score

        sources.append(chunk)



    rewrite = RewriteResult(**data["rewrite"]) if data.get("rewrite") else None

    verification = (

        VerificationResult(**data["verification"])

        if data.get("verification") else None

    )



    return RAGResponse(

        answer=data["answer"],

        sources=sources,

        rewrite=rewrite,

        retrieval_stats=data.get("retrieval_stats", {}),

        refused=data.get("refused", False),

        verification=verification,

    )





# ──────────────────────────────────────────────────────────────────────────

# Cache

# ──────────────────────────────────────────────────────────────────────────

class QueryCache:

    """

    SQLite-backed cache. One row per unique rewritten query.



    Schema:

        cache_key       SHA-256 hex (primary key)

        rewritten_query Original text (debugging aid)

        response_blob   Full RAGResponse as JSON

        doc_ids         Comma-separated list of doc_ids that appear in

                        sources — enables clear_for_doc().

        created_at      Unix timestamp (debugging aid)

        hits            Counter incremented on each get() that returns

                        this row.

    """



    def __init__(self):

        self._ensure_schema()



    # ──────────────────────────────────────────────────────────────────

    # Schema

    # ──────────────────────────────────────────────────────────────────

    def _ensure_schema(self) -> None:

        conn = sqlite3.connect(str(CACHE_PATH))

        try:

            conn.execute("""

                CREATE TABLE IF NOT EXISTS cache (

                    cache_key       TEXT PRIMARY KEY,

                    rewritten_query TEXT NOT NULL,

                    response_blob   TEXT NOT NULL,

                    doc_ids         TEXT NOT NULL DEFAULT '',

                    created_at      INTEGER NOT NULL,

                    hits            INTEGER NOT NULL DEFAULT 0

                )

            """)

            conn.commit()

        finally:

            conn.close()



    # ──────────────────────────────────────────────────────────────────

    # Get / set

    # ──────────────────────────────────────────────────────────────────

    def get(self, rewritten_query: str) -> Optional["RAGResponse"]:

        """

        Look up a cached response. Returns None on miss. On hit, increments

        the row's hit counter (cheap, useful for cache stats).

        """

        if not rewritten_query:

            return None

        key = _hash_query(rewritten_query)

        conn = sqlite3.connect(str(CACHE_PATH))

        try:

            row = conn.execute(

                "SELECT response_blob FROM cache WHERE cache_key = ?",

                (key,),

            ).fetchone()

            if row is None:

                return None

            # Bump hits before returning so concurrent gets each count.

            conn.execute(

                "UPDATE cache SET hits = hits + 1 WHERE cache_key = ?",

                (key,),

            )

            conn.commit()

            return _deserialize_response(row[0])

        finally:

            conn.close()



    def set(self, rewritten_query: str, response: "RAGResponse") -> None:

        """

        Store a response under the rewritten_query key. Existing entries

        are replaced (REPLACE INTO) so re-querying with a new answer

        overwrites the old one.



        We DON'T cache refusals or error responses — refusals might become

        valid answers after a re-ingest, and error responses (network

        failures) shouldn't poison the cache.

        """

        if not rewritten_query or not response:

            return

        if response.refused:

            return

        if response.answer.startswith("I encountered an error"):

            return



        key = _hash_query(rewritten_query)

        blob = _serialize_response(response)

        # Extract doc_ids from sources for clear_for_doc() support.

        doc_ids = ",".join(sorted({s.doc_id for s in response.sources}))



        conn = sqlite3.connect(str(CACHE_PATH))

        try:

            conn.execute("""

                INSERT OR REPLACE INTO cache (

                    cache_key, rewritten_query, response_blob,

                    doc_ids, created_at, hits

                ) VALUES (?, ?, ?, ?, ?, 0)

            """, (key, rewritten_query, blob, doc_ids, int(time.time())))

            conn.commit()

        finally:

            conn.close()



    # ──────────────────────────────────────────────────────────────────

    # Maintenance

    # ──────────────────────────────────────────────────────────────────

    def clear(self) -> int:

        """Drop all cache entries. Returns count deleted."""

        conn = sqlite3.connect(str(CACHE_PATH))

        try:

            cur = conn.execute("DELETE FROM cache")

            conn.commit()

            return cur.rowcount

        finally:

            conn.close()



    def clear_for_doc(self, doc_id: str) -> int:

        """

        Drop cache entries whose sources include this doc_id.

        Use after re-ingesting a single document so stale answers don't

        leak through.

        """

        conn = sqlite3.connect(str(CACHE_PATH))

        try:

            # LIKE with comma-bracketed doc_ids ensures partial-name

            # matches (e.g., 'sofa_202' matching 'sofa_2021' and

            # 'sofa_2023') don't happen.

            cur = conn.execute(

                "DELETE FROM cache "

                "WHERE doc_ids = ? "

                "   OR doc_ids LIKE ? "

                "   OR doc_ids LIKE ? "

                "   OR doc_ids LIKE ?",

                (doc_id,

                 f"{doc_id},%",

                 f"%,{doc_id},%",

                 f"%,{doc_id}"),

            )

            conn.commit()

            return cur.rowcount

        finally:

            conn.close()



    def stats(self) -> dict:

        """Return summary stats about the cache."""

        conn = sqlite3.connect(str(CACHE_PATH))

        try:

            total = conn.execute(

                "SELECT COUNT(*) FROM cache"

            ).fetchone()[0]

            total_hits = conn.execute(

                "SELECT COALESCE(SUM(hits), 0) FROM cache"

            ).fetchone()[0]

            top_hits = conn.execute(

                "SELECT rewritten_query, hits FROM cache "

                "ORDER BY hits DESC LIMIT 5"

            ).fetchall()

            return {

                "total_entries": total,

                "total_hits": total_hits,

                "top_queries": [

                    {"query": q, "hits": h} for q, h in top_hits

                ],

            }

        finally:

            conn.close()





# ──────────────────────────────────────────────────────────────────────────

# Test entry point

# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    import sys

    from rag_chain import RAGChain



    # Two queries — second is a paraphrase of the first that the rewriter

    # should normalize to (roughly) the same canonical form. Watch the

    # hit/miss pattern.

    test_queries = [

        "What is the PPFI?",

        "What is the PPFI?",   # identical — must hit cache

        "How many people cannot afford a healthy diet?",

        "How many people cannot afford a healthy diet?",  # identical — hit

    ]



    if len(sys.argv) > 1:

        test_queries = sys.argv[1:]



    print("\n" + "=" * 70)

    print("Query cache smoke test")

    print("=" * 70)



    chain = RAGChain()

    cache = QueryCache()



    # Clear so test is reproducible

    n_cleared = cache.clear()

    print(f"\nCleared {n_cleared} prior entries.")



    for i, query in enumerate(test_queries, start=1):

        print(f"\n{'─' * 70}")

        print(f"Query {i}: {query}")

        print(f"{'─' * 70}")



        # First check cache directly using the rewritten query.

        # Run the rewriter only — don't pay for full chain yet.

        rewrite = chain._rewriter.rewrite(query, chat_history=None)

        print(f"Rewritten: {rewrite.rewritten_query}")



        t0 = time.time()

        cached = cache.get(rewrite.rewritten_query)

        if cached is not None:

            elapsed = time.time() - t0

            print(f"  CACHE HIT  ({elapsed*1000:.1f}ms)")

            print(f"  Answer (first 120 chars): "

                  f"{cached.answer[:120]}...")

        else:

            elapsed = time.time() - t0

            print(f"  CACHE MISS ({elapsed*1000:.1f}ms) — running full chain")

            t1 = time.time()

            response = chain.answer(query, chat_history=None)

            full_elapsed = time.time() - t1

            print(f"  Full chain took {full_elapsed:.2f}s")

            cache.set(rewrite.rewritten_query, response)

            print(f"  Stored in cache.")

            print(f"  Answer (first 120 chars): "

                  f"{response.answer[:120]}...")



    print(f"\n{'─' * 70}")

    print("Cache stats")

    print(f"{'─' * 70}")

    stats = cache.stats()

    print(f"Total entries: {stats['total_entries']}")

    print(f"Total hits:    {stats['total_hits']}")

    print(f"Top queries:")

    for q in stats["top_queries"]:

        print(f"  {q['hits']} hits — {q['query']}")