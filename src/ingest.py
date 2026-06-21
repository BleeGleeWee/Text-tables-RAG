"""

Ingest — write enriched chunks to all three storage backends.



Backends:

  1. ChromaDB (dense vector store)

       embed: embedding_text via all-MiniLM-L6-v2

       store: page_content as the document, to_metadata() as metadata

  2. BM25 index (sparse retrieval)

       tokenize: page_content (no contextual prefix — keep IDF clean)

       persist: pickled BM25Okapi + corpus tokens + chunk_id list

  3. SQLite docstore

       store: full chunk record (chunk_id PK + all fields) for lookup

              after retrieval returns chunk_ids



Run end-to-end for one PDF or all PDFs. Idempotent: re-running clears

prior storage for the affected doc_id(s) and rewrites them, so chunk_id

collisions can't accumulate stale records.



Public API:

    ingest_document(filename) -> dict      # ingest one PDF

    ingest_all() -> dict                   # ingest every doc in DOCUMENTS

"""



from __future__ import annotations



import json

import pickle

import re

import sqlite3

from pathlib import Path

from typing import List



from langchain_huggingface import HuggingFaceEmbeddings

from langchain_chroma import Chroma

from rank_bm25 import BM25Okapi

from tqdm import tqdm



from chunker import Chunk, chunk_document

from config import (

    BM25_PATH,

    CHROMA_COLLECTION,

    CHROMA_DIR,

    DATA_DIR,

    DOCSTORE_PATH,

    DOCUMENTS,

    EMBEDDING_DEVICE,

    EMBEDDING_MODEL,

)

from enricher import enrich_chunks

from parser import parse_pdf





# ──────────────────────────────────────────────────────────────────────────

# Tokenization for BM25

# ──────────────────────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"\b\w+\b")





def _tokenize(text: str) -> List[str]:

    """

    Simple regex tokenizer for BM25.

    Lowercases, splits on word boundaries. Preserves acronyms (PPFI, DSFI,

    SMAE) as single tokens because they're alphanumeric and survive \\w+.

    No stemming — FAO terminology is precise and stemming hurts more than

    it helps on acronyms and proper nouns.

    """

    return _TOKEN_RE.findall(text.lower())





# ──────────────────────────────────────────────────────────────────────────

# SQLite docstore

# ──────────────────────────────────────────────────────────────────────────

def _ensure_docstore_schema(conn: sqlite3.Connection) -> None:

    """Create the chunks table if it doesn't exist."""

    conn.execute("""

        CREATE TABLE IF NOT EXISTS chunks (

            chunk_id       TEXT PRIMARY KEY,

            doc_id         TEXT NOT NULL,

            doc_title      TEXT NOT NULL,

            doc_year       INTEGER NOT NULL,

            page           INTEGER NOT NULL,

            section        TEXT NOT NULL,

            chunk_type     TEXT NOT NULL,

            page_content   TEXT NOT NULL,

            embedding_text TEXT NOT NULL,

            table_caption  TEXT NOT NULL DEFAULT '',

            table_json     TEXT NOT NULL DEFAULT ''

        )

    """)

    # Index on doc_id so we can quickly delete + count by doc.

    conn.execute(

        "CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id)"

    )

    conn.commit()





def _docstore_clear_doc(conn: sqlite3.Connection, doc_id: str) -> int:

    """Remove all rows for a doc_id. Returns count deleted."""

    cur = conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))

    conn.commit()

    return cur.rowcount





def _docstore_insert(conn: sqlite3.Connection, chunks: List[Chunk]) -> None:

    """Bulk insert chunks."""

    rows = [(

        c.chunk_id, c.doc_id, c.doc_title, c.doc_year, c.page,

        c.section, c.chunk_type, c.page_content, c.embedding_text,

        c.table_caption, c.table_json,

    ) for c in chunks]

    conn.executemany("""

        INSERT INTO chunks (

            chunk_id, doc_id, doc_title, doc_year, page,

            section, chunk_type, page_content, embedding_text,

            table_caption, table_json

        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

    """, rows)

    conn.commit()





# ──────────────────────────────────────────────────────────────────────────

# BM25 persistence

# ──────────────────────────────────────────────────────────────────────────

def _load_bm25_state() -> dict:

    """

    Load the full BM25 corpus state from disk, or return an empty state.



    We persist three parallel lists:

        chunk_ids      — order matches BM25 doc indices

        corpus_tokens  — token lists used to build the index

        doc_ids        — used for filtering on rebuild

    The actual BM25Okapi object is rebuilt from corpus_tokens at load time

    (or after any update) because BM25Okapi pickles fine but rebuilding it

    is fast and guarantees consistency.

    """

    if BM25_PATH.exists():

        with open(BM25_PATH, "rb") as f:

            return pickle.load(f)

    return {"chunk_ids": [], "corpus_tokens": [], "doc_ids": []}





def _save_bm25_state(state: dict) -> None:

    """Write BM25 state to disk."""

    BM25_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(BM25_PATH, "wb") as f:

        pickle.dump(state, f)





def _bm25_drop_doc(state: dict, doc_id: str) -> int:

    """

    Remove all entries for doc_id from the BM25 state in place.

    Returns count dropped. Caller is responsible for saving.

    """

    keep_ids: List[str] = []

    keep_tokens: List[List[str]] = []

    keep_docs: List[str] = []

    dropped = 0

    for cid, toks, did in zip(

        state["chunk_ids"], state["corpus_tokens"], state["doc_ids"]

    ):

        if did == doc_id:

            dropped += 1

        else:

            keep_ids.append(cid)

            keep_tokens.append(toks)

            keep_docs.append(did)

    state["chunk_ids"] = keep_ids

    state["corpus_tokens"] = keep_tokens

    state["doc_ids"] = keep_docs

    return dropped





def _bm25_append(state: dict, chunks: List[Chunk]) -> None:

    """Append new chunks' tokens to the BM25 state in place."""

    for c in chunks:

        state["chunk_ids"].append(c.chunk_id)

        state["corpus_tokens"].append(_tokenize(c.page_content))

        state["doc_ids"].append(c.doc_id)





# ──────────────────────────────────────────────────────────────────────────

# Chroma

# ──────────────────────────────────────────────────────────────────────────

def _get_embeddings() -> HuggingFaceEmbeddings:

    """Build the embedding function. Cached model downloads on first run."""

    return HuggingFaceEmbeddings(

        model_name=EMBEDDING_MODEL,

        model_kwargs={"device": EMBEDDING_DEVICE},

        encode_kwargs={"normalize_embeddings": True},

    )





def _get_chroma(embeddings: HuggingFaceEmbeddings) -> Chroma:

    """Open the persistent Chroma collection."""

    return Chroma(

        collection_name=CHROMA_COLLECTION,

        embedding_function=embeddings,

        persist_directory=str(CHROMA_DIR),

    )





def _chroma_drop_doc(chroma: Chroma, doc_id: str) -> int:

    """

    Delete all chunks for a doc_id from Chroma. Returns count deleted.

    Uses metadata filter — chunks were stored with doc_id in metadata.

    """

    # Chroma's delete() takes a where filter directly.

    existing = chroma.get(where={"doc_id": doc_id})

    n = len(existing.get("ids", []))

    if n:

        chroma.delete(where={"doc_id": doc_id})

    return n





def _chroma_add(chroma: Chroma, chunks: List[Chunk]) -> None:

    """

    Add chunks to Chroma. We embed embedding_text (with context prefix)

    but store page_content as the user-facing document field.

    """

    ids = [c.chunk_id for c in chunks]

    metadatas = [c.to_metadata() for c in chunks]

    # Chroma embeds `texts`. We pass embedding_text so the prefix is

    # included in the embedding. The document field stored in Chroma

    # comes from `texts` too — but we want the LLM to see page_content,

    # not embedding_text. So we use the lower-level add_texts with both

    # custom texts (for embedding) and a separate metadata mapping

    # that holds page_content for downstream lookup.

    #

    # Cleanest approach: use add_texts(texts=embedding_text, ...) and

    # rely on the SQLite docstore for the canonical page_content. The

    # text stored in Chroma's `documents` column is only used if we

    # query without consulting the docstore — which we never do.

    chroma.add_texts(

        texts=[c.embedding_text for c in chunks],

        metadatas=metadatas,

        ids=ids,

    )





# ──────────────────────────────────────────────────────────────────────────

# Main entry points

# ──────────────────────────────────────────────────────────────────────────

def ingest_document(filename: str, embeddings=None, chroma=None) -> dict:

    """

    Ingest one PDF end-to-end: parse → chunk → enrich → write to all stores.

    Idempotent: clears prior storage for this doc_id before writing.



    embeddings/chroma are optional — pass them in to avoid re-loading the

    model when ingesting multiple docs in a loop.

    """

    if filename not in DOCUMENTS:

        raise ValueError(

            f"{filename} is not in config.DOCUMENTS. Known: "

            f"{list(DOCUMENTS.keys())}"

        )



    doc_meta = DOCUMENTS[filename]

    doc_id = doc_meta["doc_id"]

    pdf_path = DATA_DIR / filename



    print(f"\n{'=' * 70}")

    print(f"Ingesting: {filename} ({doc_id})")

    print(f"{'=' * 70}")



    # ── 1. Parse + chunk + enrich ──

    print("\n[1/4] Parsing PDF...")

    pages = parse_pdf(pdf_path, doc_id=doc_id)

    print(f"      {len(pages)} pages parsed")



    print("[2/4] Chunking...")

    chunks = chunk_document(pages, doc_meta)

    text_n = sum(1 for c in chunks if c.chunk_type == "text")

    table_n = sum(1 for c in chunks if c.chunk_type == "table")

    print(f"      {len(chunks)} chunks ({text_n} text, {table_n} table)")



    print("[3/4] Enriching...")

    enrich_chunks(chunks)

    print(f"      contextual prefixes prepended to embedding_text")



    # ── 2. Write to backends ──

    print("[4/4] Writing to backends...")



    # SQLite

    conn = sqlite3.connect(str(DOCSTORE_PATH))

    try:

        _ensure_docstore_schema(conn)

        dropped = _docstore_clear_doc(conn, doc_id)

        _docstore_insert(conn, chunks)

        print(f"      SQLite:  -{dropped} stale, +{len(chunks)} new")

    finally:

        conn.close()



    # BM25

    bm25_state = _load_bm25_state()

    bm25_dropped = _bm25_drop_doc(bm25_state, doc_id)

    _bm25_append(bm25_state, chunks)

    _save_bm25_state(bm25_state)

    print(f"      BM25:    -{bm25_dropped} stale, +{len(chunks)} new "

          f"(corpus now {len(bm25_state['chunk_ids'])})")



    # Chroma (heaviest — load model only when needed)

    if embeddings is None:

        print("      (loading embedding model — first run downloads ~80MB)")

        embeddings = _get_embeddings()

    if chroma is None:

        chroma = _get_chroma(embeddings)

    chroma_dropped = _chroma_drop_doc(chroma, doc_id)

    # Embed in batches via tqdm for visibility

    batch_size = 32

    for i in tqdm(range(0, len(chunks), batch_size),

                  desc="      Chroma embedding", leave=False):

        _chroma_add(chroma, chunks[i:i + batch_size])

    print(f"      Chroma:  -{chroma_dropped} stale, +{len(chunks)} new")



    return {

        "doc_id": doc_id,

        "filename": filename,

        "pages": len(pages),

        "chunks": len(chunks),

        "text_chunks": text_n,

        "table_chunks": table_n,

    }





def ingest_all() -> List[dict]:

    """Ingest every PDF in DOCUMENTS. Loads embedding model once."""

    print("Loading embedding model (one-time, used across all docs)...")

    embeddings = _get_embeddings()

    chroma = _get_chroma(embeddings)



    results = []

    for filename in DOCUMENTS:

        result = ingest_document(filename, embeddings=embeddings, chroma=chroma)

        results.append(result)



    print(f"\n{'=' * 70}")

    print("INGESTION COMPLETE")

    print(f"{'=' * 70}")

    print(f"{'Doc':<12} {'Pages':>6} {'Chunks':>7} {'Text':>5} {'Table':>5}")

    for r in results:

        print(f"{r['doc_id']:<12} {r['pages']:>6} {r['chunks']:>7} "

              f"{r['text_chunks']:>5} {r['table_chunks']:>5}")



    # Final corpus totals

    conn = sqlite3.connect(str(DOCSTORE_PATH))

    try:

        total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

        by_doc = conn.execute(

            "SELECT doc_id, COUNT(*) FROM chunks GROUP BY doc_id"

        ).fetchall()

    finally:

        conn.close()



    print(f"\nDocstore now contains {total} chunks total:")

    for did, n in by_doc:

        print(f"  {did}: {n}")



    bm25_state = _load_bm25_state()

    print(f"BM25 corpus: {len(bm25_state['chunk_ids'])} chunks")

    print(f"Chroma collection '{CHROMA_COLLECTION}': "

          f"{chroma._collection.count()} embeddings")



    return results





# ──────────────────────────────────────────────────────────────────────────

# CLI

# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    import sys

    if len(sys.argv) > 1:

        # Single doc: python src/ingest.py cb7351en.pdf

        ingest_document(sys.argv[1])

    else:

        # All docs: python src/ingest.py

        ingest_all()