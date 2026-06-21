"""

Enricher — Anthropic-style contextual chunk enrichment.



What it does:

  For each chunk, build a 1-2 sentence context prefix from doc_title and

  section_path, then prepend it to embedding_text only. page_content stays

  unchanged so the LLM sees the original at retrieval time.



Why:

  Chunks lose context when split. "This index measures..." or "As shown

  above..." becomes meaningless out of context. Prepending doc + section

  context restores topical grounding for the embedding model without

  cluttering what the LLM actually reads.



  Anthropic's research showed 35-49% reduction in retrieval failures with

  this technique. Rule-based version (no LLM needed) using metadata we

  already have on every chunk.



Public API:

    enrich_chunks(chunks) -> List[Chunk]    # mutates and returns same list

"""



from __future__ import annotations



from typing import List



from chunker import Chunk





# ──────────────────────────────────────────────────────────────────────────

# Context prefix builders

# ──────────────────────────────────────────────────────────────────────────

def _build_text_context_prefix(chunk: Chunk) -> str:

    """

    Build the context prefix for a TEXT chunk.



    Format:

        [Context: This chunk is from "<doc_title>", section "<section>".]



    Falls back gracefully when section is "Front Matter" — in that case we

    say "from the front matter of" instead of naming a section, since

    "front matter" isn't a meaningful searchable section name.

    """

    if chunk.section == "Front Matter":

        return (

            f'[Context: This chunk is from the front matter of '

            f'"{chunk.doc_title}".]'

        )

    return (

        f'[Context: This chunk is from "{chunk.doc_title}", '

        f'section "{chunk.section}".]'

    )





def _build_table_context_prefix(chunk: Chunk) -> str:

    """

    Build the context prefix for a TABLE chunk.



    Tables already have rich embedding_text (caption + section + columns

    via _build_table_embedding_text in chunker.py). We just add the

    document title so cross-document table queries can disambiguate

    ("which year's TABLE 2?").



    Format:

        [Context: This table is from "<doc_title>".]

    """

    return f'[Context: This table is from "{chunk.doc_title}".]'





# ──────────────────────────────────────────────────────────────────────────

# Main entry point

# ──────────────────────────────────────────────────────────────────────────

def enrich_chunks(chunks: List[Chunk]) -> List[Chunk]:

    """

    Mutate each chunk in place: prepend a context prefix to embedding_text.

    page_content is NOT modified — the LLM still sees the original chunk

    at retrieval time.



    Returns the same list for chaining convenience.

    """

    for chunk in chunks:

        if chunk.chunk_type == "table":

            prefix = _build_table_context_prefix(chunk)

        else:

            prefix = _build_text_context_prefix(chunk)



        # Prepend with a blank line so the prefix is visually distinct in

        # the embedded string. Embedding models tokenize whitespace cheaply;

        # the readability of the prefix in logs matters more.

        chunk.embedding_text = f"{prefix}\n\n{chunk.embedding_text}"



    return chunks





# ──────────────────────────────────────────────────────────────────────────

# Test entry point

# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    import sys

    from config import DATA_DIR, DOCUMENTS

    from parser import parse_pdf

    from chunker import chunk_document



    target_filename = sys.argv[1] if len(sys.argv) > 1 else "cb7351en.pdf"

    doc_meta = DOCUMENTS[target_filename]

    pdf_path = DATA_DIR / target_filename



    print(f"\n{'=' * 70}")

    print(f"Enriching: {target_filename} ({doc_meta['doc_id']})")

    print(f"{'=' * 70}\n")



    pages = parse_pdf(pdf_path, doc_id=doc_meta["doc_id"])

    chunks = chunk_document(pages, doc_meta)

    chunks = enrich_chunks(chunks)



    text_chunks = [c for c in chunks if c.chunk_type == "text"]

    table_chunks = [c for c in chunks if c.chunk_type == "table"]

    print(f"Total enriched chunks: {len(chunks)}")

    print(f"  Text chunks:  {len(text_chunks)}")

    print(f"  Table chunks: {len(table_chunks)}\n")



    # Show before/after for a few text chunks across different sections

    print(f"\n{'─' * 70}")

    print("Sample enriched text chunks (showing embedding_text vs page_content):")

    print(f"{'─' * 70}")



    # Pick one from front matter, one from a real section, one mid-doc

    samples = []

    seen_sections = set()

    for c in text_chunks:

        if c.section not in seen_sections:

            samples.append(c)

            seen_sections.add(c.section)

        if len(samples) >= 3:

            break



    for c in samples:

        print(f"\n[{c.chunk_id}]")

        print(f"  Page: {c.page} | Section: {c.section}")

        print(f"  embedding_text (what gets embedded):")

        preview = c.embedding_text[:400].replace("\n", " ")

        print(f"    {preview}{'...' if len(c.embedding_text) > 400 else ''}")

        print(f"  page_content (what the LLM sees — should be UNCHANGED):")

        preview = c.page_content[:200].replace("\n", " ")

        print(f"    {preview}{'...' if len(c.page_content) > 200 else ''}")



    # Show all table chunks

    print(f"\n{'─' * 70}")

    print("All enriched table chunks:")

    print(f"{'─' * 70}")

    for c in table_chunks:

        print(f"\n[{c.chunk_id}]")

        print(f"  Page: {c.page} | Section: {c.section}")

        print(f"  Caption: {c.table_caption}")

        print(f"  embedding_text:")

        print(f"    {c.embedding_text}")

        print(f"  page_content preview:")

        preview = c.page_content[:200].replace("\n", " ")

        print(f"    {preview}...")



    # Sanity check: page_content should be unchanged

    print(f"\n{'─' * 70}")

    print("Sanity check — page_content vs embedding_text should differ:")

    print(f"{'─' * 70}")

    sample = text_chunks[0] if text_chunks else None

    if sample:

        same = sample.page_content == sample.embedding_text

        print(f"  page_content == embedding_text? {same}  "

              f"(expected False)")

        if same:

            print("  WARNING: enricher did not modify embedding_text!")

        else:

            print(f"  page_content length:    {len(sample.page_content)}")

            print(f"  embedding_text length:  {len(sample.embedding_text)}")

            print(f"  diff (prefix added):    "

                  f"{len(sample.embedding_text) - len(sample.page_content)} chars")