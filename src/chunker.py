"""

Chunker — turns parsed PageContent into Chunk objects ready for embedding.



Chunk types:

  - text:  body prose, split by RecursiveCharacterTextSplitter, section-aware

  - table: one full table = one atomic chunk (never split)



Every chunk carries rich metadata for citations and UI rendering.



Public API:

    chunk_document(pages, doc_meta) -> List[Chunk]

"""



from __future__ import annotations



import json

import re

from dataclasses import dataclass, field, asdict

from typing import List, Optional



from langchain_text_splitters import RecursiveCharacterTextSplitter



from config import (

    DOCUMENTS,

    TEXT_CHUNK_OVERLAP,

    TEXT_CHUNK_SIZE,

)

from parser import ExtractedTable, PageContent





# ──────────────────────────────────────────────────────────────────────────

# Data class

# ──────────────────────────────────────────────────────────────────────────

@dataclass

class Chunk:

    """A single chunk ready for embedding + retrieval."""

    chunk_id: str

    doc_id: str

    doc_title: str

    doc_year: int

    page: int

    section: str

    chunk_type: str               # "text" or "table"



    # The content the LLM will see when this chunk is retrieved

    page_content: str



    # The content we EMBED (may differ from page_content for tables)

    embedding_text: str



    # Table-only fields (empty strings for text chunks, for Chroma compatibility)

    table_caption: str = ""

    table_json: str = ""



    def to_metadata(self) -> dict:

        """

        Return only the fields safe to store in Chroma metadata.

        Chroma requires str/int/float/bool — no nested dicts, no lists.

        page_content and embedding_text are NOT included here; they live

        elsewhere (page_content in Chroma's document, embedding_text in

        embedding pipeline, raw chunk content in SQLite docstore).

        """

        return {

            "chunk_id": self.chunk_id,

            "doc_id": self.doc_id,

            "doc_title": self.doc_title,

            "doc_year": self.doc_year,

            "page": self.page,

            "section": self.section,

            "chunk_type": self.chunk_type,

            "table_caption": self.table_caption,

            "table_json": self.table_json,

        }





# ──────────────────────────────────────────────────────────────────────────

# Helpers

# ──────────────────────────────────────────────────────────────────────────

def _clean_section_path(section: str) -> str:

    """

    Trim noisy prefixes from section paths.

    E.g., 'AGRICULTURE > THE STATE OF > SUMMARY > Foo' → 'SUMMARY > Foo'

    The 'AGRICULTURE > THE STATE OF' part comes from cover-page title text

    that the section tracker initially picked up.

    """

    if not section or section == "Unknown Section":

        return "Front Matter"

    # Strip a leading "AGRICULTURE > THE STATE OF > " if present

    prefix = "AGRICULTURE > THE STATE OF > "

    if section.startswith(prefix):

        section = section[len(prefix):]

    # If after stripping nothing useful remains, fall back

    if not section.strip():

        return "Front Matter"

    return section.strip()





def _make_chunk_id(doc_id: str, page: int, chunk_type: str, ordinal: int) -> str:

    """Deterministic chunk ID. Stable across re-ingestion."""

    return f"{doc_id}_p{page:03d}_{chunk_type}_{ordinal}"





def _build_table_embedding_text(table: ExtractedTable, section: str) -> str:

    """

    Build the semantic text we EMBED for a table chunk.

    Includes the caption, section context, and column headers — the things

    users actually search for. Excludes the numeric cell content (noise for

    embedding similarity).

    """

    parts = [table.caption]

    if section and section != "Front Matter":

        parts.append(f"Section: {section}")

    # Include the header row (semantic context, not numeric noise)

    if table.rows:

        header = table.rows[0]

        parts.append("Columns: " + " | ".join(str(c) for c in header))

    return ". ".join(parts)





def _build_table_page_content(table: ExtractedTable) -> str:

    """

    What the LLM sees when this table chunk is retrieved.

    Caption + markdown rendering = LLM gets full numeric content.

    """

    return f"{table.caption}\n\n{table.to_markdown()}"





# ──────────────────────────────────────────────────────────────────────────

# Section-aware text chunking

# ──────────────────────────────────────────────────────────────────────────

def _group_pages_by_section(pages: List[PageContent]) -> List[dict]:

    """

    Group consecutive pages with the same section into one block.

    Empty pages (text=='') are skipped (these are the figure-only pages).



    Returns a list of dicts:

        { "section": ..., "pages": [list of page numbers],

          "text": "concatenated text" }

    """

    groups: List[dict] = []

    current = None



    for p in pages:

        if not p.text.strip():

            # Skipped page — close any open group

            if current is not None:

                groups.append(current)

                current = None

            continue



        cleaned_section = _clean_section_path(p.section)



        if current is None or current["section"] != cleaned_section:

            # Start a new group

            if current is not None:

                groups.append(current)

            current = {

                "section": cleaned_section,

                "pages": [p.page],

                "text": p.text,

            }

        else:

            # Same section as previous page — append

            current["pages"].append(p.page)

            current["text"] += "\n\n" + p.text



    if current is not None:

        groups.append(current)



    return groups





def _split_text_into_chunks(text: str) -> List[str]:

    """Split a block of text into overlapping chunks."""

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=TEXT_CHUNK_SIZE,

        chunk_overlap=TEXT_CHUNK_OVERLAP,

        # Try to split on paragraph, then sentence, then word boundaries

        separators=["\n\n", "\n", ". ", " ", ""],

        keep_separator=False,

    )

    return splitter.split_text(text)





def _assign_text_chunks_to_pages(

    chunks: List[str], group: dict

) -> List[tuple[int, str]]:

    """

    A section group may span multiple pages. For each chunk, figure out

    which page it primarily came from so the citation points to the right page.



    Strategy: walk through the concatenated source text. For each chunk,

    locate it in the source, then map that position back to a page number

    using a per-page character offset table.



    If a chunk straddles a page boundary, attribute it to the page where it

    starts (this is the most useful for citations — the user can scroll

    forward).

    """

    pages_in_group = group["pages"]

    raw_pages_text = group["text"]



    # Build a page-offset table:

    # We re-segment the group's text into per-page pieces using "\n\n" joins.

    # The grouping was done as `current["text"] += "\n\n" + p.text` so the

    # boundaries are predictable.

    # But to be safe, we just search for the chunk substring inside the

    # full group text and use that position.

    # If the chunk isn't found verbatim (because splitter trimmed whitespace),

    # we fall back to the first page in the group.



    # Build [(start_offset, page_num)] for each page

    page_offsets: List[tuple[int, int]] = []

    cursor = 0

    page_texts = raw_pages_text.split("\n\n")

    # NOTE: this split is imperfect because page text itself may contain "\n\n".

    # We use a more reliable method: build the offsets at concatenation time

    # would be ideal, but for simplicity we use the search-based fallback.

    # In practice the chunk almost always falls inside one page.



    # Simpler reliable method: keep a separate index of per-page boundaries.

    # We rebuild what _group_pages_by_section did, tracking offsets.

    boundaries: List[tuple[int, int]] = []  # (char_offset, page_num)

    offset = 0

    # Reconstruct: first page added directly, subsequent prefixed with "\n\n"

    # (matching the join logic in _group_pages_by_section)

    # We need original per-page texts — which we don't have here directly.

    # Workaround: split by "\n\n" and walk along.

    # If a page boundary is between text pages, "\n\n" was the joiner.

    # We can't perfectly recover that, so we attribute each chunk by

    # finding its starting char position and looking up which page it falls in

    # based on the cumulative lengths of `page_texts`.

    cum = 0

    # We rebuild boundaries using a simple heuristic: divide the cumulative

    # text length equally? No — we know join used "\n\n" so we can split.

    # We use the original page_texts list lengths in order.

    for i, page_num in enumerate(pages_in_group):

        boundaries.append((cum, page_num))

        # advance cum by the length of this page's text plus the joiner

        if i < len(page_texts):

            cum += len(page_texts[i])

            if i > 0:

                cum += 2  # for the "\n\n" joiner



    def page_for_offset(off: int) -> int:

        chosen = pages_in_group[0]

        for boundary_off, page_num in boundaries:

            if off >= boundary_off:

                chosen = page_num

            else:

                break

        return chosen



    out: List[tuple[int, str]] = []

    search_start = 0

    for chunk in chunks:

        # Search for the chunk in the source from the current cursor

        # (a chunk's first ~50 chars are usually unique enough)

        probe = chunk[:60].strip()

        pos = raw_pages_text.find(probe, search_start) if probe else -1

        if pos == -1:

            # Fall back: attribute to first page in group

            pg = pages_in_group[0]

        else:

            pg = page_for_offset(pos)

            search_start = pos + 1

        out.append((pg, chunk))



    return out





# ──────────────────────────────────────────────────────────────────────────

# Main entry point

# ──────────────────────────────────────────────────────────────────────────

def chunk_document(pages: List[PageContent], doc_meta: dict) -> List[Chunk]:

    """

    Turn one document's parsed pages into a list of Chunk objects.



    doc_meta is one entry from config.DOCUMENTS, e.g.:

        {"doc_id": "sofa_2021", "title": "...", "year": 2021, ...}

    """

    doc_id = doc_meta["doc_id"]

    doc_title = doc_meta["title"]

    doc_year = doc_meta["year"]



    chunks: List[Chunk] = []

    text_ordinal_by_page: dict[int, int] = {}

    table_ordinal_by_page: dict[int, int] = {}



    # ── TEXT CHUNKS ──

    groups = _group_pages_by_section(pages)

    for group in groups:

        raw_chunks = _split_text_into_chunks(group["text"])

        chunk_page_pairs = _assign_text_chunks_to_pages(raw_chunks, group)



        for page_num, text in chunk_page_pairs:

            text = text.strip()

            if not text:

                continue

            ord_for_page = text_ordinal_by_page.get(page_num, 0)

            text_ordinal_by_page[page_num] = ord_for_page + 1



            chunk_id = _make_chunk_id(doc_id, page_num, "text", ord_for_page)

            chunks.append(Chunk(

                chunk_id=chunk_id,

                doc_id=doc_id,

                doc_title=doc_title,

                doc_year=doc_year,

                page=page_num,

                section=group["section"],

                chunk_type="text",

                page_content=text,

                embedding_text=text,

            ))



    # ── TABLE CHUNKS ──

    for page in pages:

        if not page.tables:

            continue

        cleaned_section = _clean_section_path(page.section)

        for table in page.tables:

            ord_for_page = table_ordinal_by_page.get(page.page, 0)

            table_ordinal_by_page[page.page] = ord_for_page + 1



            chunk_id = _make_chunk_id(doc_id, page.page, "table", ord_for_page)

            chunks.append(Chunk(

                chunk_id=chunk_id,

                doc_id=doc_id,

                doc_title=doc_title,

                doc_year=doc_year,

                page=page.page,

                section=cleaned_section,

                chunk_type="table",

                page_content=_build_table_page_content(table),

                embedding_text=_build_table_embedding_text(table, cleaned_section),

                table_caption=table.caption,

                table_json=json.dumps(table.to_dict(), ensure_ascii=False),

            ))



    return chunks





# ──────────────────────────────────────────────────────────────────────────

# Test entry point

# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    import sys

    from config import DATA_DIR

    from parser import parse_pdf



    # Default to SOFA 2021

    target_filename = sys.argv[1] if len(sys.argv) > 1 else "cb7351en.pdf"

    doc_meta = DOCUMENTS[target_filename]

    pdf_path = DATA_DIR / target_filename



    print(f"\n{'=' * 70}")

    print(f"Chunking: {target_filename} ({doc_meta['doc_id']})")

    print(f"{'=' * 70}\n")



    pages = parse_pdf(pdf_path, doc_id=doc_meta["doc_id"])

    chunks = chunk_document(pages, doc_meta)



    text_chunks = [c for c in chunks if c.chunk_type == "text"]

    table_chunks = [c for c in chunks if c.chunk_type == "table"]

    print(f"Total chunks: {len(chunks)}")

    print(f"  Text chunks:  {len(text_chunks)}")

    print(f"  Table chunks: {len(table_chunks)}\n")



    # Show first 3 text chunks

    print(f"\n{'─' * 70}")

    print("Sample text chunks:")

    print(f"{'─' * 70}")

    for c in text_chunks[:3]:

        print(f"\n[{c.chunk_id}]")

        print(f"  Page: {c.page} | Section: {c.section}")

        print(f"  Content ({len(c.page_content)} chars):")

        preview = c.page_content[:250].replace("\n", " ")

        print(f"  {preview}...")



    # Show all table chunks (there are only 2-3)

    print(f"\n{'─' * 70}")

    print("All table chunks:")

    print(f"{'─' * 70}")

    for c in table_chunks:

        print(f"\n[{c.chunk_id}]")

        print(f"  Page: {c.page} | Section: {c.section}")

        print(f"  Caption: {c.table_caption}")

        print(f"  Embedding text:")

        print(f"    {c.embedding_text}")

        print(f"  page_content preview ({len(c.page_content)} chars):")

        print("    " + c.page_content[:300].replace("\n", " ") + "...")



    # Distribution by page

    print(f"\n{'─' * 70}")

    print("Chunks per page:")

    print(f"{'─' * 70}")

    by_page: dict[int, int] = {}

    for c in chunks:

        by_page[c.page] = by_page.get(c.page, 0) + 1

    for page in sorted(by_page.keys()):

        print(f"  Page {page:3d}: {by_page[page]} chunk(s)")