"""

PDF parsing module — v2 with improved section detection and table filtering.



Strategy:

  - PyMuPDF (fitz) extracts text blocks with bounding boxes + font sizes.

  - pdfplumber extracts tables, filtered aggressively.

  - Table bounding boxes are masked out of the text pass.

  - Headers/footers/page-numbers stripped via regex + heuristics.

  - Dropcap letters are stitched back into following text.



Public API:

    parse_pdf(pdf_path) -> List[PageContent]

"""



from __future__ import annotations



import re

from dataclasses import dataclass, field

from pathlib import Path

from typing import List, Optional, Tuple



import fitz  # PyMuPDF

import pdfplumber

from config import (

    HEADING_MIN_FONT_SIZE,

    NOISE_PATTERNS,
    SKIP_TEXT_PAGES,
    SECTION_SKIP_PAGES,
    TOP_LEVEL_SECTIONS 

)



_NOISE_REGEXES = [re.compile(p, re.IGNORECASE) for p in NOISE_PATTERNS]





# ──────────────────────────────────────────────────────────────────────────

# Data classes

# ──────────────────────────────────────────────────────────────────────────

@dataclass

class ExtractedTable:

    page: int

    caption: str

    rows: List[List[str]]

    bbox: Tuple[float, float, float, float]



    def to_markdown(self) -> str:

        if not self.rows:

            return ""

        header = self.rows[0]

        body = self.rows[1:] if len(self.rows) > 1 else []

        md = "| " + " | ".join(_clean_cell(c) for c in header) + " |\n"

        md += "| " + " | ".join("---" for _ in header) + " |\n"

        for row in body:

            padded = row + [""] * (len(header) - len(row))

            md += "| " + " | ".join(_clean_cell(c) for c in padded[: len(header)]) + " |\n"

        return md.strip()



    def to_dict(self) -> dict:

        return {

            "caption": self.caption,

            "page": self.page,

            "rows": [[_clean_cell(c) for c in row] for row in self.rows],

        }





@dataclass

class PageContent:

    page: int

    section: str

    text: str

    tables: List[ExtractedTable] = field(default_factory=list)





# ──────────────────────────────────────────────────────────────────────────

# Helpers

# ──────────────────────────────────────────────────────────────────────────

def _clean_cell(cell: Optional[str]) -> str:

    if cell is None:

        return ""

    return re.sub(r"\s+", " ", str(cell)).strip()





def _is_noise_line(line: str) -> bool:
    """Drop page numbers, repeated section headers, dropcaps, and caption lines."""
    stripped = line.strip()
    if not stripped:
        return True
    # Pure digit strings (page numbers, TOC entries)
    if re.fullmatch(r"\d{1,3}", stripped):
        return True
    # Single character (orphan dropcaps that didn't get stitched)
    if len(stripped) == 1:
        return True
    # FIGURE/TABLE caption lines should not appear in body text — they belong
    # with the figure/table itself, or with the table caption we extracted.
    # Catches both:
    #   "FIGURE 3 Spectrum of land degradation..."
    #   "TABLE 1 INDICATORS OF RESILIENCE..."
    if re.match(r"^(FIGURE|TABLE)(\s+FROM\s+BOX)?\s+\d+\b", stripped, re.IGNORECASE):
        return True
    for rgx in _NOISE_REGEXES:
        if rgx.search(stripped):
            return True
    return False



def _bbox_overlaps(bbox_a: Tuple, bbox_b: Tuple, tol: float = 2.0) -> bool:

    ax0, ay0, ax1, ay1 = bbox_a

    bx0, by0, bx1, by1 = bbox_b

    return not (

        ax1 < bx0 - tol

        or bx1 < ax0 - tol

        or ay1 < by0 - tol

        or by1 < ay0 - tol

    )



def _is_caption_line(text: str) -> bool:

    """

    Detect caption lines like 'FIGURE 3 ...', 'TABLE 1 ...', 'FIGURE FROM BOX 1 ...'.

    Used to drop entire text blocks that start with a caption.

    """

    stripped = text.strip()

    return bool(

        re.match(r"^(FIGURE|TABLE)(\s+FROM\s+BOX)?\s+\d+\b", stripped, re.IGNORECASE)

    )






def _is_valid_heading(text: str, font_size: float) -> bool:

    """

    Stricter heading detection.

    Rejects dropcap letters, page numbers, and tiny fragments.

    """

    text = text.strip()

    if not text:

        return False

    if font_size < HEADING_MIN_FONT_SIZE:

        return False

    # Reject pure numbers (page numbers in big fonts)

    if re.fullmatch(r"\d+", text):

        return False

    # Reject single characters (dropcaps)

    if len(text) <= 2:

        return False

    # Reject very short fragments (must be at least one real word)

    if len(text.split()) < 1 or not re.search(r"[A-Za-z]{3,}", text):

        return False

    return True





# ──────────────────────────────────────────────────────────────────────────

# Section tracker

# ──────────────────────────────────────────────────────────────────────────

class SectionTracker:

    """

    Tracks current heading path using font sizes.

    Larger font = higher level. Stack approach for hierarchy.



    Special case: TOP_LEVEL_SECTIONS (FOREWORD, SUMMARY, etc.) are peers.

    When one appears, the entire stack is cleared before pushing it, so

    FOREWORD never nests under (or above) SUMMARY.

    """



    def __init__(self):

        self.stack: List[Tuple[float, str]] = []



    def update(self, text: str, font_size: float) -> None:

        if not _is_valid_heading(text, font_size):

            return

        text = text.strip()



        # Peer-level top sections: clear the stack and push as root.

        if text.upper() in TOP_LEVEL_SECTIONS:

            self.stack = [(font_size, text.upper())]

            return



        # Normal hierarchy: pop entries with smaller or equal font size,

        # then push.

        while self.stack and self.stack[-1][0] <= font_size:

            self.stack.pop()

        self.stack.append((font_size, text))



    def current_path(self) -> str:

        if not self.stack:

            return "Unknown Section"

        return " > ".join(h for _, h in self.stack)






# ──────────────────────────────────────────────────────────────────────────

# Text extraction with dropcap stitching

# ──────────────────────────────────────────────────────────────────────────

def _extract_text_from_page(

    page: fitz.Page,

    section_tracker: SectionTracker,

    table_bboxes: List[Tuple],

    update_sections: bool = True,

) -> str:

    """

    Extract clean body text.



    Skips:

      - blocks inside table regions (bbox overlap with detected tables)

      - blocks whose first line is a FIGURE/TABLE caption (drops captions +

        their continuation lines, notes, source attributions, etc.)

      - noise lines (page numbers, dropcaps, repeated headers)



    Heading handling:

      Consecutive lines within the same block at a heading-eligible font size

      are MERGED into a single heading string before being pushed to the

      section tracker. This handles FAO's multi-line wrapped titles like

      "UNDERSTANDING SYSTEMS' FUNCTIONS AND VULNERABILITIES" which span 3

      visual lines but are logically one heading.



    Dropcap letters are stitched into the following line before noise filtering.



    The `update_sections` flag lets the caller suppress section-tracker

    updates on cover/copyright/TOC pages whose large-font text isn't a

    real section heading.

    """

    blocks = page.get_text("dict")["blocks"]

    raw_lines: List[Tuple[str, float]] = []  # (text, font_size) for body text



    for block in blocks:

        if block.get("type") != 0:  # 0 = text block

            continue

        block_bbox = block["bbox"]



        # Skip blocks inside detected tables / figure regions

        if any(_bbox_overlaps(block_bbox, tb) for tb in table_bboxes):

            continue



        # Collect this block's lines as (text, font_size)

        block_lines: List[Tuple[str, float]] = []

        for line in block.get("lines", []):

            line_text = ""

            max_font_size = 0.0

            for span in line.get("spans", []):

                line_text += span["text"]

                if span["size"] > max_font_size:

                    max_font_size = span["size"]

            line_text = line_text.strip()

            if line_text:

                block_lines.append((line_text, max_font_size))



        if not block_lines:

            continue



        # ── CAPTION BLOCK CHECK ──

        # If the first non-empty line is a FIGURE/TABLE caption, drop the

        # entire block (caption + continuation + notes/source attributions).

        first_line = block_lines[0][0]

        if _is_caption_line(first_line):

            continue



        # ── HEADING MERGE PASS ──

        # Walk block_lines and merge consecutive lines that share a

        # heading-eligible font size into a single heading. This fixes the

        # multi-line title bug where each visual line of a wrapped heading

        # was treated as a separate heading and only the last one survived.

        merged: List[Tuple[str, float]] = []

        i = 0

        while i < len(block_lines):

            text, fs = block_lines[i]

            if fs >= HEADING_MIN_FONT_SIZE:

                # Start a heading run. Greedily absorb following lines whose

                # font size is essentially the same (tolerance 0.5pt covers

                # minor rendering noise across spans).

                heading_parts = [text]

                j = i + 1

                while j < len(block_lines):

                    next_text, next_fs = block_lines[j]

                    if (

                        next_fs >= HEADING_MIN_FONT_SIZE

                        and abs(next_fs - fs) < 0.5

                    ):

                        heading_parts.append(next_text)

                        j += 1

                    else:

                        break

                merged_text = " ".join(heading_parts).strip()

                # Use the largest font size seen in the run (they're ~equal

                # anyway, but be defensive).

                merged.append((merged_text, fs))

                i = j

            else:

                merged.append((text, fs))

                i += 1



        # ── PROCESS MERGED LINES ──

        # For each merged line: if it's heading-eligible, push to the

        # section tracker (when allowed). Either way, keep it in raw_lines

        # so it appears in the page text (the chunker will see headings

        # inline with body, which is what we want for context).

        for line_text, font_size in merged:

            if update_sections:

                section_tracker.update(line_text, font_size)

            raw_lines.append((line_text, font_size))



    # ── DROPCAP STITCHING ──

    # FAO PDFs render the first letter of some paragraphs as a large dropcap

    # in its own line. Stitch single uppercase letters into the following

    # lowercase-starting line before noise filtering (otherwise the dropcap

    # gets dropped as a single character and the next line loses its first

    # letter).

    stitched: List[Tuple[str, float]] = []

    i = 0

    while i < len(raw_lines):

        cur_text, cur_size = raw_lines[i]

        nxt_text = raw_lines[i + 1][0] if i + 1 < len(raw_lines) else ""



        if (

            len(cur_text) == 1

            and cur_text.isupper()

            and cur_text.isalpha()

            and nxt_text

            and nxt_text[0].islower()

        ):

            stitched.append((cur_text + nxt_text, cur_size))

            i += 2

        else:

            stitched.append((cur_text, cur_size))

            i += 1



    # ── NOISE FILTER ──

    # Line-level safety net for page numbers, orphan dropcaps that didn't

    # get stitched, and any caption fragments that slipped past the

    # block-level check.

    final = [t for t, _ in stitched if not _is_noise_line(t)]



    return "\n".join(final)





# ──────────────────────────────────────────────────────────────────────────

# Public API

# ──────────────────────────────────────────────────────────────────────────

def parse_pdf(pdf_path: str | Path, doc_id: str | None = None) -> List[PageContent]:

    """

    Parse a PDF end-to-end.



    Tables are extracted via custom per-table functions in table_extractors.py.

    Each function uses word-level coordinates (not generic line detection)

    so complex layouts (merged cells, bulleted cell content, section bands)

    are handled accurately.



    Bounding boxes from pdfplumber's generic table finder are still used to

    mask figure/table regions from text extraction (so visual content doesn't

    leak into body prose).

    """

    from table_extractors import extract_tables_for_doc



    pdf_path = Path(pdf_path)

    if not pdf_path.exists():

        raise FileNotFoundError(f"PDF not found: {pdf_path}")



    # Run custom extractors

    custom_tables = extract_tables_for_doc(doc_id, pdf_path) if doc_id else []

    tables_by_page: dict[int, List[ExtractedTable]] = {}

    for t in custom_tables:

        tables_by_page.setdefault(t.page, []).append(t)



    pages_out: List[PageContent] = []

    section_tracker = SectionTracker()

    skip_pages = SKIP_TEXT_PAGES.get(doc_id,set()) if doc_id else set()
    section_skip_pages = SECTION_SKIP_PAGES.get(doc_id,set()) if doc_id else set()

    with fitz.open(pdf_path) as doc:

        for page_idx, page in enumerate(doc, start=1):

            mask_bboxes = _get_figure_bboxes(pdf_path, page_idx)

            if page_idx in skip_pages:
                text = ""
            else :
                text = _extract_text_from_page(page, section_tracker, mask_bboxes,update_sections=(page_idx not in section_skip_pages))

            current_section = section_tracker.current_path()



            pages_out.append(

                PageContent(

                    page=page_idx,

                    section=current_section,

                    text=text,

                    tables=tables_by_page.get(page_idx, []),

                )

            )



    return pages_out





def _get_figure_bboxes(pdf_path: Path, page_num: int) -> List[Tuple]:

    """

    Get bounding boxes of any visual table-like region (charts, real tables).

    Only the bboxes are used — for masking these regions out of text extraction.

    """

    bboxes: List[Tuple] = []

    with pdfplumber.open(pdf_path) as pdf:

        page = pdf.pages[page_num - 1]

        table_objects = page.find_tables(

            table_settings={

                "vertical_strategy": "lines",

                "horizontal_strategy": "lines",

                "snap_tolerance": 3,

            }

        )

        for tbl_obj in table_objects:

            bboxes.append(tbl_obj.bbox)

    return bboxes





# ──────────────────────────────────────────────────────────────────────────

# Test entry point

# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    import sys

    from config import DATA_DIR



    target = sys.argv[1] if len(sys.argv) > 1 else "cb7351en.pdf"

    pdf_path = DATA_DIR / target



    print(f"\n{'=' * 70}")

    print(f"Parsing: {pdf_path.name}")

    print(f"{'=' * 70}\n")



    pages = parse_pdf(pdf_path)

    print(f"Total pages parsed: {len(pages)}\n")



    text_chars_total = sum(len(p.text) for p in pages)

    tables_total = sum(len(p.tables) for p in pages)

    print(f"Total text characters: {text_chars_total:,}")

    print(f"Total tables extracted: {tables_total}\n")



    for p in pages[:6]:

        print(f"\n--- Page {p.page} ---")

        print(f"Section: {p.section}")

        print(f"Text preview ({len(p.text)} chars):")

        print(p.text[:400] + ("..." if len(p.text) > 400 else ""))

        if p.tables:

            print(f"\n  Tables on this page: {len(p.tables)}")

            for t in p.tables:

                print(f"    - {t.caption} [{len(t.rows)} rows x "

                      f"{max(len(r) for r in t.rows) if t.rows else 0} cols]")



    print(f"\n\n{'=' * 70}")

    print("All pages with tables (for verification):")

    print(f"{'=' * 70}")

    for p in pages:

        for t in p.tables:

            print(f"  Page {p.page:3d} | {t.caption} | {len(t.rows)} rows")