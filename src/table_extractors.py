"""

Custom per-table extractors for the FAO SOFA PDFs.



Why custom?

  Generic table detection (pdfplumber's line-detection, lattice mode) fails

  on these tables because:

    - TABLE 2 (2021, p18): merged header cells spanning multiple columns

    - TABLE 5 (2021, p25): cells contain bulleted lists with line breaks

    - TABLE 3 (2025, p22): section-band rows spanning all columns



  Each function below:

    1. Reads ALL words on the target page with precise (x, y) coordinates

    2. Filters to words inside the table's bounding region

    3. Buckets each word into a (row, col) cell using empirically defined

       column boundaries and row boundaries

    4. Concatenates words within each cell in reading order



  Column boundaries are in PDF points (1 inch = 72 points).

  Coordinate system: origin top-left, y increases downward.

"""



from __future__ import annotations

from typing import List, Optional

import pdfplumber

from pathlib import Path



from parser import ExtractedTable





# ──────────────────────────────────────────────────────────────────────────

# Generic helpers

# ──────────────────────────────────────────────────────────────────────────

def _get_words(pdf_path: Path, page_num: int) -> list[dict]:

    """Get all words on a page with their coordinates."""

    with pdfplumber.open(pdf_path) as pdf:

        page = pdf.pages[page_num - 1]

        return page.extract_words(

            x_tolerance=2,

            y_tolerance=2,

            keep_blank_chars=False,

        )





def _words_in_region(words: list[dict], x0: float, x1: float,

                     y_top: float, y_bottom: float) -> list[dict]:

    """Return words whose center falls inside the rectangle."""

    out = []

    for w in words:

        cx = (w["x0"] + w["x1"]) / 2

        cy = (w["top"] + w["bottom"]) / 2

        if x0 <= cx <= x1 and y_top <= cy <= y_bottom:

            out.append(w)

    return out





def _bucket_into_cell(

    words: list[dict],

    col_bounds: list[tuple[float, float]],

    row_bounds: list[tuple[float, float]],

) -> list[list[str]]:

    """

    Bucket every word into the right (row, col) cell.



    col_bounds: list of (x_start, x_end) for each column

    row_bounds: list of (y_top, y_bottom) for each row



    Returns: list of rows, each row a list of cell strings.

    """

    n_rows = len(row_bounds)

    n_cols = len(col_bounds)

    grid: list[list[list[dict]]] = [

        [[] for _ in range(n_cols)] for _ in range(n_rows)

    ]



    for w in words:

        cx = (w["x0"] + w["x1"]) / 2

        cy = (w["top"] + w["bottom"]) / 2



        # Find row

        row_idx: Optional[int] = None

        for ri, (rt, rb) in enumerate(row_bounds):

            if rt <= cy <= rb:

                row_idx = ri

                break

        if row_idx is None:

            continue



        # Find column

        col_idx: Optional[int] = None

        for ci, (cs, ce) in enumerate(col_bounds):

            if cs <= cx <= ce:

                col_idx = ci

                break

        if col_idx is None:

            continue



        grid[row_idx][col_idx].append(w)



    # Convert each cell's word list into a single string, in reading order

    result: list[list[str]] = []

    for row in grid:

        row_out = []

        for cell_words in row:

            # Sort by top first (line order), then x0 (reading order)

            cell_words.sort(key=lambda w: (round(w["top"]), w["x0"]))

            # Group words into lines by their `top` coordinate

            lines: list[list[str]] = []

            current_line: list[str] = []

            last_top: Optional[float] = None

            for w in cell_words:

                if last_top is None or abs(w["top"] - last_top) <= 3:

                    current_line.append(w["text"])

                else:

                    lines.append(current_line)

                    current_line = [w["text"]]

                last_top = w["top"]

            if current_line:

                lines.append(current_line)

            text = " ".join(" ".join(line) for line in lines).strip()

            row_out.append(text)

        result.append(row_out)

    return result





# ──────────────────────────────────────────────────────────────────────────

# TABLE 2 (SOFA 2021, page 18): Indicators of Unaffordability of Healthy Diets

# 5 columns: Region | Unaffordable %  | Unaffordable Total | At-risk % | At-risk Total

# ──────────────────────────────────────────────────────────────────────────

def extract_2021_table2(pdf_path: Path) -> ExtractedTable:

    """

    SOFA 2021 — TABLE 2 (page 18): Indicators of Unaffordability of Healthy Diets.



    Page dims: 453.5 × 623.6 (FAO In Brief booklet size).

    Table spans y ≈ 92 (header start) → y ≈ 386 (last data row).

    5 columns at x: [51–142], [169–184], [226–250], [297–311], [359–378].

    Data rows are at known y-centers (data sits on second line when label wraps).

    """

    words = _get_words(pdf_path, page_num=18)



    # Real column x ranges (from word inspection)

    col_bounds = [

        (45.0, 165.0),    # Region/income-group label (numbers start at x≈169)

        (165.0, 195.0),   # Percent (unable to afford)

        (220.0, 280.0),   # Total millions (unable to afford)

        (290.0, 325.0),   # Percent (at risk)

        (355.0, 395.0),   # Total millions (at risk)

    ]



    # Known data-row y-centers (where the NUMBERS sit).

    # Label text on the first column may span multiple lines above and below

    # the center — we'll widen the row band on column 1 to capture wrapped labels.

    data_row_centers = [

        ("WORLD",                          153.8),

        ("Central Asia",                   168.8),

        ("Eastern and South-eastern Asia", 187.0),

        ("Europe",                         205.2),

        ("Latin America and the Caribbean", 223.4),

        ("Northern Africa and Western Asia", 244.8),

        ("Northern America",               262.7),

        ("Oceania",                        277.1),

        ("Southern Asia",                  291.5),

        ("Sub-Saharan Africa",             306.2),

        ("Low-income",                     336.2),

        ("Lower-middle-income",            351.2),

        ("Upper-middle-income",            366.2),

        ("High-income",                    380.4),

    ]



    header = [

        "Region / Income group",

        "Unable to afford healthy diet 2019 — Percent",

        "Unable to afford healthy diet 2019 — Total (millions)",

        "At risk if income reduced by 1/3 — Percent",

        "At risk if income reduced by 1/3 — Total (millions)",

    ]



    rows = [header]



    for label_hint, y_center in data_row_centers:

        row_cells = []



        # For column 1 (region label), use a wide vertical band to capture

        # multi-line wrapped labels. Numbers are tightly aligned to y_center.

        for col_idx, (cx0, cx1) in enumerate(col_bounds):

            if col_idx == 0:

                # Label column — widen y range to grab wrapped lines

                y_top = y_center - 11.0

                y_bot = y_center + 11.0

            else:

                # Number columns — tight y range (numbers are on one line)

                y_top = y_center - 5.0

                y_bot = y_center + 5.0



            cell_words = []

            for w in words:

                wcx = (w["x0"] + w["x1"]) / 2

                wcy = (w["top"] + w["bottom"]) / 2

                if cx0 <= wcx <= cx1 and y_top <= wcy <= y_bot:

                    cell_words.append(w)



            # Sort by top, then x0, and join

            cell_words.sort(key=lambda w: (round(w["top"]), w["x0"]))

            text = " ".join(w["text"] for w in cell_words).strip()

            row_cells.append(text)



        rows.append(row_cells)



    return ExtractedTable(

        page=18,

        caption="TABLE 2 INDICATORS OF UNAFFORDABILITY OF HEALTHY DIETS",

        rows=rows,

        bbox=(45.0, 90.0, 395.0, 386.0),

    )





# ──────────────────────────────────────────────────────────────────────────

# TABLE 5 (SOFA 2021, page 25): Entry Points to Manage Agrifood Systems' Risk

# 4 columns: Level | Ensuring diversity | Managing connectivity | Managing risks

# 4 data rows: Contextual / National / Food supply / Households

# Each cell contains bulleted text (often multi-line).

# ──────────────────────────────────────────────────────────────────────────

def extract_2021_table5(pdf_path: Path) -> ExtractedTable:

    """

    SOFA 2021 — TABLE 5 (page 25): Entry Points to Manage Agrifood Systems' Risk and Uncertainty.



    Page dims: 453.5 × 623.6.

    4 columns: [Level label] [Ensuring diversity] [Managing connectivity] [Managing risks].

    4 data rows: Contextual / National Agrifood / Food Supply Chains / Households.



    Each cell holds bulleted text. Bullet markers come through as `}` from the

    font mapping — we strip them and join bullets with "; ".

    """

    words = _get_words(pdf_path, page_num=25)



    # Column x ranges. Labels in col 1 start at x≈51. Bullets in subsequent

    # columns sit at x={113, 212, 311}, with text content extending to x≈410.

    col_bounds = [

        (45.0, 110.0),    # Row label (CONTEXTUAL FACTORS, etc.)

        (110.0, 212.0),   # Ensuring diversity

        (212.0, 311.0),   # Managing connectivity

        (311.0, 415.0),   # Managing risks

    ]



    # Data-row y ranges (from inspection).

    data_rows_spec = [

        ("Contextual factors",         130.0, 245.0),

        ("National agrifood systems",  248.0, 355.0),

        ("Food supply chains and actors", 359.0, 430.0),

        ("Households and livelihoods (small-scale producers and vulnerable households)",

                                       434.0, 555.0),

    ]



    header = [

        "Agrifood system level",

        "Shocks difficult to foresee — Ensuring diversity",

        "Shocks difficult to foresee — Managing connectivity",

        "More predictable shocks — Managing risks",

    ]

    rows = [header]



    for label_override, y_top, y_bot in data_rows_spec:

        row_cells = []

        for col_idx, (cx0, cx1) in enumerate(col_bounds):

            # Collect all words in this cell's rectangle

            cell_words = []

            for w in words:

                wcx = (w["x0"] + w["x1"]) / 2

                wcy = (w["top"] + w["bottom"]) / 2

                if cx0 <= wcx <= cx1 and y_top <= wcy <= y_bot:

                    cell_words.append(w)



            # Sort by line first (top), then by x within line

            cell_words.sort(key=lambda w: (round(w["top"]), w["x0"]))



            # Reconstruct text, grouping by line via `top` clustering.

            # Within a line, words are space-separated.

            lines: list[list[str]] = []

            current_line: list[str] = []

            last_top: float | None = None

            for w in cell_words:

                if last_top is None or abs(w["top"] - last_top) <= 3.5:

                    current_line.append(w["text"])

                else:

                    lines.append(current_line)

                    current_line = [w["text"]]

                last_top = w["top"]

            if current_line:

                lines.append(current_line)



            # Join lines into a string. For bulleted cells (cols 2-4), the

            # bullet character renders as "}" — convert it into a "; "

            # separator between bullets.

            if col_idx == 0:

                # Label column — just join lines with space.

                text = " ".join(" ".join(line) for line in lines).strip()

                # Override with cleaner label

                text = label_override

            else:

                # Bulleted column. Split bullets on "}" markers.

                joined = " ".join(" ".join(line) for line in lines)

                # Split on the bullet marker, drop empty parts, clean each item.

                parts = [p.strip(" }").strip() for p in joined.split("}")]

                parts = [p for p in parts if p]

                text = "; ".join(parts)



            row_cells.append(text)

        rows.append(row_cells)



    return ExtractedTable(

        page=25,

        caption="TABLE 5 ENTRY POINTS TO MANAGE AGRIFOOD SYSTEMS' RISK AND UNCERTAINTY",

        rows=rows,

        bbox=(45.0, 90.0, 415.0, 555.0),

    )





# ──────────────────────────────────────────────────────────────────────────

# TABLE 3 (SOFA 2025, page 22): Land Management vs Land-use Change Interventions

# 3 columns: Policy aspect | LAND MANAGEMENT | LAND-USE CHANGE

# Rows are organized in 3 sections (REGULATORY / INCENTIVE-BASED / CROSS-COMPLIANCE)

# with 4 sub-rows each.

# ──────────────────────────────────────────────────────────────────────────

def extract_2025_table3(pdf_path: Path) -> ExtractedTable:

    """

    SOFA 2025 — TABLE 3 (page 22): Land Management vs Land-use Change Interventions.



    Page dims: 453.5 × 623.6.

    3 columns: [Policy aspect] [LAND MANAGEMENT] [LAND-USE CHANGE].

    12 data rows organized in 3 sections × 4 sub-questions each:

       REGULATORY / INCENTIVE-BASED / CROSS-COMPLIANCE (CONDITIONALITY)

    Section bands span all columns but contain no data — we skip those y-ranges

    and prefix the section name onto each row's label.

    """

    words = _get_words(pdf_path, page_num=22)



    # Column x ranges. Col 1 (label) is narrow; cols 2 & 3 are wide.

    col_bounds = [

        (45.0, 100.0),    # Row label (e.g., "Does farm size matter?")

        (100.0, 260.0),   # LAND MANAGEMENT

        (260.0, 415.0),   # LAND-USE CHANGE

    ]



    # 12 rows. Each tuple: (section, sub-label, y_top, y_bottom).

    # y ranges are tight to each row, avoiding section-band areas.

    rows_spec = [

        # REGULATORY section (band at y≈100-106)

        ("Regulatory", "Does farm size matter?",      109.0, 148.0),

        ("Regulatory", "Management burden",           149.0, 175.0),

        ("Regulatory", "Monitoring requirements",     179.0, 210.0),

        ("Regulatory", "Financing needs",             212.0, 236.0),

        # INCENTIVE-BASED section (band at y≈241-247)

        ("Incentive-based", "Does farm size matter?", 250.0, 290.0),

        ("Incentive-based", "Management burden",      294.0, 332.0),

        ("Incentive-based", "Monitoring requirements",337.0, 375.0),

        ("Incentive-based", "Financing needs",        376.0, 420.0),

        # CROSS-COMPLIANCE section (band at y≈424-430)

        ("Cross-compliance", "Does farm size matter?",433.0, 465.0),

        ("Cross-compliance", "Management burden",     476.0, 515.0),

        ("Cross-compliance", "Monitoring requirements",519.0, 550.0),

        ("Cross-compliance", "Financing needs",       551.0, 570.0),

    ]



    header = [

        "Policy instrument & question",

        "LAND MANAGEMENT",

        "LAND-USE CHANGE",

    ]

    out_rows = [header]



    for section, sub_label, y_top, y_bot in rows_spec:

        row_cells = []

        for col_idx, (cx0, cx1) in enumerate(col_bounds):

            # Collect words in this cell's rectangle

            cell_words = []

            for w in words:

                wcx = (w["x0"] + w["x1"]) / 2

                wcy = (w["top"] + w["bottom"]) / 2

                if cx0 <= wcx <= cx1 and y_top <= wcy <= y_bot:

                    cell_words.append(w)



            # Sort by line (top) then x within line

            cell_words.sort(key=lambda w: (round(w["top"]), w["x0"]))



            # Group into lines using top clustering

            lines: list[list[str]] = []

            current_line: list[str] = []

            last_top: float | None = None

            for w in cell_words:

                if last_top is None or abs(w["top"] - last_top) <= 3.5:

                    current_line.append(w["text"])

                else:

                    lines.append(current_line)

                    current_line = [w["text"]]

                last_top = w["top"]

            if current_line:

                lines.append(current_line)



            text = " ".join(" ".join(line) for line in lines).strip()

            row_cells.append(text)



        # Override col 1 with the clean section-prefixed label

        row_cells[0] = f"{section} — {sub_label}"

        out_rows.append(row_cells)



    return ExtractedTable(

        page=22,

        caption="TABLE 3 LAND MANAGEMENT VS LAND-USE CHANGE INTERVENTIONS BY TYPE OF POLICY INSTRUMENT",

        rows=out_rows,

        bbox=(45.0, 90.0, 415.0, 570.0),

    )





# ──────────────────────────────────────────────────────────────────────────

# Public API

# ──────────────────────────────────────────────────────────────────────────

def extract_tables_for_doc(doc_id: str, pdf_path: Path) -> List[ExtractedTable]:

    """Return all custom-extracted tables for a given document."""

    if doc_id == "sofa_2021":

        return [

            extract_2021_table2(pdf_path),
            extract_2021_table5(pdf_path),

        ]

    if doc_id == "sofa_2025":

        return [extract_2025_table3(pdf_path)]

    # sofa_2023 has no tables we want

    return []