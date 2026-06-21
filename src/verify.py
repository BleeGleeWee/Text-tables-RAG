"""

Verification dump utility.



Runs the parser on all configured PDFs and writes the extracted content

to disk in human-readable formats:



  verify_output/

    summary.html               <- index page (open this first)

    sofa_2021/

      text.md                  <- all extracted text, page by page

      tables.html              <- all extracted tables rendered

    sofa_2023/...

    sofa_2025/...



Usage:

    python src/verify.py

"""



from __future__ import annotations



import html

from pathlib import Path



from config import DATA_DIR, DOCUMENTS, PROJECT_ROOT

from parser import parse_pdf, ExtractedTable, PageContent



VERIFY_DIR = PROJECT_ROOT / "verify_output"





# ──────────────────────────────────────────────────────────────────────────

# HTML rendering

# ──────────────────────────────────────────────────────────────────────────

HTML_STYLE = """

<style>

  body {

    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;

    max-width: 1100px;

    margin: 40px auto;

    padding: 0 20px;

    color: #222;

    line-height: 1.5;

  }

  h1 { color: #b71c1c; border-bottom: 3px solid #b71c1c; padding-bottom: 8px; }

  h2 { color: #1565c0; margin-top: 40px; }

  h3 { color: #555; margin-top: 25px; }

  .meta { color: #777; font-size: 0.9em; margin-bottom: 12px; }

  .caption { font-weight: bold; color: #b71c1c; margin-bottom: 6px; font-size: 1.05em; }

  table {

    border-collapse: collapse;

    margin: 8px 0 28px 0;

    width: 100%;

    font-size: 0.9em;

    box-shadow: 0 1px 3px rgba(0,0,0,0.08);

  }

  th, td {

    border: 1px solid #ccc;

    padding: 8px 10px;

    text-align: left;

    vertical-align: top;

  }

  th { background: #f0f4f8; font-weight: 600; }

  tr:nth-child(even) td { background: #fafbfc; }

  .nav {

    background: #f6f6f6;

    padding: 16px;

    border-left: 4px solid #1565c0;

    margin-bottom: 30px;

  }

  .nav a { display: inline-block; margin-right: 16px; }

  .stats {

    background: #fff8e1;

    padding: 12px 16px;

    border-radius: 4px;

    margin: 16px 0;

  }

  .card {

    border: 1px solid #ddd;

    border-radius: 6px;

    padding: 20px;

    margin: 16px 0;

    background: #fff;

  }

</style>

"""





def _render_table_html(table: ExtractedTable) -> str:

    """Render one extracted table as styled HTML."""

    rows = table.rows

    if not rows:

        return ""



    header = rows[0]

    body = rows[1:] if len(rows) > 1 else []



    parts = [f'<div class="caption">{html.escape(table.caption)}</div>']

    parts.append(f'<div class="meta">Page {table.page} &middot; {len(rows)} rows &middot; '

                 f'{max(len(r) for r in rows)} columns</div>')

    parts.append("<table>")



    # Header

    parts.append("<thead><tr>")

    for cell in header:

        parts.append(f"<th>{html.escape(str(cell or '').strip())}</th>")

    parts.append("</tr></thead>")



    # Body

    parts.append("<tbody>")

    for row in body:

        padded = list(row) + [""] * (len(header) - len(row))

        parts.append("<tr>")

        for cell in padded[: len(header)]:

            cell_str = str(cell or "").strip().replace("\n", "<br>")

            parts.append(f"<td>{cell_str}</td>")

        parts.append("</tr>")

    parts.append("</tbody>")

    parts.append("</table>")



    return "\n".join(parts)





def _write_tables_html(doc_id: str, doc_title: str, pages: list[PageContent], out_path: Path):

    """Write all tables for one document to HTML."""

    all_tables = [(p.page, t) for p in pages for t in p.tables]



    content = [

        "<!DOCTYPE html><html><head>",

        f"<title>{html.escape(doc_title)} — Extracted Tables</title>",

        HTML_STYLE,

        "</head><body>",

        f"<h1>{html.escape(doc_title)}</h1>",

        '<div class="nav">',

        '<a href="../summary.html">← Back to summary</a>',

        f'<a href="text.md">View extracted text →</a>',

        "</div>",

        f'<div class="stats"><b>Total tables extracted:</b> {len(all_tables)}</div>',

    ]



    if not all_tables:

        content.append("<p><em>No tables were extracted from this document.</em></p>")

    else:

        for page_num, table in all_tables:

            content.append('<div class="card">')

            content.append(_render_table_html(table))

            content.append("</div>")



    content.append("</body></html>")

    out_path.write_text("\n".join(content), encoding="utf-8")





def _write_text_md(doc_id: str, doc_title: str, pages: list[PageContent], out_path: Path):

    """Write all extracted text for one document to a Markdown file."""

    lines = [f"# {doc_title}", "", f"Document ID: `{doc_id}`", ""]

    lines.append(f"**Total pages:** {len(pages)}")

    total_chars = sum(len(p.text) for p in pages)

    total_tables = sum(len(p.tables) for p in pages)

    lines.append(f"**Total text characters:** {total_chars:,}")

    lines.append(f"**Total tables:** {total_tables}")

    lines.append("\n---\n")



    for p in pages:

        lines.append(f"## Page {p.page}")

        lines.append(f"**Section:** {p.section}")

        lines.append("")

        if p.tables:

            lines.append(f"**Tables on this page ({len(p.tables)}):**")

            for t in p.tables:

                lines.append(f"  - {t.caption} ({len(t.rows)} rows)")

            lines.append("")

        lines.append("**Text:**")

        lines.append("```")

        lines.append(p.text if p.text else "(no text extracted)")

        lines.append("```")

        lines.append("")

        lines.append("---")

        lines.append("")



    out_path.write_text("\n".join(lines), encoding="utf-8")





def _write_summary_html(doc_stats: list[dict], out_path: Path):

    """Write the top-level index page."""

    content = [

        "<!DOCTYPE html><html><head>",

        "<title>FAO RAG — Parser Verification</title>",

        HTML_STYLE,

        "</head><body>",

        "<h1>FAO RAG — Parser Verification Output</h1>",

        "<p>This page summarizes everything extracted by the PDF parser. "

        "Use the links below to inspect each document's text and tables.</p>",

    ]



    for d in doc_stats:

        content.append('<div class="card">')

        content.append(f'<h2>{html.escape(d["title"])}</h2>')

        content.append(f'<div class="meta">Document ID: <code>{d["doc_id"]}</code> &middot; '

                       f'Source: <code>{d["filename"]}</code></div>')

        content.append('<div class="stats">')

        content.append(f'<b>Pages:</b> {d["pages"]} &nbsp;|&nbsp; '

                       f'<b>Text characters:</b> {d["chars"]:,} &nbsp;|&nbsp; '

                       f'<b>Tables extracted:</b> {d["tables"]}')

        content.append('</div>')

        content.append('<p>')

        content.append(f'<a href="{d["doc_id"]}/tables.html"><b>View extracted tables →</b></a>')

        content.append('&nbsp;&nbsp;&nbsp;')

        content.append(f'<a href="{d["doc_id"]}/text.md"><b>View extracted text →</b></a>')

        content.append('</p>')



        # List of table captions

        if d["table_captions"]:

            content.append('<h3>Tables detected:</h3><ul>')

            for page, cap in d["table_captions"]:

                content.append(f'<li>Page {page} — {html.escape(cap)}</li>')

            content.append('</ul>')

        else:

            content.append('<p><em>No tables detected.</em></p>')



        content.append('</div>')



    content.append("</body></html>")

    out_path.write_text("\n".join(content), encoding="utf-8")





# ──────────────────────────────────────────────────────────────────────────

# Main

# ──────────────────────────────────────────────────────────────────────────

def main():

    VERIFY_DIR.mkdir(parents=True, exist_ok=True)

    doc_stats = []



    for filename, meta in DOCUMENTS.items():

        pdf_path = DATA_DIR / filename

        if not pdf_path.exists():

            print(f"⚠️  Skipping {filename} — not found in {DATA_DIR}")

            continue



        print(f"Parsing {filename}...")

        pages = parse_pdf(pdf_path,doc_id=meta["doc_id"])



        doc_dir = VERIFY_DIR / meta["doc_id"]

        doc_dir.mkdir(exist_ok=True)



        # Write text dump

        _write_text_md(meta["doc_id"], meta["title"], pages, doc_dir / "text.md")

        # Write tables HTML

        _write_tables_html(meta["doc_id"], meta["title"], pages, doc_dir / "tables.html")



        chars = sum(len(p.text) for p in pages)

        tables = sum(len(p.tables) for p in pages)

        captions = [(p.page, t.caption) for p in pages for t in p.tables]



        doc_stats.append({

            "doc_id": meta["doc_id"],

            "title": meta["title"],

            "filename": filename,

            "pages": len(pages),

            "chars": chars,

            "tables": tables,

            "table_captions": captions,

        })

        print(f"  ✓ {len(pages)} pages, {chars:,} chars, {tables} tables")



    _write_summary_html(doc_stats, VERIFY_DIR / "summary.html")

    print(f"\n✅ Verification output written to: {VERIFY_DIR}")

    print(f"   Open this in your browser: {VERIFY_DIR / 'summary.html'}")





if __name__ == "__main__":

    main()