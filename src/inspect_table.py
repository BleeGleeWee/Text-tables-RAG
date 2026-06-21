"""
Inspect raw word coordinates on a specific page.
Used to discover column/row boundaries for writing custom table extractors.

Usage:
    python src/inspect_table.py cb7351en.pdf 18
"""

import sys
from pathlib import Path
import pdfplumber
from config import DATA_DIR


def inspect_page(pdf_path: Path, page_num: int):
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num - 1]
        print(f"Page dimensions: width={page.width:.1f}, height={page.height:.1f}\n")

        words = page.extract_words(
            x_tolerance=2,
            y_tolerance=2,
            keep_blank_chars=False,
        )
        print(f"Total words on page: {len(words)}\n")
        print(f"{'TEXT':<50} {'x0':>7} {'x1':>7} {'top':>7} {'bottom':>7}")
        print("-" * 85)
        for w in words:
            text = w["text"][:48]
            print(f"{text:<50} {w['x0']:>7.1f} {w['x1']:>7.1f} {w['top']:>7.1f} {w['bottom']:>7.1f}")


if __name__ == "__main__":
    pdf_name = sys.argv[1]
    page_num = int(sys.argv[2])
    inspect_page(DATA_DIR / pdf_name, page_num)