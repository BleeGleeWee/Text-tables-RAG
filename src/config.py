"""
Central configuration for the FAO Multimodal RAG system.
All paths, model names, and tuning parameters live here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "pdfs"
STORAGE_DIR = PROJECT_ROOT / "storage"
CHROMA_DIR = STORAGE_DIR / "chroma"
DOCSTORE_PATH = STORAGE_DIR / "docstore.sqlite"
BM25_PATH = STORAGE_DIR / "bm25.pkl"
CACHE_PATH = STORAGE_DIR / "cache.sqlite"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories
for p in [DATA_DIR, STORAGE_DIR, CHROMA_DIR, LOGS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────
# Document registry — maps PDF filename to clean doc_id and title
# ──────────────────────────────────────────────────────────────────────────
DOCUMENTS = {
    "cb7351en.pdf": {
        "doc_id": "sofa_2021",
        "title": "The State of Food and Agriculture 2021",
        "subtitle": "Making agrifood systems more resilient to shocks and stresses",
        "year": 2021,
    },
    "cc7937en-1.pdf": {
        "doc_id": "sofa_2023",
        "title": "The State of Food and Agriculture 2023",
        "subtitle": "Revealing the true cost of food to transform agrifood systems",
        "year": 2023,
    },
    "cd7071en.pdf": {
        "doc_id": "sofa_2025",
        "title": "The State of Food and Agriculture 2025",
        "subtitle": "Addressing land degradation across landholding scales",
        "year": 2025,
    },
}

# ──────────────────────────────────────────────────────────────────────────

# Pages to skip during text extraction

# These pages contain only figures/charts/graphics whose text content

# (legends, axis labels, country labels) is not useful as prose context

# and only adds noise to the RAG corpus.

# Tables on these pages, if any, are still extracted via table_extractors.

# ──────────────────────────────────────────────────────────────────────────

SKIP_TEXT_PAGES = {

    "sofa_2021": {14, 15, 17, 19},

    "sofa_2023": {11, 18, 19, 21, 24},

    "sofa_2025": {13},

}

SECTION_SKIP_PAGES = {

    "sofa_2021": {1, 2, 3},   # cover, copyright, contents

    "sofa_2023": {1, 2, 3},

    "sofa_2025": {1, 2, 3},

}

TOP_LEVEL_SECTIONS = {
    "CORE MESSAGES",
    "FOREWORD",
    "SUMMARY",
    "CONTENTS",
}



# ──────────────────────────────────────────────────────────────────────────
# Parsing parameters
# ──────────────────────────────────────────────────────────────────────────
# Font size threshold (in points) for treating text as a heading.
# FAO "In Brief" PDFs use ~11pt body, 14-22pt headings.
HEADING_MIN_FONT_SIZE = 13.0

# Table filtering: drop tables that don't look like real data tables
TABLE_MIN_ROWS = 2
TABLE_MIN_COLS = 2
TABLE_MIN_CELL_LENGTH = 2  # at least one cell must have >= this many chars

# Page header/footer patterns to strip
NOISE_PATTERNS = [
    r"^\s*\|\s*\d+\s*\|\s*$",            # | 14 |
    r"^\s*SUMMARY\s*$",
    r"^\s*FOREWORD\s*$",
    r"^\s*CONTENTS\s*$",
    r"^\s*CORE MESSAGES\s*$",
    r"THE STATE OF FOOD AND AGRICULTURE \d{4}\s+IN BRIEF",
]

# ──────────────────────────────────────────────────────────────────────────
# Chunking parameters
# ──────────────────────────────────────────────────────────────────────────
TEXT_CHUNK_SIZE = 800
TEXT_CHUNK_OVERLAP = 150

# ──────────────────────────────────────────────────────────────────────────
# Embedding & retrieval
# ──────────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = r"C:\Users\Asmit.Verma\Desktop\multi-modal-rag-table-and-text\all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"  # change to "cuda" if GPU available

CHROMA_COLLECTION = "fao_sofa_chunks"

# Retrieval knobs
DENSE_K = 15        # candidates from dense retrieval
SPARSE_K = 15       # candidates from BM25
RRF_K = 60          # RRF constant
FINAL_K = 5         # chunks passed to LLM after reranking

# ──────────────────────────────────────────────────────────────────────────
# Reranker
# ──────────────────────────────────────────────────────────────────────────
RERANKER_MODEL = r"C:\Users\Asmit.Verma\.cache\kagglehub\models\johnsonhk88\cross-encoderms-marco-minilm-l-6-v2\transformers\v1\1\ms-marco-MiniLM-L-6-v2"
RERANKER_DEVICE = "cpu"
REFUSAL_RERANKER_FLOOR=0.0

# ──────────────────────────────────────────────────────────────────────────
# LLM (Gemini)
# ──────────────────────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_TEMPERATURE = 0.1
GEMINI_MAX_OUTPUT_TOKENS = 2048

# ──────────────────────────────────────────────────────────────────────────
# Memory
# ──────────────────────────────────────────────────────────────────────────
CHAT_HISTORY_TURNS = 4  # how many prior user/assistant turns to keep

# ──────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
QUERY_LOG_FILE = LOGS_DIR / "queries.jsonl"