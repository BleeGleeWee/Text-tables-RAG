"""
Citation verifier — catches hallucinated chunk IDs in generated answers.

The grounded-generation prompt instructs Gemini to cite every claim with an
inline [chunk_id] marker matching one of the provided sources. Gemini
mostly complies, but two failure modes slip through:

  1. Pure invention: [sofa_2021_p999_text_0] (chunk doesn't exist).
  2. Off-by-one: [sofa_2021_p018_text_0] when the real source was
     [sofa_2021_p018_table_0] on the same page.

Either breaks user trust the first time it happens. This module:
  - regex-extracts every [chunk_id] marker in the answer
  - validates each against the chunks actually shown to the LLM
  - strips bad citations from the answer text
  - returns both the cleaned answer and a structured report

Definition of "valid":
  A citation is valid iff the chunk_id appears in response.sources — i.e.,
  the LLM was actually shown that chunk. Citations to real chunks in the
  docstore that weren't in sources are HALLUCINATIONS, because the LLM
  couldn't have read them and likely guessed from the ID pattern.

Public API:
    verify_citations(answer, sources) -> VerificationResult
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Set

from retriever import RetrievedChunk


# ──────────────────────────────────────────────────────────────────────────
# Citation pattern
# ──────────────────────────────────────────────────────────────────────────
# Matches our chunk_id format: {doc_id}_p{NNN}_{type}_{ordinal}
# Examples:  sofa_2021_p018_table_0   sofa_2025_p022_text_3
# We're strict about the pattern so we don't accidentally strip
# legitimate bracketed text in the answer (rare but possible).
_CITATION_RE = re.compile(
    r"\[(sofa_\d{4}_p\d{3}_(?:text|table)_\d+)\]"
)


# ──────────────────────────────────────────────────────────────────────────
# Result dataclass
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class VerificationResult:
    """
    Output of one verification pass.

    cleaned_answer    The answer text with hallucinated [chunk_id] markers
                      removed. Whitespace around removed markers is also
                      tidied so the prose still reads naturally.
    cited_ids         All chunk_ids extracted from the original answer,
                      in order of appearance, deduplicated.
    valid_ids         Subset of cited_ids that match a chunk in sources.
    hallucinated_ids  Subset of cited_ids that did NOT match a chunk in
                      sources. These were stripped from cleaned_answer.
    unused_sources    chunk_ids that were in sources but never cited.
                      Useful for the eval harness — high unused counts
                      suggest the retrieval window is too wide.
    is_clean          True iff hallucinated_ids is empty.
    n_strips          How many [chunk_id] markers were stripped in total
                      (including repeated occurrences of the same bad id).
    """
    cleaned_answer: str
    cited_ids: List[str] = field(default_factory=list)
    valid_ids: List[str] = field(default_factory=list)
    hallucinated_ids: List[str] = field(default_factory=list)
    unused_sources: List[str] = field(default_factory=list)
    is_clean: bool = True
    n_strips: int = 0


# ──────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────
def verify_citations(
    answer: str,
    sources: List[RetrievedChunk],
) -> VerificationResult:
    """
    Verify citations in the answer against the sources passed to the LLM.

    Returns a VerificationResult with cleaned_answer (hallucinated cites
    stripped) and a full report of what was found.

    Empty answer or empty sources are handled gracefully:
      - empty answer → empty result, is_clean=True
      - empty sources but cited_ids non-empty → everything hallucinated
    """
    answer = answer or ""

    # Build the valid-id set from sources actually shown to the LLM.
    valid_set: Set[str] = {c.chunk_id for c in sources}

    # Extract citations in order of appearance, dedupe while preserving order.
    raw_matches = _CITATION_RE.findall(answer)
    cited_ordered: List[str] = []
    seen: Set[str] = set()
    for cid in raw_matches:
        if cid not in seen:
            cited_ordered.append(cid)
            seen.add(cid)

    valid_ids = [c for c in cited_ordered if c in valid_set]
    hallucinated_ids = [c for c in cited_ordered if c not in valid_set]
    unused_sources = [c.chunk_id for c in sources if c.chunk_id not in seen]

    # Strip hallucinated citations from the answer text. We do this by
    # replacing each bad marker with empty string, then tidying up double
    # spaces and any orphaned punctuation around the removed slot.
    cleaned = answer
    n_strips = 0
    if hallucinated_ids:
        bad_set = set(hallucinated_ids)
        def _replace(match):
            nonlocal n_strips
            if match.group(1) in bad_set:
                n_strips += 1
                return ""
            return match.group(0)
        cleaned = _CITATION_RE.sub(_replace, cleaned)
        # Tidy whitespace and orphaned punctuation left behind.
        # E.g., "X happened . The next claim..."  →  "X happened. The next claim..."
        cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
        cleaned = re.sub(r" {2,}", " ", cleaned)
        cleaned = cleaned.strip()

    return VerificationResult(
        cleaned_answer=cleaned,
        cited_ids=cited_ordered,
        valid_ids=valid_ids,
        hallucinated_ids=hallucinated_ids,
        unused_sources=unused_sources,
        is_clean=(len(hallucinated_ids) == 0),
        n_strips=n_strips,
    )


# ──────────────────────────────────────────────────────────────────────────
# Test entry point
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # We don't need to run the real pipeline to test the verifier. We
    # build synthetic RetrievedChunk objects with just chunk_id populated.

    def _stub(cid: str) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=cid, doc_id="sofa_2021", doc_title="", doc_year=2021,
            page=0, section="", chunk_type="text",
            page_content="", table_caption="", table_json="",
            rrf_score=0.0, dense_rank=None, sparse_rank=None,
        )

    sources = [
        _stub("sofa_2021_p018_table_0"),
        _stub("sofa_2021_p018_text_2"),
        _stub("sofa_2021_p007_text_3"),
    ]

    test_cases = [
        {
            "label": "Clean answer — all citations valid",
            "answer": "Around 3 billion people cannot afford a healthy "
                      "diet [sofa_2021_p018_table_0]. The figure rises by "
                      "1 billion under an income shock "
                      "[sofa_2021_p007_text_3].",
        },
        {
            "label": "Single hallucinated citation",
            "answer": "Some claim from a real source "
                      "[sofa_2021_p018_table_0] and another claim from a "
                      "fake source [sofa_2021_p999_text_0].",
        },
        {
            "label": "Off-by-one hallucination (real ID, wrong chunk)",
            "answer": "Hidden costs are large [sofa_2021_p018_text_0] "
                      "and rising [sofa_2021_p018_text_2].",
            # p018_text_0 is plausible-looking but NOT in sources.
        },
        {
            "label": "Repeated hallucinated id (should strip both)",
            "answer": "Fact one [sofa_2021_p888_text_5]. Fact two "
                      "[sofa_2021_p888_text_5]. Fact three "
                      "[sofa_2021_p018_table_0].",
        },
        {
            "label": "Refusal — no citations at all",
            "answer": "I don't have information about this in the documents.",
        },
        {
            "label": "Unused source (cited only one of three)",
            "answer": "Just one cite [sofa_2021_p018_table_0].",
        },
    ]

    print("\n" + "=" * 70)
    print("Citation verifier smoke test")
    print("=" * 70)

    for case in test_cases:
        print(f"\n{'─' * 70}")
        print(f"{case['label']}")
        print(f"{'─' * 70}")
        print(f"Original:\n  {case['answer']}")
        result = verify_citations(case["answer"], sources)
        print(f"\nCleaned:\n  {result.cleaned_answer}")
        print(f"\nis_clean:         {result.is_clean}")
        print(f"cited_ids:        {result.cited_ids}")
        print(f"valid_ids:        {result.valid_ids}")
        print(f"hallucinated_ids: {result.hallucinated_ids}")
        print(f"unused_sources:   {result.unused_sources}")
        print(f"n_strips:         {result.n_strips}")