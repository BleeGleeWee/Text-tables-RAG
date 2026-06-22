"""

Query rewriter — turns raw user input into something the retriever can use.



Three jobs in one Gemini call:

  1. History-aware rewriting:

       "what about for kilocalories?"  →  "what is the DSFI for kilocalories?"

       Resolves coreferences ("that", "it", "for X", "what about Y") using

       prior chat turns.



  2. Decomposition:

       "compare hidden costs in low-income vs high-income countries"

        → ["hidden costs in low-income countries",

           "hidden costs in high-income countries"]

       Splits multi-part queries so we run retrieval per sub-query and

       merge results before reranking. Improves recall dramatically on

       comparative questions.



  3. Metadata extraction:

       "what did the 2023 report say about X"  →  doc_id=sofa_2023

       "in SOFA 2021..."                       →  doc_id=sofa_2021

       Resolves year/doc references to a Chroma metadata filter so we

       only retrieve from the relevant document(s).



Fast path:

  Short standalone queries (no history, <10 words) skip Gemini entirely

  and return a passthrough RewriteResult. Saves a round-trip and the

  free-tier quota for the cases that actually need rewriting.



Public API:

    QueryRewriter().rewrite(query, chat_history=None) -> RewriteResult

"""



from __future__ import annotations



import json

import re

from dataclasses import dataclass, field

from typing import List, Optional



from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.messages import HumanMessage, SystemMessage



from config import (

    DOCUMENTS,

    GEMINI_MAX_OUTPUT_TOKENS,

    GEMINI_MODEL,

    GEMINI_TEMPERATURE,

    GOOGLE_API_KEY,

)





# ──────────────────────────────────────────────────────────────────────────

# Data class

# ──────────────────────────────────────────────────────────────────────────

@dataclass

class RewriteResult:

    """

    Output of the rewriter.



    rewritten_query   The standalone, coreference-resolved version of the

                      user's query. Always populated.

    sub_queries       If the query decomposes into multiple information

                      needs, the list of sub-queries to retrieve in

                      parallel. If the query is atomic, this list contains

                      just rewritten_query (so callers can always iterate).

    metadata_filter   Chroma-compatible filter dict, or None. Currently

                      supports {"doc_id": "sofa_YYYY"} only — that's what

                      retriever.py understands.

    used_llm          True if Gemini was called; False if the fast path

                      was taken. Useful for debugging and cost tracking.

    """

    rewritten_query: str

    sub_queries: List[str]

    metadata_filter: Optional[dict] = None

    used_llm: bool = False





# ──────────────────────────────────────────────────────────────────────────

# Doc registry — resolve year → doc_id

# ──────────────────────────────────────────────────────────────────────────

# Build a year → doc_id lookup once at import time from config.DOCUMENTS,

# so the rewriter doesn't depend on hardcoded strings.

_YEAR_TO_DOC_ID = {

    str(meta["year"]): meta["doc_id"]

    for meta in DOCUMENTS.values()

}

# Also support doc_id references in the query like "sofa_2021"

_DOC_IDS = {meta["doc_id"] for meta in DOCUMENTS.values()}





def _detect_doc_filter_regex(query: str) -> Optional[dict]:

    """

    Lightweight regex-based doc filter detection used by the fast path.

    Catches the obvious cases (year mentioned, doc_id mentioned). The

    LLM path catches subtler cases ("in the latest report", etc.).

    """

    # Direct doc_id mention

    for did in _DOC_IDS:

        if did in query.lower():

            return {"doc_id": did}

    # Year mention — only if the year is one of our docs

    for year, did in _YEAR_TO_DOC_ID.items():

        # Word-boundary match so "2021" doesn't match "20210" or "1.2021"

        if re.search(rf"\b{year}\b", query):

            return {"doc_id": did}

    return None





# ──────────────────────────────────────────────────────────────────────────

# Fast-path heuristics

# ──────────────────────────────────────────────────────────────────────────

def _is_simple_query(query: str, chat_history: Optional[list]) -> bool:

    """

    Decide whether the query needs Gemini at all.



    Skip the LLM only when ALL of:

      - no chat history (nothing to resolve coreferences against)

      - query is short (<10 words)

      - no comparative trigger words (signal decomposition)

      - no semantic doc-reference triggers ("latest", "earliest", etc.) —

        these need the LLM because the regex filter can't resolve them.

    """

    if chat_history:

        return False

    if len(query.split()) >= 10:

        return False

    q_lower = query.lower()

    comparative_triggers = (

        "compare", "comparison", " vs ", " vs.", "versus", "between",

        "difference", "differ"

    )

    if any(t in q_lower for t in comparative_triggers):

        return False

    semantic_doc_triggers = (

        "latest", "earliest", "most recent", "newest",

        "previous report", "earlier report",

    )

    if any(t in q_lower for t in semantic_doc_triggers):

        return False

    return True





# ──────────────────────────────────────────────────────────────────────────

# Prompt

# ──────────────────────────────────────────────────────────────────────────

_KNOWN_DOCS = "\n".join(

    f"  - doc_id={meta['doc_id']}, year={meta['year']}, "

    f"title=\"{meta['title']}\""

    for meta in DOCUMENTS.values()

)



_SYSTEM_PROMPT = f"""You are a query rewriter for a RAG system over FAO State of Food and Agriculture reports.



The knowledge base contains these documents:

{_KNOWN_DOCS}



Your job: given a user query and (optionally) recent chat history, produce a JSON object with three fields:



1. "rewritten_query": A standalone version of the user's query. Resolve coreferences ("it", "that", "for X", "what about Y") using the chat history. If the query is already standalone, return it unchanged. Always a string.



2. "sub_queries": If the query asks about multiple distinct information needs (e.g., "compare A vs B", "X and Y and Z"), split it into independent sub-queries. Each sub-query must be a complete, standalone information need. If the query is atomic, return a single-element list containing rewritten_query.



3. "metadata_filter": If the user references a specific year or document (e.g., "the 2023 report", "in SOFA 2021", "what did the latest report say"), return {{"doc_id": "<matching doc_id>"}} using the registry above. If no specific document is implied, return null. Only the doc_ids listed above are valid.



Rules:

- Output ONLY the JSON object. No prose, no markdown fences, no explanation.

- Keep rewritten_query and sub_queries faithful to the original intent — do not invent new questions.

- "the latest report" → the most recent year in the registry.

- "the earliest report" → the earliest year.



Examples:



User: "What is the PPFI?"

Output: {{"rewritten_query": "What is the PPFI?", "sub_queries": ["What is the PPFI?"], "metadata_filter": null}}



User: "Compare hidden costs in low-income vs high-income countries"

Output: {{"rewritten_query": "Compare hidden costs in low-income vs high-income countries", "sub_queries": ["hidden costs of agrifood systems in low-income countries", "hidden costs of agrifood systems in high-income countries"], "metadata_filter": null}}



User: "What did the 2023 report say about hidden costs?"

Output: {{"rewritten_query": "What did the 2023 SOFA report say about hidden costs of agrifood systems?", "sub_queries": ["hidden costs of agrifood systems in the 2023 SOFA report"], "metadata_filter": {{"doc_id": "sofa_2023"}}}}



Chat history (User: "What is the DSFI for protein?" / Assistant: "<answer about DSFI for protein>")

User: "what about for kilocalories?"

Output: {{"rewritten_query": "What is the DSFI for kilocalories?", "sub_queries": ["What is the DSFI for kilocalories?"], "metadata_filter": null}}

"""





# ──────────────────────────────────────────────────────────────────────────

# Rewriter

# ──────────────────────────────────────────────────────────────────────────

class QueryRewriter:

    """

    Builds a Gemini client at construction and reuses it across calls.

    Stateless otherwise — chat history is passed in per call.

    """



    def __init__(self):

        if not GOOGLE_API_KEY:

            raise RuntimeError(

                "GOOGLE_API_KEY not set in .env — query rewriter cannot start."

            )

        self._llm = ChatGoogleGenerativeAI(

            model=GEMINI_MODEL,

            temperature=GEMINI_TEMPERATURE,

            max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,

            google_api_key=GOOGLE_API_KEY,

        )



    # ──────────────────────────────────────────────────────────────────

    # Public entry point

    # ──────────────────────────────────────────────────────────────────

    def rewrite(

        self,

        query: str,

        chat_history: Optional[List[dict]] = None,

    ) -> RewriteResult:

        """

        Run the rewriter on a single user query.



        chat_history is a list of {"role": "user"|"assistant", "content": "..."}

        dicts representing prior turns, oldest first. Pass None or [] for the

        first turn of a conversation.

        """

        query = query.strip()

        if not query:

            return RewriteResult(

                rewritten_query="",

                sub_queries=[""],

                metadata_filter=None,

                used_llm=False,

            )



        # ── Fast path ──

        if _is_simple_query(query, chat_history):

            print("FAST PATH")

            return RewriteResult(

                rewritten_query=query,

                sub_queries=[query],

                metadata_filter=_detect_doc_filter_regex(query),

                used_llm=False,

            )



        # ── LLM path ──

        print("LLM PATH")

        return self._llm_rewrite(query, chat_history or [])



    # ──────────────────────────────────────────────────────────────────

    # LLM path

    # ──────────────────────────────────────────────────────────────────

    def _llm_rewrite(

        self,

        query: str,

        chat_history: List[dict],

    ) -> RewriteResult:

        """Build the prompt, call Gemini, parse JSON, validate, fall back."""

        # Format chat history compactly into the prompt rather than using

        # the LLM's native history support — the rewriter is a one-shot

        # transformation, not a conversation, and structured JSON output

        # is easier with a single HumanMessage.

        history_block = self._format_history(chat_history)

        user_block = (

            f"{history_block}User: \"{query}\"\n"

            f"Output:"

        )



        try:

            response = self._llm.invoke([

                SystemMessage(content=_SYSTEM_PROMPT),

                HumanMessage(content=user_block),

            ])

            raw = response.content.strip()

        except Exception as e:

            # Network error, quota error, etc. Fall back to the fast path

            # so the user still gets an answer — degrading is better than

            # 500-ing.

            print(f"[rewriter] Gemini call failed ({type(e).__name__}: {e}). "

                  f"Falling back to passthrough.")

            return RewriteResult(

                rewritten_query=query,

                sub_queries=[query],

                metadata_filter=_detect_doc_filter_regex(query),

                used_llm=False,

            )



        return self._parse_response(raw, query)



    # ──────────────────────────────────────────────────────────────────

    # Helpers

    # ──────────────────────────────────────────────────────────────────

    @staticmethod

    def _format_history(chat_history: List[dict]) -> str:

        """Format chat history into a short text block for the prompt."""

        if not chat_history:

            return ""

        lines = ["Chat history (most recent last):"]

        for turn in chat_history:

            role = turn.get("role", "user").capitalize()

            content = turn.get("content", "").strip()

            # Truncate long assistant answers — only the gist matters for

            # coreference resolution.

            if len(content) > 300:

                content = content[:300] + "..."

            lines.append(f"  {role}: \"{content}\"")

        lines.append("")  # blank line before the new user turn

        return "\n".join(lines) + "\n"



    def _parse_response(self, raw: str, original_query: str) -> RewriteResult:

        """

        Parse Gemini's JSON response. Tolerant of code fences (```json ... ```)

        because models sometimes add them despite instructions. Falls back

        to the original query if parsing fails.

        """

        # Strip code fences if present

        cleaned = raw

        if cleaned.startswith("```"):

            # Remove leading fence (```json or just ```)

            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)

            cleaned = re.sub(r"\s*```$", "", cleaned)

        cleaned = cleaned.strip()



        try:

            data = json.loads(cleaned)

        except json.JSONDecodeError:

            print(f"[rewriter] Could not parse JSON from Gemini, falling "

                  f"back. Raw response was:\n{raw[:300]}")

            return RewriteResult(

                rewritten_query=original_query,

                sub_queries=[original_query],

                metadata_filter=_detect_doc_filter_regex(original_query),

                used_llm=True,

            )



        # Validate and normalize each field

        rewritten = str(data.get("rewritten_query", original_query)).strip()

        if not rewritten:

            rewritten = original_query



        sub_queries = data.get("sub_queries") or [rewritten]

        if not isinstance(sub_queries, list):

            sub_queries = [rewritten]

        sub_queries = [str(s).strip() for s in sub_queries if str(s).strip()]

        if not sub_queries:

            sub_queries = [rewritten]



        # Validate metadata_filter — only allow {"doc_id": <known doc_id>}

        mfilter = data.get("metadata_filter")

        if isinstance(mfilter, dict) and "doc_id" in mfilter:

            if mfilter["doc_id"] in _DOC_IDS:

                metadata_filter = {"doc_id": mfilter["doc_id"]}

            else:

                metadata_filter = None

        else:

            metadata_filter = None



        return RewriteResult(

            rewritten_query=rewritten,

            sub_queries=sub_queries,

            metadata_filter=metadata_filter,

            used_llm=True,

        )





# ──────────────────────────────────────────────────────────────────────────

# Test entry point

# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Test cases cover:

    #   - simple standalone query (fast path)

    #   - simple query with year (fast path, with regex filter)

    #   - coreference query needing history (LLM path)

    #   - comparative query needing decomposition (LLM path)

    #   - long query that's not comparative (LLM path, no decomposition)

    #   - "latest report" reference (LLM path, filter via doc registry)



    rewriter = QueryRewriter()



    test_cases = [

        {

            "label": "Simple standalone — should take fast path",

            "query": "What is the PPFI?",

            "history": None,

        },

        {

            "label": "Simple query with year — fast path + regex filter",

            "query": "What did the 2023 report say?",

            "history": None,

        },

        {

            "label": "Coreference resolution — needs LLM + history",

            "query": "what about for kilocalories?",

            "history": [

                {"role": "user", "content": "What is the DSFI for protein?"},

                {"role": "assistant",

                 "content": "The DSFI for protein measures the diversity "

                            "of dietary protein sourcing pathways..."},

            ],

        },

        {

            "label": "Decomposition — comparative query",

            "query": "Compare hidden costs in low-income vs high-income countries",

            "history": None,

        },

        {

            "label": "Long query, not comparative — LLM but no decomposition",

            "query": "How do diverse food supply chains and trade networks "

                     "help countries handle disruptions to food systems?",

            "history": None,

        },

        {

            "label": "Latest-report reference — LLM resolves to doc_id",

            "query": "What does the latest report say about land degradation?",

            "history": None,

        },

    ]



    print("\n" + "=" * 70)

    print("Query rewriter smoke test")

    print("=" * 70)



    for case in test_cases:

        print(f"\n{'─' * 70}")

        print(f"{case['label']}")

        print(f"Query:   {case['query']}")

        if case["history"]:

            print(f"History: {len(case['history'])} turn(s)")

        print(f"{'─' * 70}")



        result = rewriter.rewrite(case["query"], case["history"])



        path = "LLM" if result.used_llm else "fast"

        print(f"  Path:            {path}")

        print(f"  Rewritten:       {result.rewritten_query}")

        print(f"  Sub-queries ({len(result.sub_queries)}):")

        for sq in result.sub_queries:

            print(f"    - {sq}")

        print(f"  Metadata filter: {result.metadata_filter}")