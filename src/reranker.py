"""
Reranker — cross-encoder reranking of retrieved candidates.



The hybrid retriever (dense + sparse → RRF) returns up to 20 candidates

based on independent representations. A cross-encoder reads the query and

each chunk JOINTLY and produces a relevance score that captures fine-grained

query-chunk interaction the bi-encoder embedding can't.



For technical reports like SOFA with many similar-sounding sections, this

is the single biggest quality lever per the architecture screenshots.



Scoring input:

  We score on page_content (the actual chunk text), NOT embedding_text.

  The cross-encoder sees the full query-chunk pair jointly and doesn't

  benefit from the metadata prefix — it would just dilute the signal.



Model:

  cross-encoder/ms-marco-MiniLM-L-6-v2, loaded from a LOCAL path.

  No HuggingFace Hub access required at runtime.



Public API:

    Reranker().rerank(query, candidates, top_k=FINAL_K) -> List[RetrievedChunk]

"""



from __future__ import annotations



from pathlib import Path

from typing import List



from sentence_transformers import CrossEncoder



from config import FINAL_K, RERANKER_DEVICE, RERANKER_MODEL

from retriever import RetrievedChunk





# ──────────────────────────────────────────────────────────────────────────

# Model path validation

# ──────────────────────────────────────────────────────────────────────────

def _validate_local_model_path(path_str: str) -> Path:

    """

    Confirm the reranker model path points to a valid local directory

    with the files the CrossEncoder loader needs.



    We do this upfront so the user gets a clear error instead of a

    confusing failure deep inside transformers when it falls back to

    HF Hub (which is blocked on this laptop).

    """

    path = Path(path_str)

    if not path.exists():

        raise FileNotFoundError(

            f"Reranker model path does not exist: {path}\n"

            f"Check RERANKER_MODEL in config.py."

        )

    if not path.is_dir():

        raise NotADirectoryError(

            f"Reranker model path is not a directory: {path}\n"

            f"Expected a folder containing config.json, model weights, and "

            f"tokenizer files."

        )



    # Required files for sentence-transformers CrossEncoder

    required = ["config.json"]

    # Need at least one of these for the weights

    weights_options = ["pytorch_model.bin", "model.safetensors"]

    # Need at least one of these for the tokenizer

    tokenizer_options = ["tokenizer.json", "vocab.txt"]



    missing = [f for f in required if not (path / f).exists()]

    if missing:

        raise FileNotFoundError(

            f"Reranker model folder is missing required files: {missing}\n"

            f"Folder: {path}"

        )

    if not any((path / f).exists() for f in weights_options):

        raise FileNotFoundError(

            f"Reranker model folder has no weights file. Expected one of: "

            f"{weights_options}\nFolder: {path}"

        )

    if not any((path / f).exists() for f in tokenizer_options):

        raise FileNotFoundError(

            f"Reranker model folder has no tokenizer file. Expected one of: "

            f"{tokenizer_options}\nFolder: {path}"

        )



    return path





# ──────────────────────────────────────────────────────────────────────────

# Reranker

# ──────────────────────────────────────────────────────────────────────────

class Reranker:

    """

    Cross-encoder reranker. Loads the model once at construction so a

    single instance can rerank many queries cheaply.



    Typical flow:

        retriever = Retriever()

        reranker  = Reranker()

        candidates = retriever.retrieve(query, k=20)

        reranked   = reranker.rerank(query, candidates, top_k=5)

    """



    def __init__(self):

        model_path = _validate_local_model_path(RERANKER_MODEL)

        # CrossEncoder accepts a local directory path directly.

        # max_length=512 is the model's native limit. Longer chunks are

        # truncated by the tokenizer, which is fine — page_content is

        # usually well under 1000 chars (~250 tokens).

        self._model = CrossEncoder(

            str(model_path),

            device=RERANKER_DEVICE,

            max_length=512,

        )



    def rerank(

        self,

        query: str,

        candidates: List[RetrievedChunk],

        top_k: int = FINAL_K,

    ) -> List[RetrievedChunk]:

        """

        Score each (query, candidate.page_content) pair jointly and return

        the top_k candidates sorted by cross-encoder score descending.



        We mutate each RetrievedChunk by attaching a `rerank_score` attribute

        so downstream code (citation verifier, UI) can show how the chunk

        ranked. The original rrf_score / dense_rank / sparse_rank fields

        are preserved for debugging.



        Empty candidate list → empty result, no model call.

        """

        if not candidates:

            return []



        # Build (query, chunk_text) pairs. predict() handles batching

        # internally; for 20 candidates on CPU this is a single batch.

        pairs = [(query, c.page_content) for c in candidates]

        scores = self._model.predict(

            pairs,

            convert_to_numpy=True,

            show_progress_bar=False,

        )



        # Attach scores and sort. We use setattr because RetrievedChunk is

        # a @dataclass without a rerank_score field declared — adding it

        # dynamically keeps the dataclass definition clean and avoids

        # forcing rerank_score=None on every retrieval that doesn't go

        # through the reranker.

        for chunk, score in zip(candidates, scores):

            chunk.rerank_score = float(score)



        candidates.sort(key=lambda c: c.rerank_score, reverse=True)

        return candidates[:top_k]





# ──────────────────────────────────────────────────────────────────────────

# Test entry point

# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    import sys

    from retriever import Retriever



    # Same smoke test queries as retriever.py so we can see before/after

    # ranking shifts directly.

    test_queries = [

        ("Exact-term (BM25 should shine)",

         "PPFI primary production flexibility index", None),

        ("Semantic paraphrase (dense should shine)",

         "how can countries make their food systems handle disruptions better",

         None),

        ("Table-grounded query (should surface TABLE 2 high)",

         "how many people cannot afford a healthy diet by income group", None),

        ("Filtered to SOFA 2023",

         "what are hidden costs of agrifood systems",

         {"doc_id": "sofa_2023"}),

        ("Filtered to SOFA 2025",

         "land degradation policy interventions",

         {"doc_id": "sofa_2025"}),
        
         

    ]



    if len(sys.argv) > 1:

        custom_query = " ".join(sys.argv[1:])

        test_queries = [("Custom query", custom_query, None)]



    print("\n" + "=" * 70)

    print("Reranker smoke test (retrieve top 20 → rerank to top 5)")

    print("=" * 70)



    # Build once, reuse across queries

    reranker = Reranker()



    for label, query, mfilter in test_queries:

        print(f"\n{'─' * 70}")

        print(f"{label}")

        print(f"Query:  {query}")

        if mfilter:

            print(f"Filter: {mfilter}")

        print(f"{'─' * 70}")



        retriever = Retriever(metadata_filter=mfilter)

        candidates = retriever.retrieve(query, k=20)



        if not candidates:

            print("  (no candidates from retrieval)")

            continue



        # Capture pre-rerank ranks so we can show movement

        pre_rank = {c.chunk_id: i + 1 for i, c in enumerate(candidates)}



        reranked = reranker.rerank(query, candidates, top_k=5)



        for i, r in enumerate(reranked, start=1):

            old = pre_rank.get(r.chunk_id, "?")

            move = ""

            if isinstance(old, int):

                delta = old - i

                if delta > 0:

                    move = f"↑{delta}"

                elif delta < 0:

                    move = f"↓{-delta}"

                else:

                    move = "="

            d = f"d#{r.dense_rank}" if r.dense_rank else "d#-"

            s = f"s#{r.sparse_rank}" if r.sparse_rank else "s#-"

            print(f"  {i}. [{r.chunk_id}]  "

                  f"rerank={r.rerank_score:+.3f}  "

                  f"(was #{old} {move})  "

                  f"rrf={r.rrf_score:.4f}  {d}/{s}")

            print(f"     {r.doc_id} p{r.page} | {r.chunk_type} | "

                  f"{r.section[:60]}")

            preview = r.page_content[:140].replace("\n", " ")

            print(f"     \"{preview}...\"")