"""
Search tool for SlackChat QA Environment
"""

from os.path import join
from typing import Any
import logging

import numpy as np
import bm25s
from bm25s import BM25
from Stemmer import Stemmer
# from transformers import PreTrainedTokenizerBase

from ...base import BaseTool


logger = logging.getLogger(__name__)


RESULT_TEMPLATE = """## Document View:
'''
{document}
'''{scroll_message}
"""


def build_retriever(
    retriever_name: str,
    corpus: list[dict[str, Any]],
    stemmer: Stemmer | None = None,
    retriever_config: dict[str, Any] | None = None,
    stopwords: str = "en",
    save_path: str | None = None,
) -> BM25 | None:
    """
    Build BM25 retriever from corpus and stemmer
    """
    # Always load new retriever for LongBench (bc we index over individual sample docs)
    try:
        return BM25.load(save_path or "does_not_exist", load_corpus=True)
    except FileNotFoundError:
        print(f"-> No retriever found at {save_path}, building new retriever")
    
    if retriever_name.startswith("bm25"):
        # Lowercase corpus for BM25
        corpus = [c["text"].lower() for c in corpus]
        corpus_tokens = bm25s.tokenize(corpus, stopwords=stopwords, stemmer=stemmer)
        # Create the BM25 model and index the corpus
        retriever = (
            bm25s.BM25(**retriever_config)
            if retriever_config is not None
            else bm25s.BM25()
        )
    else:
        raise NotImplementedError(f"Sorry, retriever {retriever_name} not implemented")

    # Index and save corpus
    retriever.index(corpus_tokens)
    if save_path is not None:
        retriever.save(save_path, corpus=corpus)
    return retriever


class SearchTool(BaseTool):
    """
    Search tool
    """

    def __init__(
        self,
        corpus: list[dict[str, Any]],
        # tokenizer: PreTrainedTokenizerBase,
        retriever_name: str = "bm25_index",
        use_stemmer: bool = True,
        top_k: int = 1,
        save_path: str | None = None,
        retriever_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.retriever_name = retriever_name
        self.stemmer = Stemmer("english") if use_stemmer else None  
        self.retriever_config = retriever_config
        
        # self.tokenizer = tokenizer
        self.corpus = corpus
        self.top_k = 1
        if top_k != 1:
            logger.warning(
                f"top_k is {top_k} != 1, but for LongBench-v2 we only return the top result."
            )

        if save_path is not None:
            self.bm25_save_path = join(save_path, f"{retriever_name}")
        self.retriever = build_retriever(
            retriever_name=retriever_name,
            corpus=corpus,
            stemmer=self.stemmer,
            retriever_config=self.retriever_config,
            stopwords="en",
            save_path=self.bm25_save_path,
        )

    def __call__(self, query: str, **kwargs: Any) -> tuple[dict[str, Any] | None, str]:
        """
        Search the corpus for top document based on the given query

        Returns:
        - top-k document dicts or string
        """
        try:
            # Query the corpus
            bm25_query_tokens = bm25s.tokenize(query.lower(), self.stemmer)
            # results, scores = self.retriever.retrieve(bm25_query_tokens, k=self.top_k)
            results, _ = self.retriever.retrieve(bm25_query_tokens, k=self.top_k)
            # tokenizer_fn = cast(Callable[[str], dict[str, Any]], self.tokenizer)
            # llm_query_input_ids = tokenizer_fn(query)["input_ids"]
            topk_results: list[dict[str, Any]] = []
            topk_results_str: list[str] = []
            for i in range(results.shape[1]):
                # doc_idx, score = results[0, i], scores[0, i]
                doc_id = results[0, i]["id"]
                doc_dict = self.corpus[doc_id]
                # Return string representation of the result
                scroll_msg = ""
                if doc_dict["next_chunk_idx"] is not None:
                    scroll_msg += "\n- Scroll down for more..."
                if doc_dict["prev_chunk_idx"] is not None:
                    scroll_msg += "\n- Scroll up for more..."
                topk_results_str.append(
                    RESULT_TEMPLATE.format(document=doc_dict["text"], scroll_message=scroll_msg)
                )
                topk_results.append(doc_dict)
                break
        
            new_doc_dict = topk_results[0]
            result_str = topk_results_str[0]
            result_str = f"# Search Result:\n\n{result_str}"

        except Exception as e:
            _error_class = type(e).__name__
            new_doc_dict = None
            result_str = f"Error in SearchTool ({_error_class}: {e})"
            logger.error(result_str)
            breakpoint()

        return new_doc_dict, result_str

    def get_tool_desc(self) -> dict:
        """
        Get the description of the search tool
        """
        return {
            "type": "function",
            "name": "search",
            "description": "Search the corpus for relevant information based on the given query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to search for.",
                    },
                },
                "required": ["query"],
            },
        }


def best_window_total_hits_np(
    doc: list[int],
    query: list[int],
    window: int = 256,
) -> tuple[int, int, list[int]]:
    """
    Returns (start_idx, score, best_window_slice) where score counts how many
    ints in the window are members of the query (duplicates count).
    """
    if not doc or not query or window <= 0:
        return (0, 0, [])
    n = len(doc)
    if window >= n:
        qset = set(query)
        score = sum(1 for x in doc if x in qset)
        return (0, score, doc)

    arr = np.asarray(doc)
    quniq = np.unique(np.asarray(query))
    marks = np.isin(arr, quniq).astype(np.int32)  # 1 if arr[i] in query, else 0
    sums = np.convolve(marks, np.ones(window, dtype=np.int32), mode="valid")
    best_start = int(np.argmax(sums))
    best_score = int(sums[best_start])
    return best_start, best_score, doc[best_start : best_start + window]
