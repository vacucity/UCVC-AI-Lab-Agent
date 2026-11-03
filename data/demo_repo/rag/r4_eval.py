"""8.19 evaluation: retrieval Hit@5 / MRR per method, then answer groundedness and refusals."""
import sys
import time

from r2_retrieve import *

EVAL_SET = [
    # exact terms: keyword search should shine
    {"q": "What does GSSC stand for in the ContextBuilder?", "evidence": "gather-select-structure-compress"},
    {"q": "What does BFCL stand for?", "evidence": "berkeley function calling leaderboard"},
    {"q": "Which AutoGen condition stops the chat when the word TERMINATE appears?", "evidence": "textmentiontermination"},
    {"q": "What chunk_size and chunk_overlap does the RAG pipeline use?", "evidence": "chunk_overlap=200"},
    # paraphrases: no shared keywords, vector search should shine
    {"q": "Why does stuffing more and more tokens into the context window make the model worse?", "evidence": "context rot"},
    {"q": "What is the trick of writing a fake answer first and searching with its embedding?", "evidence": "hypothetical document embeddings"},
    {"q": "Which thinker argued that intelligence emerges from many simple mindless agents working together?", "evidence": "society of mind"},
    {"q": "How does CAMEL keep two role-playing agents on task without a human in the loop?", "evidence": "inception prompting"},
    {"q": "Which RL algorithm drops the value model and uses group-relative rewards instead?", "evidence": "no need for value model"},
    # Chinese questions over an English book: keyword search can't match at all
    {"q": "为什么上下文越长，模型的表现反而会下降？", "evidence": "context rot"},
    {"q": "多查询扩展是怎么提高检索召回率的？", "evidence": "multi-query expansion (mqe) is a technique"},
    {"q": "第 12 章用什么指标比较生成的题目和真实题目谁更好？", "evidence": "win rate"},
    {"q": "深度研究智能体用的是什么研究范式？", "evidence": "todo-driven research"},
    {"q": "GRPO 相比 PPO 去掉了哪个模型？", "evidence": "no need for value model"},
]


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def first_hit(results: list[dict], evidence: str) -> int | None:
    """1-based rank of the first chunk that contains the evidence string."""
    for rank, d in enumerate(results, 1):
        if evidence in normalize(d["content"]):
            return rank
    return None


METHODS = {
    "vector": lambda q: vector_search(q, 5),
    "pg keyword": lambda q: keyword_search(q, 5),
    "bm25": lambda q: bm25_search(q, 5),
    "hybrid": lambda q: hybrid_search(q, 5),
    "multi-query": lambda q: multi_query_search(q, 5),
    "hybrid+rerank": lambda q: llm_rerank(q, hybrid_search(q, k=12), top_n=5),
}


def eval_retrieval(methods: dict = METHODS, eval_set: list[dict] = EVAL_SET):
    rows = {}
    for name, search in methods.items():
        t0 = time.perf_counter()
        ranks = [first_hit(search(item["q"]), item["evidence"]) for item in eval_set]
        hit = sum(r is not None for r in ranks) / len(ranks)
        mrr = sum(1 / r for r in ranks if r) / len(ranks)
        rows[name] = ranks
        print(f"{name:<14} Hit@5 {hit:.2f}  MRR {mrr:.2f}  ({(time.perf_counter() - t0) / len(eval_set):.1f}s/query)  "
              + " ".join(str(r) if r else "-" for r in ranks))
    return rows


if __name__ == "__main__":
    with pool.connection() as conn:   # sanity check: every evidence string exists somewhere in the corpus
        for item in EVAL_SET:
            n = conn.execute("SELECT count(*) AS n FROM doc_chunks WHERE lower(regexp_replace(content, '\\s+', ' ', 'g')) LIKE %s",
                             (f"%{item['evidence']}%",)).fetchone()["n"]
            if n == 0:
                print("MISSING evidence:", item["evidence"])
    eval_retrieval()
