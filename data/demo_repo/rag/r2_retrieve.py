"""8.13 online retrieval: vector, keyword, hybrid (RRF), multi-query, LLM rerank."""
import sys

sys.path.insert(0, r"C:\Users\23017\AppData\Local\Temp\claude\C--Users-23017-Desktop-UCVC-Agent\02fa6926-7754-485a-8016-abe8b45d82c6\scratchpad\ch8e")
from s0_setup import *

from pydantic import BaseModel, Field

COLUMNS = "id, chapter_no, section, page, content"


def vector_search(query: str, k: int = 10, chapter_no: int | None = None) -> list[dict]:
    """Nearest chunks by cosine distance (pgvector HNSW index)."""
    qv = str(embeddings.embed_query(query))
    where = "WHERE chapter_no = %(ch)s" if chapter_no else ""
    sql = (f"SELECT {COLUMNS}, 1 - (embedding <=> %(qv)s::vector) AS score FROM doc_chunks {where} "
           "ORDER BY embedding <=> %(qv)s::vector LIMIT %(k)s")
    with pool.connection() as conn:
        return conn.execute(sql, {"qv": qv, "k": k, "ch": chapter_no}).fetchall()


def keyword_search(query: str, k: int = 10, chapter_no: int | None = None) -> list[dict]:
    """Full-text search: any query term may match (OR), ranked by ts_rank_cd."""
    where = "AND chapter_no = %(ch)s" if chapter_no else ""
    sql = (f"SELECT {COLUMNS}, ts_rank_cd(tsv, query.q) AS score FROM doc_chunks, "
           "(SELECT replace(plainto_tsquery('english', %(q)s)::text, '&', '|')::tsquery AS q) AS query "
           f"WHERE tsv @@ query.q {where} ORDER BY score DESC LIMIT %(k)s")
    with pool.connection() as conn:
        return conn.execute(sql, {"q": query, "k": k, "ch": chapter_no}).fetchall()


import re

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())   # keeps identifiers like chunk_overlap


with pool.connection() as conn:   # 1834 chunks fit in memory; rebuild the index when documents change
    corpus = conn.execute(f"SELECT {COLUMNS} FROM doc_chunks ORDER BY id").fetchall()
bm25 = BM25Okapi([tokenize(d["content"]) for d in corpus])


def bm25_search(query: str, k: int = 10, chapter_no: int | None = None) -> list[dict]:
    """Real BM25: rare terms weigh more (IDF), long chunks are normalized."""
    scores = bm25.get_scores(tokenize(query))
    results = []
    for i in scores.argsort()[::-1]:
        if scores[i] <= 0 or len(results) == k:
            break
        if chapter_no is None or corpus[i]["chapter_no"] == chapter_no:
            results.append(corpus[i] | {"score": float(scores[i])})
    return results


def rrf(result_lists: list[list[dict]], k: int = 5, c: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion: score = sum of 1 / (c + rank) over all lists."""
    scores, docs = {}, {}
    for results in result_lists:
        for rank, d in enumerate(results, 1):
            scores[d["id"]] = scores.get(d["id"], 0) + 1 / (c + rank)
            docs[d["id"]] = d
    best = sorted(scores, key=scores.get, reverse=True)[:k]
    return [docs[i] | {"score": scores[i]} for i in best]


def hybrid_search(query: str, k: int = 5, chapter_no: int | None = None, pool_k: int = 20) -> list[dict]:
    return rrf([vector_search(query, pool_k, chapter_no), bm25_search(query, pool_k, chapter_no)], k)


class Queries(BaseModel):
    queries: list[str] = Field(description="3 short English search queries, each using different keywords")


query_writer = model.with_structured_output(Queries, method="function_calling")


def multi_query_search(question: str, k: int = 5, chapter_no: int | None = None) -> list[dict]:
    """Rewrite the question into several English queries, search each, fuse with RRF."""
    queries = query_writer.invoke(
        "Write 3 short English search queries for finding the answer to this question "
        f"in an English tutorial about building AI agents.\nQuestion: {question}").queries
    lists = []
    for q in [question, *queries]:
        lists += [vector_search(q, 20, chapter_no), bm25_search(q, 20, chapter_no)]
    return rrf(lists, k)


class Ranking(BaseModel):
    ids: list[int] = Field(description="Numbers of the passages that help answer the question, most useful first")


ranker = model.with_structured_output(Ranking, method="function_calling")


def llm_rerank(question: str, docs: list[dict], top_n: int = 4) -> list[dict]:
    """Listwise rerank with the LLM: it reads every candidate and orders the useful ones."""
    listing = "\n\n".join(f"[{i}] {d['content'][:600]}" for i, d in enumerate(docs))
    ids = ranker.invoke(f"Question: {question}\n\nPassages:\n{listing}\n\n"
                        "Return the numbers of the passages that help answer the question, most useful first.").ids
    return [docs[i] for i in ids if 0 <= i < len(docs)][:top_n]


def show(results: list[dict], width: int = 90):
    for d in results:
        text = " ".join(d["content"].split())[:width]
        print(f"  {d['score']:.3f}  p.{d['page']:<4} {text}")
