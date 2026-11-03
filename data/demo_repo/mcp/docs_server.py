"""Hello-Agents docs MCP server: Chapter 8's hybrid search over Streamable HTTP, behind a bearer token."""
import os
import re
from typing import Annotated

import uvicorn
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from mcp.server.fastmcp import FastMCP
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

load_dotenv()
TOKEN = os.getenv("DOCS_MCP_TOKEN", "dev-token-change-me")
PORT = int(os.getenv("DOCS_MCP_PORT", "8931"))

embeddings = OpenAIEmbeddings(model="bge-m3", base_url="http://localhost:11434/v1", api_key="ollama",
                              check_embedding_ctx_length=False)
pool = ConnectionPool(os.getenv("DATABASE_URL"), max_size=5, open=True,
                      kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row})

COLUMNS = "id, chapter_no, page, content"
with pool.connection() as conn:
    corpus = conn.execute(f"SELECT {COLUMNS} FROM doc_chunks ORDER BY id").fetchall()
bm25 = BM25Okapi([re.findall(r"[a-z0-9_]+", d["content"].lower()) for d in corpus])


def vector_search(query: str, k: int, chapter_no: int | None) -> list[dict]:
    qv = str(embeddings.embed_query(query))
    where = "WHERE chapter_no = %(ch)s" if chapter_no else ""
    with pool.connection() as conn:
        return conn.execute(f"SELECT {COLUMNS} FROM doc_chunks {where} "
                            "ORDER BY embedding <=> %(qv)s::vector LIMIT %(k)s",
                            {"qv": qv, "k": k, "ch": chapter_no}).fetchall()


def bm25_search(query: str, k: int, chapter_no: int | None) -> list[dict]:
    scores = bm25.get_scores(re.findall(r"[a-z0-9_]+", query.lower()))
    hits = [corpus[i] for i in scores.argsort()[::-1] if scores[i] > 0
            and (chapter_no is None or corpus[i]["chapter_no"] == chapter_no)]
    return hits[:k]


def hybrid_search(query: str, k: int = 5, chapter_no: int | None = None) -> list[dict]:
    scores, docs = {}, {}
    for results in (vector_search(query, 20, chapter_no), bm25_search(query, 20, chapter_no)):
        for rank, d in enumerate(results, 1):
            scores[d["id"]] = scores.get(d["id"], 0) + 1 / (60 + rank)
            docs[d["id"]] = d
    return [docs[i] for i in sorted(scores, key=scores.get, reverse=True)[:k]]


mcp = FastMCP("hello-agents-docs", stateless_http=True)


class Passage(BaseModel):
    chapter: int | None
    page: int
    text: str


class SearchResult(BaseModel):
    passages: list[Passage]


@mcp.tool()
def search_docs(
    query: Annotated[str, Field(description="A focused English search query; translate other languages first")],
    chapter: Annotated[int | None, Field(description="Optional chapter number (1-16) to search within")] = None,
) -> SearchResult:
    """Search the Hello-Agents tutorial (an English book about building AI agents). Cite results as (Ch.N, p.M)."""
    return SearchResult(passages=[Passage(chapter=d["chapter_no"], page=d["page"], text=d["content"])
                                  for d in hybrid_search(query, 5, chapter)])


@mcp.resource("docs://chapters")
def chapters() -> str:
    """Chapter numbers and titles of the tutorial."""
    with pool.connection() as conn:
        rows = conn.execute("SELECT chapter_no, min(chapter) AS title FROM doc_chunks "
                            "WHERE chapter_no IS NOT NULL GROUP BY chapter_no ORDER BY chapter_no").fetchall()
    return "\n".join(r["title"] for r in rows)


class BearerToken(BaseHTTPMiddleware):
    """Reject requests without the right bearer token (a stand-in for OAuth or an API gateway)."""

    async def dispatch(self, request, call_next):
        if request.headers.get("authorization") != f"Bearer {TOKEN}":
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)


if __name__ == "__main__":
    app = mcp.streamable_http_app()            # endpoint: /mcp
    app.add_middleware(BearerToken)
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
