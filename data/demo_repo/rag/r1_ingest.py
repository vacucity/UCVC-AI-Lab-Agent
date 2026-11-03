"""8.12 offline ingestion: PDF -> pages with chapter/section -> chunks -> embeddings -> Postgres."""
import hashlib
import re
import sys
import time

sys.path.insert(0, r"C:\Users\23017\AppData\Local\Temp\claude\C--Users-23017-Desktop-UCVC-Agent\02fa6926-7754-485a-8016-abe8b45d82c6\scratchpad\ch8e")
from s0_setup import *

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

PDF_PATH = r"C:\Users\23017\Desktop\UCVC Agent\Hello-Agents-EN-V1.0.3-20260717.pdf"
SOURCE = "hello-agents-en-v1.0.3"


def clean_title(title: str) -> str:
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", title).strip()   # "AgentParadigms" -> "Agent Paradigms"


def load_pages(path: str) -> list[dict]:
    """One dict per page, tagged with the chapter and section it falls in (from the PDF outline)."""
    reader = PdfReader(path)
    marks = []

    def walk(items, depth):
        for it in items:
            if isinstance(it, list):
                walk(it, depth + 1)
            else:
                marks.append((reader.get_destination_page_number(it), depth, clean_title(it.title)))

    walk(reader.outline, 0)
    marks.sort(key=lambda m: m[0])

    pages, chapter, section, j = [], "", "", 0
    for i, page in enumerate(reader.pages):
        while j < len(marks) and marks[j][0] <= i:
            _, depth, title = marks[j]
            if depth == 0:
                chapter, section = title, ""
            elif depth == 1:
                section = title
            j += 1
        text = re.sub(r"[ \t]+", " ", page.extract_text() or "").strip()
        pages.append({"page": i + 1, "chapter": chapter, "section": section, "text": text})
    return pages


splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)


def make_chunks(pages: list[dict]) -> list[dict]:
    chunks = []
    for p in pages:
        for piece in splitter.split_text(p["text"]):
            if len(piece) < 50:          # page numbers, stray headers
                continue
            chunk_id = hashlib.sha1(f"{SOURCE}:{p['page']}:{piece}".encode()).hexdigest()
            chunks.append({"id": chunk_id, "source": SOURCE, "chapter": p["chapter"],
                           "section": p["section"], "page": p["page"], "content": piece})
    return chunks


SCHEMA = """
CREATE TABLE IF NOT EXISTS doc_chunks (
    id        text PRIMARY KEY,
    source    text NOT NULL,
    chapter   text,
    chapter_no int GENERATED ALWAYS AS (substring(chapter from 'Chapter ([0-9]+)')::int) STORED,
    section   text,
    page      int,
    content   text NOT NULL,
    embedding vector(1024) NOT NULL,
    tsv       tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);
CREATE INDEX IF NOT EXISTS doc_chunks_embedding_idx ON doc_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS doc_chunks_tsv_idx ON doc_chunks USING gin (tsv);
"""


def ingest(chunks: list[dict], batch_size: int = 64) -> int:
    with pool.connection() as conn:
        for statement in SCHEMA.split(";"):      # one statement per call: the pool prepares every query
            if statement.strip():
                conn.execute(statement)
        existing = {r["id"] for r in conn.execute("SELECT id FROM doc_chunks WHERE source = %s", (SOURCE,))}
    todo = [c for c in chunks if c["id"] not in existing]     # idempotent: only embed new chunks
    t0 = time.perf_counter()
    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        vectors = embeddings.embed_documents([c["content"] for c in batch])
        with pool.connection() as conn:
            for c, v in zip(batch, vectors):
                conn.execute(
                    "INSERT INTO doc_chunks (id, source, chapter, section, page, content, embedding) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s::vector) ON CONFLICT (id) DO NOTHING",
                    (c["id"], c["source"], c["chapter"], c["section"], c["page"], c["content"], str(v)),
                )
        if (i // batch_size) % 5 == 0:
            print(f"  embedded {i + len(batch)}/{len(todo)} ({time.perf_counter() - t0:.0f}s)")
    return len(todo)


if __name__ == "__main__":
    t0 = time.perf_counter()
    pages = load_pages(PDF_PATH)
    print(f"pages: {len(pages)} ({time.perf_counter() - t0:.0f}s)")
    print("page 320:", pages[319]["chapter"], "|", pages[319]["section"])
    chunks = make_chunks(pages)
    print(f"chunks: {len(chunks)}, avg {sum(len(c['content']) for c in chunks) // len(chunks)} chars")
    print("new chunks embedded:", ingest(chunks))
    with pool.connection() as conn:
        print(conn.execute("SELECT count(*) AS chunks, count(DISTINCT chapter) AS chapters FROM doc_chunks").fetchone())
        print(conn.execute("SELECT chapter, count(*) AS n FROM doc_chunks GROUP BY chapter ORDER BY min(page) LIMIT 20").fetchall())
    print(f"total {time.perf_counter() - t0:.0f}s")
