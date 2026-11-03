"""Cell 1: model, embeddings, Postgres checkpointer and store."""
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

load_dotenv(r"C:\Users\23017\Desktop\UCVC Agent\.env")

model = ChatOpenAI(
    model=os.getenv("LLM_MODEL_ID", "deepseek-flash"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
    temperature=0,
    extra_body={"thinking": {"type": "disabled"}},
)

# bge-m3 served by local Ollama through its OpenAI-compatible API
embeddings = OpenAIEmbeddings(
    model="bge-m3",
    base_url="http://localhost:11434/v1",
    api_key="ollama",                  # Ollama ignores the key, but the client needs one
    check_embedding_ctx_length=False,  # send raw text, not OpenAI token ids
)

# One connection pool shared by the checkpointer, the store and background workers
pool = ConnectionPool(
    os.getenv("DATABASE_URL"),
    max_size=10,
    kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    open=True,
)

checkpointer = PostgresSaver(pool)          # short-term memory: thread state
store = PostgresStore(                      # long-term memory: per-user items
    pool,
    index={"dims": 1024, "embed": embeddings, "fields": ["content"]},
)
checkpointer.setup()   # creates tables on first run, no-op afterwards
store.setup()
