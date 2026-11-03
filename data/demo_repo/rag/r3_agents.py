"""8.15-8.17 generation and three ways to combine an agent with RAG."""
import sys
from typing import TypedDict

from r2_retrieve import *

from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.graph import END, START, StateGraph


# ---------- grounded generation ----------
ANSWER_PROMPT = """Answer the question using only the numbered passages from the Hello-Agents tutorial.
Cite the passages you use like [1] or [2][3].
If the passages don't contain the answer, say you couldn't find it in the tutorial. Don't guess.
Answer in the same language as the question."""


def format_passages(docs: list[dict]) -> str:
    return "\n\n".join(f"[{i}] (Ch.{d['chapter_no']}, p.{d['page']}) {d['content']}" for i, d in enumerate(docs, 1))


def generate(question: str, docs: list[dict]) -> str:
    return model.invoke([
        {"role": "system", "content": ANSWER_PROMPT},
        {"role": "user", "content": f"Passages:\n{format_passages(docs)}\n\nQuestion: {question}"},
    ]).content


# ---------- pattern 1: two-step RAG (always retrieve, then answer) ----------
def retrieve(question: str) -> list[dict]:
    return llm_rerank(question, hybrid_search(question, k=12), top_n=4)


def two_step_rag(question: str) -> str:
    docs = retrieve(question)
    print("  pages:", [d["page"] for d in docs])
    return generate(question, docs)


# ---------- pattern 2: agentic RAG (retriever as a tool) ----------
@tool
def search_docs(query: str, chapter: int | None = None) -> str:
    """Search the Hello-Agents tutorial, an English book about building AI agents.

    Args:
        query: A focused English search query. Translate non-English questions first.
        chapter: Optional chapter number (1-16) to search within.
    """
    docs = hybrid_search(query, k=5, chapter_no=chapter)
    return "\n\n".join(f"(Ch.{d['chapter_no']}, p.{d['page']}) {d['content']}" for d in docs) or "No passages found."


RAG_AGENT_PROMPT = """You are a tutor for the Hello-Agents tutorial, and you answer only from the tutorial.
For every question that isn't small talk, call search_docs first, even if you think you already know the answer.
Search more than once when a question has several parts, for example once per chapter when comparing.
Cite pages like (Ch.9, p.372). If the results don't cover the question, say the tutorial doesn't cover it.
Reply to small talk directly, without searching. Answer in the user's language."""

rag_agent = create_agent(model=model, tools=[search_docs], system_prompt=RAG_AGENT_PROMPT)

from langchain.agents.middleware import ModelRequest, wrap_model_call


@wrap_model_call
def search_first(request: ModelRequest, handler):
    """Force a tool call on the first model call of each turn (the last message is the user's)."""
    if request.messages[-1].type == "human":
        request = request.override(tool_choice="any")
    return handler(request)


strict_agent = create_agent(model=model, tools=[search_docs], system_prompt=RAG_AGENT_PROMPT, middleware=[search_first])


def ask_agent(question: str, agent=rag_agent):
    result = agent.invoke({"messages": [{"role": "user", "content": question}]}, {"recursion_limit": 12})
    for m in result["messages"]:
        for call in getattr(m, "tool_calls", None) or []:
            print(f"  tool call -> {call['name']}({call['args']})")
    print("REPLY:", result["messages"][-1].content)


# ---------- pattern 3: corrective RAG (a graph that checks its own retrieval) ----------
class RAGState(TypedDict, total=False):
    question: str
    queries: list[str]
    docs: list[dict]
    attempts: int
    enough: bool
    missing: str
    answer: str
    grounded: bool


class Grade(BaseModel):
    enough: bool = Field(description="True only if the passages contain everything needed to fully answer the question")
    missing: str = Field(description="What information is still missing, or 'nothing'")


class Rewrite(BaseModel):
    queries: list[str] = Field(description="1-3 short English search queries, one per missing piece")


class Grounding(BaseModel):
    grounded: bool = Field(description="True if every claim in the answer is supported by the passages")


grader = model.with_structured_output(Grade, method="function_calling")
rewriter = model.with_structured_output(Rewrite, method="function_calling")
grounding_judge = model.with_structured_output(Grounding, method="function_calling")

MAX_ATTEMPTS = 2


def retrieve_node(state: RAGState) -> RAGState:
    per_query = 4 if len(state["queries"]) == 1 else 2
    found = []
    for q in state["queries"]:                    # each sub-query is searched and reranked on its own
        found += llm_rerank(q, hybrid_search(q, k=10), top_n=per_query)
    seen = {d["id"] for d in state.get("docs", [])}
    docs = state.get("docs", []) + [d for d in found if d["id"] not in seen]   # keep earlier finds
    attempts = state.get("attempts", 0) + 1
    print(f"  retrieve #{attempts}: {state['queries']} -> new pages {[d['page'] for d in found if d['id'] not in seen]}")
    return {"docs": docs[:10], "attempts": attempts}             # cap the context, keep the earliest finds


def grade_node(state: RAGState) -> RAGState:
    g = grader.invoke(f"Question: {state['question']}\n\nPassages:\n{format_passages(state['docs'])}")
    print(f"  grade: enough={g.enough}, missing={g.missing!r}")
    return {"enough": g.enough, "missing": g.missing}


def rewrite_node(state: RAGState) -> RAGState:
    r = rewriter.invoke(f"Question: {state['question']}\nAlready searched: {state['queries']}\n"
                        f"Still missing: {state['missing']}\n"
                        "Write one short English search query per missing piece. Use only specific terms "
                        "from the topic; don't add generic words like 'tutorial', 'AI', 'agent' or 'explanation'.")
    queries = r.queries[:3]                       # the schema says 1-3, but nothing enforces it
    print(f"  rewrite -> {queries}")
    return {"queries": queries}


def generate_node(state: RAGState) -> RAGState:
    question = state["question"]
    if not state.get("enough"):
        question += "\n(The passages may not cover everything. Answer only what they support and say what is missing.)"
    return {"answer": generate(question, state["docs"])}


def check_node(state: RAGState) -> RAGState:
    v = grounding_judge.invoke(f"Passages:\n{format_passages(state['docs'])}\n\nAnswer:\n{state['answer']}")
    print(f"  grounded: {v.grounded}")
    return {"grounded": v.grounded}


def after_grade(state: RAGState) -> str:
    if state["enough"] or state["attempts"] >= MAX_ATTEMPTS:
        return "generate"
    return "rewrite"


builder = StateGraph(RAGState)
builder.add_node("retrieve", retrieve_node)
builder.add_node("grade", grade_node)
builder.add_node("rewrite", rewrite_node)
builder.add_node("generate", generate_node)
builder.add_node("check", check_node)
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "grade")
builder.add_conditional_edges("grade", after_grade, ["generate", "rewrite"])
builder.add_edge("rewrite", "retrieve")
builder.add_edge("generate", "check")
builder.add_edge("check", END)
corrective_rag = builder.compile()


def ask_graph(question: str) -> RAGState:
    state = corrective_rag.invoke({"question": question, "queries": [question]})
    print("ANSWER:", state["answer"])
    return state


if __name__ == "__main__":
    part = sys.argv[1]
    if part == "two":
        for q in ["What does GSSC stand for, and what does each step do?", "为什么上下文越长，模型的表现反而会下降？"]:
            print("Q:", q)
            print("A:", two_step_rag(q), "\n")
    elif part == "agent":
        for q in ["Hi! Thanks for helping me study yesterday.",
                  "比较一下第 8 章的 RAG 和第 9 章的上下文工程，它们分别解决什么问题？",
                  "What is the capital of Australia?"]:
            print("Q:", q)
            ask_agent(q)
            print()
    elif part == "graph":
        for q in ["What does GSSC stand for, and what does each step do?", "What is the capital of Australia?"]:
            print("Q:", q)
            ask_graph(q)
            print()
