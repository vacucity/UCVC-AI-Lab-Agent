"""8.8 the full system: injected reads + background writes + search tool."""
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from s0_setup import *
from s3_tools import UserContext, print_turn, search_memories
from s4_extract import remember, show_memory
from s5_read import memory_prompt

from langchain.agents import create_agent

agent = create_agent(
    model=model,
    tools=[search_memories],          # for deeper lookups; the common case is already injected
    middleware=[memory_prompt],
    checkpointer=checkpointer,
    store=store,
    context_schema=UserContext,
)

memory_worker = ThreadPoolExecutor(max_workers=2)   # stands in for a real job queue


def timed_remember(user_id: str, conversation: str, thread_id: str) -> None:
    t0 = time.perf_counter()
    remember(user_id, conversation, thread_id)
    print(f"  (memory job took {time.perf_counter() - t0:.1f}s)")


def chat(text: str, thread_id: str, user_id: str = "user123"):
    t0 = time.perf_counter()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": text}]},
        {"configurable": {"thread_id": thread_id}, "metadata": {"user_id": user_id}},
        context=UserContext(user_id=user_id),
    )
    print(f"USER: {text}")
    print_turn(result)
    print(f"  (reply took {time.perf_counter() - t0:.1f}s)")

    last_human = max(i for i, m in enumerate(result["messages"]) if m.type == "human")
    roles = {"human": "user", "ai": "assistant"}
    conversation = "\n".join(f"{roles[m.type]}: {m.text}" for m in result["messages"][last_human:]
                             if m.type in roles and m.text)
    return memory_worker.submit(timed_remember, user_id, conversation, thread_id)   # returns at once


if __name__ == "__main__":
    if sys.argv[1] == "a":
        chat("How do I reverse a list in Python?", "s-1").result()
        chat("What do you know about me?", "s-1").result()
        chat("I've started learning Rust for a side project: a CLI tool for log analysis.", "s-1").result()
    else:
        chat("What side project am I working on these days?", "s-2").result()
        show_memory("user123")
