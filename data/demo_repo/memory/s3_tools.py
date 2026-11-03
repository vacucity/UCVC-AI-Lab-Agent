"""8.5 write path 1: hot path, the agent saves memories with tools."""
import sys
import uuid
from dataclasses import dataclass
from datetime import date
from typing import Literal

from s0_setup import *

from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool


@dataclass
class UserContext:
    user_id: str   # filled by the server after auth, never by the model


@tool
def save_memory(content: str, kind: Literal["fact", "episode"], importance: float,
                runtime: ToolRuntime[UserContext]) -> str:
    """Save something worth remembering about the user for future conversations.

    Args:
        content: The full information to remember, in one self-contained sentence.
        kind: "fact" for lasting knowledge or preferences, "episode" for a dated event.
        importance: How important it is, from 0.0 to 1.0.
    """
    key = str(uuid.uuid4())
    runtime.store.put((runtime.context.user_id, "memories"), key,
                      {"content": content, "kind": kind, "importance": importance})
    return f"Saved {kind} {key[:8]}: {content}"


@tool
def search_memories(query: str, runtime: ToolRuntime[UserContext]) -> str:
    """Search what you remember about the user. Use it before saying you don't know something.

    Args:
        query: What to look for, e.g. "user's travel plans".
    """
    items = runtime.store.search((runtime.context.user_id, "memories"), query=query, limit=5)
    lines = [f"[{i.value['kind']}] {i.value['content']} (score {i.score:.2f})" for i in items]
    return "\n".join(lines) or "No memories found."


def print_turn(result):
    """Print this turn's tool calls and the final reply."""
    last_human = max(i for i, m in enumerate(result["messages"]) if m.type == "human")
    for m in result["messages"][last_human:]:
        for call in getattr(m, "tool_calls", None) or []:
            print(f"  tool call -> {call['name']}({call['args']})")
    print("REPLY:", result["messages"][-1].content)


if __name__ == "__main__":
    hot_agent = create_agent(
        model=model,
        tools=[save_memory, search_memories],
        system_prompt=("You are an assistant with long-term memory. Save important facts about the user, "
                       "and search your memory before saying you don't know something. "
                       f"Today is {date.today().isoformat()}; turn relative dates into real dates."),
        checkpointer=checkpointer,
        store=store,
        context_schema=UserContext,
    )

    def hot_chat(text: str, thread_id: str):
        result = hot_agent.invoke(
            {"messages": [{"role": "user", "content": text}]},
            {"configurable": {"thread_id": thread_id}},
            context=UserContext(user_id="user123"),
        )
        print_turn(result)

    if sys.argv[1] == "a":
        hot_chat("Please remember: I'm Zhang San, a Python developer, and I like Python, Go and Rust.", "hot-1")
        hot_chat("Also, I finished my first LangGraph project yesterday.", "hot-1")
    else:
        hot_chat("What do you know about me?", "hot-2")
