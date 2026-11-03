"""8.3 short-term memory: thread state in Postgres."""
import sys

from s0_setup import *

from langchain.agents import create_agent

agent = create_agent(
    model=model,
    tools=[],
    system_prompt="You are a helpful assistant. Answer in one or two sentences.",
    checkpointer=checkpointer,
)


def ask(text: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke({"messages": [{"role": "user", "content": text}]}, config)
    print(f"[{thread_id}] USER: {text}")
    print(f"[{thread_id}] AI:   {result['messages'][-1].content}")


stage = sys.argv[1]
if stage == "a":
    ask("Hi, I'm Zhang San. I'm a Python developer.", "demo-1")
    ask("What's my name and what do I do?", "demo-1")
    ask("What's my name?", "demo-2")
elif stage == "b":
    config = {"configurable": {"thread_id": "demo-1"}}
    state = agent.get_state(config)
    print("messages in demo-1:", len(state.values["messages"]))
    for m in state.values["messages"]:
        print(f"  {m.type:>5}: {m.content[:70]}")
    print("checkpoints in demo-1:", len(list(agent.get_state_history(config))))
    ask("Remind me, what's my job?", "demo-1")
    with pool.connection() as conn:
        rows = conn.execute(
            "SELECT thread_id, count(*) AS checkpoints FROM checkpoints GROUP BY thread_id ORDER BY thread_id"
        ).fetchall()
    print(rows)
