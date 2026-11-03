"""8.18 one agent with both: user memory (last lesson) + the tutorial knowledge base."""
import json
import sys
from datetime import date

from r3_agents import *

sys.path.insert(0, r"C:\Users\23017\AppData\Local\Temp\claude\C--Users-23017-Desktop-UCVC-Agent\02fa6926-7754-485a-8016-abe8b45d82c6\scratchpad\ch8e")
from s3_tools import UserContext, print_turn, search_memories
from s4_extract import remember, show_memory
from s5_read import rerank

from langchain.agents.middleware import ModelRequest, dynamic_prompt


@dynamic_prompt
def tutor_prompt(request: ModelRequest) -> str:
    """Tutor rules + what we know about this user. Documents are fetched by the search_docs tool."""
    user_id = request.runtime.context.user_id
    store = request.runtime.store
    parts = [RAG_AGENT_PROMPT, f"Today is {date.today().isoformat()}."]

    item = store.get((user_id, "profile"), "main")
    profile = {k: v for k, v in (item.value if item else {}).items() if v}
    if profile:
        parts.append(f"User profile: {json.dumps(profile, ensure_ascii=False)}")
    if profile.get("answer_style"):
        parts.append(f"Always follow the user's preferred answer style: {profile['answer_style']}.")

    query = next(m.text for m in reversed(request.messages) if m.type == "human")
    top = rerank(store.search((user_id, "memories"), query=query, limit=10))[:3]
    if top:
        parts.append("What you remember about the user (use only when it helps):\n"
                     + "\n".join(f"- {row[3].value['content']}" for row in top))
    return "\n\n".join(parts)


tutor = create_agent(
    model=model,
    tools=[search_docs, search_memories],
    middleware=[tutor_prompt, search_first],
    checkpointer=checkpointer,
    store=store,
    context_schema=UserContext,
)


def tutor_chat(text: str, thread_id: str, user_id: str = "user123"):
    result = tutor.invoke(
        {"messages": [{"role": "user", "content": text}]},
        {"configurable": {"thread_id": thread_id}, "metadata": {"user_id": user_id}, "recursion_limit": 12},
        context=UserContext(user_id=user_id),
    )
    print(f"USER: {text}")
    print_turn(result)


if __name__ == "__main__":
    remember("user123", "user: I'm Zhang San, an agent engineer. I have agent engineer interviews next month. "
                        "Please keep answers short and show code first.", "seed-rag")
    show_memory("user123")
    tutor_chat("RAG 里的 HyDE 是什么？怎么实现？", "tutor-1")
    tutor_chat("这个知识点面试时可能会怎么问？", "tutor-1")
