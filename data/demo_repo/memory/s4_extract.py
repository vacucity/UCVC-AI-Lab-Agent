"""8.6 write path 2: background extraction (profile + memory collection)."""
import sys
import uuid
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from s0_setup import *


# ---------- profile: one document per user, updated with patches ----------
class UserProfile(BaseModel):
    """Every field is optional: in a patch, None means "unchanged"."""
    name: str | None = None
    job: str | None = Field(None, description="Current job title and employer")
    languages: list[str] | None = Field(None, description="Full list of programming languages the user likes")
    answer_style: str | None = Field(None, description="How the user wants answers written, e.g. 'short, code first'")


profile_writer = model.with_structured_output(UserProfile, method="function_calling")

PROFILE_PROMPT = """You maintain a user profile.
Current profile (JSON): {profile}

Read the conversation below and return a patch: fill only the fields the user changed or revealed, leave the rest null.
Only use what the user said, not the assistant."""


def update_profile(user_id: str, conversation: str) -> UserProfile:
    item = store.get((user_id, "profile"), "main")
    old = UserProfile(**item.value) if item else UserProfile()
    patch = profile_writer.invoke([
        {"role": "system", "content": PROFILE_PROMPT.format(profile=old.model_dump_json())},
        {"role": "user", "content": conversation},
    ])
    new = old.model_copy(update=patch.model_dump(exclude_none=True))   # the code merges, not the LLM
    if new != old:
        store.put((user_id, "profile"), "main", new.model_dump(), index=False)
        print(f"  profile patch: {patch.model_dump(exclude_none=True)}")
    return new


# ---------- collection: many small memories, kept clean with ADD/UPDATE/DELETE/NOOP ----------
class Facts(BaseModel):
    facts: list[str] = Field(description="Self-contained facts about the user. Empty list if none.")


class MemoryDecision(BaseModel):
    action: Literal["ADD", "UPDATE", "DELETE", "NOOP"]
    target_id: str | None = Field(None, description="Id of the existing memory to update or delete")
    content: str | None = Field(None, description="Final memory text, for ADD and UPDATE")
    kind: Literal["fact", "episode"] = Field("fact", description="'episode' for dated events and plans")
    importance: float = Field(0.5, ge=0, le=1)


fact_extractor = model.with_structured_output(Facts, method="function_calling")
decider = model.with_structured_output(MemoryDecision, method="function_calling")

EXTRACT_PROMPT = """Extract facts about the user worth remembering in future sessions: projects, plans, dated events, goals and preferences.
Do not extract anything the profile already covers (name, job, favourite programming languages, answer style).
Current profile: {profile}
Skip small talk and anything only the assistant said.
Today is {today}. Turn relative dates into absolute dates.
Write each fact as one self-contained sentence about "the user"."""

DECIDE_PROMPT = """You manage a user's long-term memory.
New fact: {fact}

Existing memories most similar to it:
{existing}

Choose one action:
- ADD: the fact is new information.
- UPDATE: it changes a detail of an existing memory, such as a new date. Give target_id and the merged content.
- DELETE: the new fact contradicts an existing memory or says it no longer holds, for example a plan was cancelled.
  Give target_id. Do not keep a record of the cancellation.
- NOOP: it is already covered, including restatements and confirmations."""


def update_memories(user_id: str, conversation: str, thread_id: str, profile: UserProfile) -> None:
    ns = (user_id, "memories")
    facts = fact_extractor.invoke([
        {"role": "system", "content": EXTRACT_PROMPT.format(today=date.today().isoformat(), profile=profile.model_dump_json())},
        {"role": "user", "content": conversation},
    ]).facts
    for fact in facts:
        similar = store.search(ns, query=fact, limit=3)
        existing = "\n".join(f"- id={i.key} | {i.value['content']}" for i in similar) or "(none)"
        d = decider.invoke(DECIDE_PROMPT.format(fact=fact, existing=existing))
        known = {i.key for i in similar}
        if d.action == "ADD":
            store.put(ns, str(uuid.uuid4()), {"content": d.content or fact, "kind": d.kind,
                                              "importance": d.importance, "source_thread": thread_id})
        elif d.action == "UPDATE" and d.target_id in known:
            store.put(ns, d.target_id, {"content": d.content, "kind": d.kind,
                                        "importance": d.importance, "source_thread": thread_id})
        elif d.action == "DELETE" and d.target_id in known:
            store.delete(ns, d.target_id)
        target = f" {d.target_id[:8]}" if d.target_id else ""
        print(f"  {d.action}{target}: {d.content or fact}")


def remember(user_id: str, conversation: str, thread_id: str) -> None:
    """Background job run after a turn: refresh the profile, then the memory collection."""
    profile = update_profile(user_id, conversation)
    update_memories(user_id, conversation, thread_id, profile)


def show_memory(user_id: str) -> None:
    item = store.get((user_id, "profile"), "main")
    print("PROFILE:", item.value if item else None)
    for i in store.search((user_id, "memories"), limit=50):
        print(f"  [{i.value['kind']}] {i.key[:8]} {i.value['content']}")


if __name__ == "__main__":
    turns = {
        "1": "user: Hi, I'm Zhang San, a Python developer at a fintech startup. I like Python, Go and Rust. "
             "Please keep answers short and show code first. I'm flying to Tokyo next month for PyCon JP, "
             "and I finished my first LangGraph project yesterday.",
        "2": "user: Change of plans: PyCon JP got moved, so I'm flying to Tokyo in December instead.",
        "3": "user: I just joined an AI startup as an agent engineer. "
             "And yes, that LangGraph project I told you about is done.",
        "4": "user: The Tokyo trip is off, my visa didn't come through.",
    }
    # start from a clean state (same as clear_user_memories("user123") in 8.4)
    for namespace in store.list_namespaces(prefix=("user123",)):
        for item in store.search(namespace, limit=1000):
            store.delete(namespace, item.key)
    for n in sys.argv[1:]:
        print(f"--- turn {n}: {turns[n]}")
        remember("user123", turns[n], thread_id=f"seed-{n}")
    show_memory("user123")
