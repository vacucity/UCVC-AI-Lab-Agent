"""8.7 read path: rerank + inject profile and memories with middleware."""
import json
from datetime import date, datetime, timezone

from s0_setup import *

from langchain.agents.middleware import ModelRequest, dynamic_prompt


def rerank(items, half_life_days: float = 30, min_similarity: float = 0.4):
    """Score = 0.7 * similarity + 0.2 * recency + 0.1 * importance. Drops weak matches."""
    now = datetime.now(timezone.utc)
    scored = []
    for item in items:
        if item.score is None or item.score < min_similarity:
            continue
        age_days = (now - item.updated_at).total_seconds() / 86400
        recency = 0.5 ** (age_days / half_life_days)          # halves every 30 days
        importance = item.value.get("importance", 0.5)
        final = 0.7 * item.score + 0.2 * recency + 0.1 * importance
        scored.append((final, item.score, recency, item))
    return sorted(scored, key=lambda row: row[0], reverse=True)


BASE_PROMPT = ("You are a helpful assistant with long-term memory about the user. "
               "The profile and memories below were loaded for this message, so use them directly; "
               "call search_memories only when they don't answer the question. "
               "Use a memory only when it helps with the current message; never bring it up unprompted.")


@dynamic_prompt
def memory_prompt(request: ModelRequest) -> str:
    """Build the system prompt before every model call: profile + top memories."""
    user_id = request.runtime.context.user_id
    store = request.runtime.store
    parts = [BASE_PROMPT, f"Today is {date.today().isoformat()}."]

    item = store.get((user_id, "profile"), "main")
    profile = {k: v for k, v in (item.value if item else {}).items() if v}
    if profile:
        parts.append(f"User profile: {json.dumps(profile, ensure_ascii=False)}")
    if profile.get("answer_style"):
        parts.append(f"Always follow the user's preferred answer style: {profile['answer_style']}.")

    query = next(m.text for m in reversed(request.messages) if m.type == "human")
    top = rerank(store.search((user_id, "memories"), query=query, limit=10))[:3]
    if top:
        parts.append("Relevant memories:\n" + "\n".join(f"- {row[3].value['content']}" for row in top))
    return "\n\n".join(parts)


if __name__ == "__main__":
    ns = ("demo", "memories")
    store.put(ns, "old", {"content": "The user lives in Hangzhou.", "kind": "fact", "importance": 0.5})
    store.put(ns, "new", {"content": "The user moved to Shanghai for a new job.", "kind": "fact", "importance": 0.5})
    with pool.connection() as conn:   # demo only: pretend "old" was written 400 days ago
        conn.execute("UPDATE store SET updated_at = now() - interval '400 days' "
                     "WHERE prefix = 'demo.memories' AND key = 'old'")

    hits = store.search(ns, query="Where does the user live?", limit=5)
    print("by similarity only:")
    for item in hits:
        print(f"  {item.score:.3f}  {item.value['content']}")
    print("after rerank:")
    for final, sim, recency, item in rerank(hits):
        print(f"  {final:.3f}  (sim {sim:.3f}, recency {recency:.3f})  {item.value['content']}")
