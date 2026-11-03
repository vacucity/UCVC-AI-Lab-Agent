"""8.4 long-term memory store basics."""
from s0_setup import *

ns = ("user123", "memories")

store.put(ns, "m1", {"content": "Zhang San is a Python developer at a fintech startup.", "kind": "fact", "importance": 0.9})
store.put(ns, "m2", {"content": "Zhang San likes Python, Go and Rust.", "kind": "fact", "importance": 0.6})
store.put(ns, "m3", {"content": "On 2026-10-29 Zhang San finished their first LangGraph project.", "kind": "episode", "importance": 0.7})

# profile: one structured document, read by key, no vector index needed
store.put(("user123", "profile"), "main", {"name": "Zhang San", "answer_style": "short, code first"}, index=False)

print("get:", store.get(ns, "m1").value)

print("\nsearch 'which programming languages does Zhang San enjoy':")
for item in store.search(ns, query="which programming languages does Zhang San enjoy", limit=3):
    print(f"  {item.score:.3f}  {item.key}  {item.value['content']}")

print("\nfilter kind=episode:")
for item in store.search(ns, filter={"kind": "episode"}):
    print(f"  {item.key}  {item.value['content']}")

print("\nnamespaces of user123:", store.list_namespaces(prefix=("user123",)))
print("user999 sees:", store.search(("user999", "memories"), query="Python developer"))

store.delete(ns, "m3")
print("after delete:", sorted(item.key for item in store.search(ns)))


def clear_user_memories(user_id: str) -> int:
    """Delete every long-term item of one user ("forget me")."""
    deleted = 0
    for namespace in store.list_namespaces(prefix=(user_id,)):
        for item in store.search(namespace, limit=1000):
            store.delete(namespace, item.key)
            deleted += 1
    return deleted


print("cleared:", clear_user_memories("user123"), "items")
print("namespaces left:", store.list_namespaces(prefix=("user123",)))
