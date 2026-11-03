"""8.9 governance: "forget me" deletes long-term items and the user's threads."""
from s0_setup import *


def forget_user(user_id: str) -> dict:
    """Delete everything stored about one user: store items and conversation threads."""
    items = 0
    for namespace in store.list_namespaces(prefix=(user_id,)):
        for item in store.search(namespace, limit=1000):
            store.delete(namespace, item.key)
            items += 1
    # threads were tagged with metadata={"user_id": ...} when we called the agent
    threads = {c.config["configurable"]["thread_id"] for c in checkpointer.list(None, filter={"user_id": user_id})}
    for thread_id in threads:
        checkpointer.delete_thread(thread_id)
    return {"items": items, "threads": sorted(threads)}


if __name__ == "__main__":
    print(forget_user("user123"))
    print("namespaces left:", store.list_namespaces(prefix=("user123",)))
    print("threads left:", len(list(checkpointer.list(None, filter={"user_id": "user123"}))))
    with pool.connection() as conn:
        print(conn.execute("SELECT thread_id, count(*) AS n FROM checkpoints GROUP BY thread_id ORDER BY thread_id").fetchall())
