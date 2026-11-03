"""8.19 part 2: answer-level eval. Groundedness on in-scope questions, refusals on out-of-scope ones."""
from r3_agents import *
from r4_eval import EVAL_SET


class Refusal(BaseModel):
    refused: bool = Field(description="True if the answer says the information was not found or is not covered")


refusal_judge = model.with_structured_output(Refusal, method="function_calling")

OUT_OF_SCOPE = ["What is the capital of Australia?", "谁赢得了 2022 年世界杯？", "How do I bake sourdough bread?"]


def judge_grounded(docs: list[dict], answer: str) -> bool:
    return grounding_judge.invoke(f"Passages:\n{format_passages(docs)}\n\nAnswer:\n{answer}").grounded


def run_two_step(question: str) -> dict:
    docs = retrieve(question)
    answer = generate(question, docs)
    return {"answer": answer, "grounded": judge_grounded(docs, answer)}


def run_corrective(question: str) -> dict:
    state = corrective_rag.invoke({"question": question, "queries": [question]})
    return {"answer": state["answer"], "grounded": state["grounded"], "attempts": state["attempts"]}


if __name__ == "__main__":
    in_scope = [item["q"] for item in EVAL_SET]
    for name, run in [("two-step", run_two_step), ("corrective", run_corrective)]:
        print(f"===== {name}")
        grounded = [run(q)["grounded"] for q in in_scope]
        refused = []
        for q in OUT_OF_SCOPE:
            out = run(q)
            refused.append(refusal_judge.invoke(f"Answer:\n{out['answer']}").refused)
            print(f"  OOS {q!r}: refused={refused[-1]} | {out['answer'][:80]!r}")
        print(f"{name}: grounded {sum(grounded)}/{len(grounded)}, refused out-of-scope {sum(refused)}/{len(refused)}")
