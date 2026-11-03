"""Regression gate for CI: fail the build if tool calling gets worse. Run with: pytest -q test_agent_eval.py"""
import os

import pytest

from e1_bfcl import CASES, HARD_CASES, HARD_TOOLS, POOR_TOOLS, run_suite

ALL = CASES + HARD_CASES


@pytest.fixture(scope="module")
def rows():
    tools = POOR_TOOLS if os.environ.get("EVAL_TOOLS") == "poor" else HARD_TOOLS   # simulate a bad change to tool docs
    return run_suite("deepseek-flash", tools, temperature=0.0, cases=ALL)


def test_overall_accuracy(rows):
    accuracy = sum(r["ok"] for r in rows) / len(rows)
    assert accuracy >= 0.95, f"tool-call accuracy dropped to {accuracy:.2f}"


@pytest.mark.parametrize("category", ["irrelevance", "tricky irrelevance", "missing info"])
def test_never_calls_when_it_should_not(rows, category):
    failed = [r["question"] for r in rows if r["category"] == category and not r["ok"]]
    assert not failed, f"called a tool when it should not: {failed}"


def test_latency_budget(rows):
    average = sum(r["seconds"] for r in rows) / len(rows)
    assert average < 3.0, f"average latency {average:.1f}s exceeds the 3s budget"
