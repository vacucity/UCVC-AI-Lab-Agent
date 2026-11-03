"""12.2 BFCL-style tool-call evaluation: AST matching of the calls a model produces (nothing is executed)."""
import json
import os
import sys
import time
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import Field

load_dotenv()                                    # finds the .env in the repo root

MODELS = {
    "deepseek-flash": lambda t: ChatOpenAI(model=os.getenv("LLM_MODEL_ID", "deepseek-flash"), api_key=os.getenv("LLM_API_KEY"),
                                           base_url=os.getenv("LLM_BASE_URL"), temperature=t, timeout=60, max_retries=2,
                                           extra_body={"thinking": {"type": "disabled"}}),
    "qwen2.5:7b (local)": lambda t: ChatOpenAI(model="qwen2.5:7b", api_key="ollama", base_url="http://localhost:11434/v1",
                                               temperature=t, timeout=120, max_retries=1),
}

Currency = Annotated[str, Field(description='ISO 4217 currency code, e.g. "USD", "CNY", "EUR"')]


# ---------- the tools under test (well described) ----------
@tool
def get_rate(base: Currency, target: Currency) -> str:
    """Get the latest exchange rate from one currency to another. Use it when the user wants a rate, not a conversion."""
    return ""


@tool
def convert(amount: Annotated[float, Field(description="Amount of money in the base currency, as a plain number")],
            base: Currency, target: Currency) -> str:
    """Convert an amount of money from one currency to another at the latest rate."""
    return ""


@tool
def search_docs(query: Annotated[str, Field(description="Focused English search query")],
                chapter: Annotated[int | None, Field(description="Optional chapter number (1-16)")] = None) -> str:
    """Search the Hello-Agents tutorial, an English book about building AI agents."""
    return ""


@tool
def get_weather(city: Annotated[str, Field(description="City name in English, e.g. 'Beijing'")],
                unit: Annotated[str, Field(description='"celsius" or "fahrenheit"')] = "celsius") -> str:
    """Get today's weather for a city."""
    return ""


@tool
def get_historical_rate(base: Currency, target: Currency,
                        date: Annotated[str, Field(description="Date as YYYY-MM-DD")]) -> str:
    """Get the exchange rate on a past date."""
    return ""


@tool
def convert_crypto(amount: Annotated[float, Field(description="Amount of the cryptocurrency")],
                   coin: Annotated[str, Field(description='Crypto ticker, e.g. "BTC"')], fiat: Currency) -> str:
    """Convert a cryptocurrency amount into a fiat currency."""
    return ""


@tool
def get_stock_price(symbol: Annotated[str, Field(description='Stock ticker, e.g. "AAPL"')]) -> str:
    """Get the latest stock price."""
    return ""


@tool
def get_forecast(city: Annotated[str, Field(description="City name in English")],
                 days_ahead: Annotated[int, Field(description="How many days ahead, 1-7")]) -> str:
    """Get the weather forecast for the coming days (not today)."""
    return ""


@tool
def search_web(query: Annotated[str, Field(description="Web search query")]) -> str:
    """Search the public web."""
    return ""


@tool
def get_air_quality(city: Annotated[str, Field(description="City name in English")]) -> str:
    """Get today's air quality index for a city."""
    return ""


GOOD_TOOLS = [get_rate, convert, search_docs, get_weather]
HARD_TOOLS = GOOD_TOOLS + [get_historical_rate, convert_crypto, get_stock_price, get_forecast, search_web, get_air_quality]


# ---------- the same tools, poorly described (ablation) ----------
@tool
def get_rate_v(base: str, target: str) -> str:
    """Rate."""
    return ""


@tool
def convert_v(amount: str, base: str, target: str) -> str:
    """Convert."""
    return ""


@tool
def search_docs_v(query: str, chapter: str = "") -> str:
    """Search."""
    return ""


@tool
def get_weather_v(city: str, unit: str = "") -> str:
    """Weather."""
    return ""


@tool
def get_historical_rate_v(base: str, target: str, date: str) -> str:
    """Historical."""
    return ""


@tool
def convert_crypto_v(amount: str, coin: str, fiat: str) -> str:
    """Crypto."""
    return ""


@tool
def get_stock_price_v(symbol: str) -> str:
    """Stock."""
    return ""


@tool
def get_forecast_v(city: str, days_ahead: str) -> str:
    """Forecast."""
    return ""


@tool
def search_web_v(query: str) -> str:
    """Web."""
    return ""


@tool
def get_air_quality_v(city: str) -> str:
    """Air."""
    return ""


for t in [get_rate_v, convert_v, search_docs_v, get_weather_v, get_historical_rate_v, convert_crypto_v,
          get_stock_price_v, get_forecast_v, search_web_v, get_air_quality_v]:
    t.name = t.name.removesuffix("_v")
POOR_TOOLS = [get_rate_v, convert_v, search_docs_v, get_weather_v, get_historical_rate_v, convert_crypto_v,
              get_stock_price_v, get_forecast_v, search_web_v, get_air_quality_v]

# expected calls: strings match case-insensitively; "~word" means the value must contain word
CASES = [
    ("simple", "What's the USD to CNY exchange rate right now?", [("get_rate", {"base": "USD", "target": "CNY"})]),
    ("simple", "Convert 250 euros to Japanese yen.", [("convert", {"amount": 250, "base": "EUR", "target": "JPY"})]),
    ("simple", "How much is 1,200 British pounds in US dollars?", [("convert", {"amount": 1200, "base": "GBP", "target": "USD"})]),
    ("simple", "100 港币能换多少人民币？", [("convert", {"amount": 100, "base": "HKD", "target": "CNY"})]),
    ("simple", "What's the weather in Chengdu in Fahrenheit?", [("get_weather", {"city": "~chengdu", "unit": "fahrenheit"})]),
    ("multiple", "In the Hello-Agents tutorial, what is the GSSC pipeline?", [("search_docs", {"query": "~gssc"})]),
    ("multiple", "Search chapter 8 of the tutorial for HyDE.", [("search_docs", {"query": "~hyde", "chapter": 8})]),
    ("multiple", "I only want the JPY to KRW exchange rate, I'm not converting anything.", [("get_rate", {"base": "JPY", "target": "KRW"})]),
    ("multiple", "Is it raining in Hangzhou today?", [("get_weather", {"city": "~hangzhou"})]),
    ("multiple", "把 3000 日元换算成美元", [("convert", {"amount": 3000, "base": "JPY", "target": "USD"})]),
    ("parallel", "Convert 100 USD to EUR and also to JPY.",
     [("convert", {"amount": 100, "base": "USD", "target": "EUR"}), ("convert", {"amount": 100, "base": "USD", "target": "JPY"})]),
    ("parallel", "What's the weather in Beijing and in Shanghai?",
     [("get_weather", {"city": "~beijing"}), ("get_weather", {"city": "~shanghai"})]),
    ("parallel", "Give me both the USD→CNY and the EUR→CNY rates.",
     [("get_rate", {"base": "USD", "target": "CNY"}), ("get_rate", {"base": "EUR", "target": "CNY"})]),
    ("parallel", "Convert 50 GBP to USD, and tell me the weather in London.",
     [("convert", {"amount": 50, "base": "GBP", "target": "USD"}), ("get_weather", {"city": "~london"})]),
    ("irrelevance", "What is 17 times 23?", []),
    ("irrelevance", "Write a haiku about autumn.", []),
    ("irrelevance", "Who wrote Pride and Prejudice?", []),
    ("irrelevance", "Translate 'good morning' into French.", []),
    ("tricky args", "How many yuan is twenty-five thousand dollars?", [("convert", {"amount": 25000, "base": "USD", "target": "CNY"})]),
    ("tricky args", "Convert 1.5k Canadian dollars to Australian dollars.", [("convert", {"amount": 1500, "base": "CAD", "target": "AUD"})]),
]

HISTORY_FX = [{"role": "user", "content": "Convert 200 USD to EUR."},
              {"role": "assistant", "content": "200 USD is about 180 EUR at today's rate."}]
HISTORY_TRIP = [{"role": "user", "content": "I'm planning a trip to Kyoto next month."},
                {"role": "assistant", "content": "Sounds great! Kyoto is lovely in autumn."}]

HARD_CASES = [
    ("missing info", "Convert 300 to euros.", []),
    ("missing info", "What's the weather like?", []),
    ("distractor", "What was the USD to CNY rate on 2024-01-02?",
     [("get_historical_rate", {"base": "USD", "target": "CNY", "date": "2024-01-02"})]),
    ("distractor", "Will it rain in Beijing tomorrow?", [("get_forecast", {"city": "~beijing", "days_ahead": 1})]),
    ("distractor", "Convert 0.5 BTC to US dollars.", [("convert_crypto", {"amount": 0.5, "coin": "BTC", "fiat": "USD"})]),
    ("distractor", "In the Hello-Agents tutorial, what does NoteTool do?", [("search_docs", {"query": "~notetool"})]),
    ("sequential", "Convert 100 USD to EUR, then convert that EUR amount to JPY.",
     [("convert", {"amount": 100, "base": "USD", "target": "EUR"})]),
    ("multi-turn", HISTORY_FX + [{"role": "user", "content": "And to GBP?"}],
     [("convert", {"amount": 200, "base": "USD", "target": "GBP"})]),
    ("multi-turn", HISTORY_TRIP + [{"role": "user", "content": "What's the weather there today, in Fahrenheit?"}],
     [("get_weather", {"city": "~kyoto", "unit": "fahrenheit"})]),
    ("hard args", "My budget is 2,000 RMB. How much is that in Thai baht?", [("convert", {"amount": 2000, "base": "CNY", "target": "THB"})]),
    ("hard args", "convrt 75 swiss franks to eur", [("convert", {"amount": 75, "base": "CHF", "target": "EUR"})]),
    ("hard args", "Half a million won in US dollars?", [("convert", {"amount": 500000, "base": "KRW", "target": "USD"})]),
    ("hard args", "上海今天华氏多少度？", [("get_weather", {"city": "~shanghai", "unit": "fahrenheit"})]),
    ("tricky irrelevance", "Which currency does Japan use?", []),
    ("tricky irrelevance", "Explain what an exchange rate is, in one sentence.", []),
    ("parallel", "Show today's weather in Paris and the 3-day forecast for Rome.",
     [("get_weather", {"city": "~paris"}), ("get_forecast", {"city": "~rome", "days_ahead": 3})]),
]

SYSTEM = "You are an assistant with tools. Call a tool only when it is needed; otherwise answer directly."


def value_ok(expected, got) -> bool:
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return abs(float(str(got).replace(",", "")) - float(expected)) < 1e-6
        except ValueError:
            return False
    if isinstance(expected, str) and expected.startswith("~"):
        return expected[1:].lower() in str(got).lower()
    return str(expected).lower() == str(got).lower()


def call_ok(expected: tuple, got: dict) -> bool:
    name, args = expected
    return got["name"] == name and all(k in got["args"] and value_ok(v, got["args"][k]) for k, v in args.items())


def calls_match(expected: list, got: list) -> bool:
    """AST match: same number of calls, each expected call matched by a different actual call, any order."""
    if len(expected) != len(got):
        return False
    remaining = list(got)
    for e in expected:
        hit = next((g for g in remaining if call_ok(e, g)), None)
        if hit is None:
            return False
        remaining.remove(hit)
    return True


def run_suite(model_name: str, tools: list, temperature: float = 0.0, cases=None) -> list[dict]:
    llm = MODELS[model_name](temperature).bind_tools(tools)
    rows = []
    for category, question, expected in (cases or CASES):
        t0 = time.perf_counter()
        turns = question if isinstance(question, list) else [{"role": "user", "content": question}]
        msg = llm.invoke([{"role": "system", "content": SYSTEM}] + turns)
        question = turns[-1]["content"]
        got = [{"name": c["name"], "args": c["args"]} for c in msg.tool_calls]
        rows.append({"category": category, "question": question, "ok": calls_match(expected, got),
                     "got": got, "seconds": time.perf_counter() - t0})
    return rows


def summarize(label: str, rows: list[dict]):
    order = [x[0] for x in CASES + HARD_CASES]
    cats = sorted({r["category"] for r in rows}, key=order.index)
    per = {c: f"{sum(r['ok'] for r in rows if r['category'] == c)}/{sum(1 for r in rows if r['category'] == c)}" for c in cats}
    total = sum(r["ok"] for r in rows)
    print(f"{label:<34} overall {total}/{len(rows)} | " + " | ".join(f"{c} {v}" for c, v in per.items())
          + f" | {sum(r['seconds'] for r in rows) / len(rows):.1f}s/case")


if __name__ == "__main__":
    results = {}
    for model_name in MODELS:
        rows = run_suite(model_name, HARD_TOOLS, cases=HARD_CASES)
        summarize(f"{model_name}, 10 tools, hard set", rows)
        results[model_name] = rows
    rows = run_suite("deepseek-flash", POOR_TOOLS, cases=HARD_CASES)
    summarize("deepseek-flash, one-word docs, hard", rows)
    results["deepseek-flash poor docs"] = rows
    for label, rs in results.items():
        for r in rs:
            if not r["ok"]:
                print(f"  FAIL [{label}] {r['category']}: {r['question'][:45]!r} -> {json.dumps(r['got'], ensure_ascii=False)[:150]}")
    json.dump(results, open("e1_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
