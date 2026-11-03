"""FX MCP server: live exchange rates exposed as MCP tools, a resource and a prompt."""
import sys
from typing import Annotated

import httpx
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

mcp = FastMCP("fx")

Currency = Annotated[str, Field(description='ISO 4217 currency code, e.g. "USD" or "CNY"')]


class Quote(BaseModel):
    base: str
    target: str
    rate: float
    source: str
    updated: str


class Conversion(Quote):
    amount: float
    converted: float


SOURCES = [
    ("open.er-api.com", "https://open.er-api.com/v6/latest/{base}"),
    ("frankfurter", "https://api.frankfurter.dev/v1/latest?base={base}"),
]


def fetch_rates(base: str) -> tuple[dict, str, str]:
    """Try each source in order; return (rates, source name, update time)."""
    errors = []
    for name, url in SOURCES:
        try:
            r = httpx.get(url.format(base=base.upper()), timeout=10)
            r.raise_for_status()
            data = r.json()
            if data.get("rates"):
                return data["rates"], name, data.get("time_last_update_utc") or data.get("date")
            errors.append(f"{name}: no rates for {base}")
        except Exception as e:
            errors.append(f"{name}: {e}")
    raise ValueError("All FX sources failed: " + "; ".join(errors))


@mcp.tool()
def get_rate(base: Currency, target: Currency) -> Quote:
    """Get the latest exchange rate from one currency to another."""
    rates, source, updated = fetch_rates(base)
    target = target.upper()
    if target not in rates:
        raise ValueError(f"Unknown currency code: {target}. Read the fx://currencies resource for valid codes.")
    print(f"[fx] get_rate {base}->{target} via {source}", file=sys.stderr)   # stdout is the protocol channel
    return Quote(base=base.upper(), target=target, rate=rates[target], source=source, updated=updated)


@mcp.tool()
def convert(amount: Annotated[float, Field(description="Amount of money in the base currency", gt=0)],
            base: Currency, target: Currency) -> Conversion:
    """Convert an amount of money from one currency to another at the latest rate."""
    quote = get_rate(base, target)
    return Conversion(**quote.model_dump(), amount=amount, converted=round(amount * quote.rate, 2))


@mcp.resource("fx://currencies")
def currencies() -> str:
    """All currency codes this server can quote."""
    rates, _, _ = fetch_rates("USD")
    return ", ".join(sorted(rates))


@mcp.prompt()
def fx_report(base: str, targets: str = "CNY,EUR,JPY") -> str:
    """A ready-made request for a short FX report."""
    return (f"Write a short FX report for {base} against {targets}. "
            "Call get_rate for every pair, state the source and update time, and keep it under 100 words.")


if __name__ == "__main__":
    print("[fx] server process started", file=sys.stderr)
    mcp.run()   # stdio transport by default
