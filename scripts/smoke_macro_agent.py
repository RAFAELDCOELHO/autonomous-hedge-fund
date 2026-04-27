"""Smoke test: TradingAgents WITH Macro Economist Agent on AAPL, 1 day.

Compares against test_api_smoke (just API) and main.py (4 default analysts).
This test adds the 5th: macro.
"""

from dotenv import load_dotenv
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

load_dotenv()

config = DEFAULT_CONFIG.copy()
config["max_debate_rounds"] = 1

# Activate Macro Agent — the key difference vs main.py
ta = TradingAgentsGraph(
    debug=True,
    config=config,
    selected_analysts=["market", "social", "news", "fundamentals", "macro"],
)

print("\n=== Running TradingAgents WITH Macro Agent ===")
print("Ticker: AAPL, Date: 2024-01-03")
print("Analysts: market, social, news, fundamentals, MACRO\n")

state, decision = ta.propagate("AAPL", "2024-01-03")

print("\n" + "=" * 60)
print("FINAL DECISION:")
print("=" * 60)
print(decision)

print("\n" + "=" * 60)
print("MACRO REPORT (first 800 chars):")
print("=" * 60)
macro_report = state.get("macro_report", "[NO MACRO REPORT GENERATED]")
if isinstance(macro_report, str):
    print(macro_report[:800])
    print(f"\n[Total macro_report length: {len(macro_report)} chars]")
else:
    print(f"[Unexpected type: {type(macro_report)}]")
    print(macro_report)
