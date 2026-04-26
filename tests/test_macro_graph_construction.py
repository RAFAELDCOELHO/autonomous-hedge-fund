"""Validates TradingAgentsGraph constructs WITH macro analyst (no API call)."""

from dotenv import load_dotenv
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

load_dotenv()

config = DEFAULT_CONFIG.copy()
config["max_debate_rounds"] = 1

print("Constructing TradingAgentsGraph with macro analyst...")
ta = TradingAgentsGraph(
    debug=False,
    config=config,
    selected_analysts=["market", "social", "news", "fundamentals", "macro"],
)

assert "macro" in ta.tool_nodes, f"macro tool node missing! Keys: {list(ta.tool_nodes.keys())}"

print("✅ Graph constructed successfully")
print(f"✅ Tool nodes registered: {list(ta.tool_nodes.keys())}")
print(f"✅ Macro tool count: {len(ta.tool_nodes['macro'].tools_by_name)}")
print(f"✅ Macro tools: {list(ta.tool_nodes['macro'].tools_by_name.keys())}")
