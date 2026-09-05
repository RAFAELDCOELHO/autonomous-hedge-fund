"""Headline Arena arms: the 2x2 factorial's two agent variants as arena entrants.

Forward-only live leg of the macro-agent hypothesis (issue #3). Each arm is a
TradingAgents graph differing only in whether the Macro Economist analyst is on.
Registration/forecast submission is done by the headlinearena plugin
(https://github.com/headlinearena/headlinearena-agent-plugin); this module only
defines the arms so both sides run the same code path.

$0 by default: `python scripts/headline_arena_arms.py` just lists the arms.
Building a graph (`--build ARM`) needs ANTHROPIC_API_KEY.
"""

import sys

BASE_ANALYSTS = ["market", "social", "news", "fundamentals"]

ARMS = {
    "macro": {
        "arena_agent": "ahf-tradingagents-macro",
        "selected_analysts": BASE_ANALYSTS + ["macro"],
    },
    "no_macro": {
        "arena_agent": "ahf-tradingagents-no-macro",
        "selected_analysts": BASE_ANALYSTS,
    },
}


def build_graph(arm: str, debug: bool = False):
    """Return a TradingAgentsGraph for `arm`. Imports lazily: needs an API key."""
    from dotenv import load_dotenv
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    load_dotenv()
    config = DEFAULT_CONFIG.copy()
    config["max_debate_rounds"] = 1
    return TradingAgentsGraph(
        debug=debug, config=config, selected_analysts=ARMS[arm]["selected_analysts"]
    )


def _check():
    assert set(ARMS) == {"macro", "no_macro"}
    a, b = ARMS["macro"]["selected_analysts"], ARMS["no_macro"]["selected_analysts"]
    assert set(a) - set(b) == {"macro"} and set(b) <= set(a)
    assert ARMS["macro"]["arena_agent"] != ARMS["no_macro"]["arena_agent"]


if __name__ == "__main__":
    _check()
    if len(sys.argv) == 3 and sys.argv[1] == "--build":
        build_graph(sys.argv[2], debug=True)
        print(f"built graph for arm {sys.argv[2]!r}")
    else:
        for name, arm in ARMS.items():
            print(f"{name:9s} arena_agent={arm['arena_agent']}  analysts={arm['selected_analysts']}")
