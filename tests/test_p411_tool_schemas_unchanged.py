"""P4.11 B1: the LLM-visible tool schemas are byte-for-byte those of e5ea062.

The trade_date cap is injected from graph state (InjectedState), so it must not
appear in what the model sees: name, description and the tool-call JSON schema of
every agent tool are compared with a snapshot generated at e5ea062
(tests/fixtures/tool_schemas_e5ea062.json).
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
from langchain_core.tools import BaseTool

SNAPSHOT = json.loads(
    (Path(__file__).parent / "fixtures" / "tool_schemas_e5ea062.json").read_text()
)
MODULES = ["core_stock_tools", "technical_indicators_tools", "fundamental_data_tools",
           "news_data_tools", "fingpt_tool", "kronos_tool", "macro_tools"]


def _current_tools() -> dict[str, BaseTool]:
    tools = {}
    for m in MODULES:
        mod = importlib.import_module(f"tradingagents.agents.utils.{m}")
        for obj in vars(mod).values():
            if isinstance(obj, BaseTool) and getattr(obj, "func", None) is not None \
                    and obj.func.__module__ == mod.__name__:
                tools[obj.name] = obj
    return tools


def test_same_tool_set_as_e5ea062():
    assert sorted(_current_tools()) == sorted(SNAPSHOT)


@pytest.mark.parametrize("name", sorted(SNAPSHOT))
def test_llm_visible_schema_and_description_unchanged(name):
    tool = _current_tools()[name]
    schema = tool.tool_call_schema.model_json_schema()
    assert "trade_date" not in schema.get("properties", {})
    assert schema == SNAPSHOT[name]["schema"]
    assert tool.description == SNAPSHOT[name]["description"]
