import sys
import types
import unittest
from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest.mock import patch


def _load_conditional_logic_class():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "tradingagents"
        / "graph"
        / "conditional_logic.py"
    )
    spec = spec_from_file_location("conditional_logic_module", module_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)

    fake_root = types.ModuleType("tradingagents")
    fake_agents = types.ModuleType("tradingagents.agents")
    fake_utils = types.ModuleType("tradingagents.agents.utils")
    fake_agent_states = types.ModuleType("tradingagents.agents.utils.agent_states")
    fake_agent_states.AgentState = dict

    with patch.dict(
        sys.modules,
        {
            "tradingagents": fake_root,
            "tradingagents.agents": fake_agents,
            "tradingagents.agents.utils": fake_utils,
            "tradingagents.agents.utils.agent_states": fake_agent_states,
        },
    ):
        spec.loader.exec_module(module)
    return module.ConditionalLogic


ConditionalLogic = _load_conditional_logic_class()


@dataclass
class DummyMessage:
    tool_calls: list


def _state_with_message(tool_calls):
    return {"messages": [DummyMessage(tool_calls=tool_calls)]}


def _state_with_messages(*tool_calls_per_message):
    return {
        "messages": [DummyMessage(tool_calls=tool_calls) for tool_calls in tool_calls_per_message]
    }


class ConditionalLogicToolRoutingTests(unittest.TestCase):
    def setUp(self):
        self.logic = ConditionalLogic()

    def test_analysis_steps_route_to_tool_node_when_tool_calls_exist(self):
        state = _state_with_message([{"name": "tool"}])
        expected = {
            "should_continue_market": "tools_market",
            "should_continue_social": "tools_social",
            "should_continue_news": "tools_news",
            "should_continue_fundamentals": "tools_fundamentals",
            "should_continue_macro": "tools_macro",
        }
        for method_name, expected_route in expected.items():
            with self.subTest(method=method_name):
                self.assertEqual(getattr(self.logic, method_name)(state), expected_route)

    def test_analysis_steps_clear_messages_when_no_tool_calls_exist(self):
        state = _state_with_message([])
        expected = {
            "should_continue_market": "Msg Clear Market",
            "should_continue_social": "Msg Clear Social",
            "should_continue_news": "Msg Clear News",
            "should_continue_fundamentals": "Msg Clear Fundamentals",
            "should_continue_macro": "Msg Clear Macro",
        }
        for method_name, expected_route in expected.items():
            with self.subTest(method=method_name):
                self.assertEqual(getattr(self.logic, method_name)(state), expected_route)

    def test_analysis_steps_use_latest_message_for_routing(self):
        state = _state_with_messages([], [{"name": "latest-tool"}])
        self.assertEqual(self.logic.should_continue_market(state), "tools_market")

    def test_analysis_steps_raise_index_error_when_message_list_empty(self):
        with self.assertRaises(IndexError):
            self.logic.should_continue_market({"messages": []})


class ConditionalLogicDebateFlowTests(unittest.TestCase):
    def test_debate_stops_at_round_limit(self):
        logic = ConditionalLogic(max_debate_rounds=2)
        state = {
            "investment_debate_state": {"count": 4, "current_response": "Bull case"},
        }
        self.assertEqual(logic.should_continue_debate(state), "Research Manager")

    def test_debate_alternates_to_bear_after_bull_response(self):
        state = {
            "investment_debate_state": {"count": 1, "current_response": "Bull opening"},
        }
        self.assertEqual(ConditionalLogic().should_continue_debate(state), "Bear Researcher")

    def test_debate_defaults_to_bull_when_latest_response_not_from_bull(self):
        state = {
            "investment_debate_state": {"count": 1, "current_response": "Bear rebuttal"},
        }
        self.assertEqual(ConditionalLogic().should_continue_debate(state), "Bull Researcher")


class ConditionalLogicRiskFlowTests(unittest.TestCase):
    def test_risk_discussion_stops_at_round_limit(self):
        logic = ConditionalLogic(max_risk_discuss_rounds=2)
        state = {"risk_debate_state": {"count": 6, "latest_speaker": "Aggressive Analyst"}}
        self.assertEqual(logic.should_continue_risk_analysis(state), "Portfolio Manager")

    def test_risk_discussion_rotates_from_aggressive_to_conservative(self):
        state = {"risk_debate_state": {"count": 1, "latest_speaker": "Aggressive Analyst"}}
        self.assertEqual(
            ConditionalLogic().should_continue_risk_analysis(state),
            "Conservative Analyst",
        )

    def test_risk_discussion_rotates_from_conservative_to_neutral(self):
        state = {"risk_debate_state": {"count": 1, "latest_speaker": "Conservative Analyst"}}
        self.assertEqual(
            ConditionalLogic().should_continue_risk_analysis(state),
            "Neutral Analyst",
        )

    def test_risk_discussion_defaults_to_aggressive_for_other_speakers(self):
        state = {"risk_debate_state": {"count": 1, "latest_speaker": "Neutral Analyst"}}
        self.assertEqual(
            ConditionalLogic().should_continue_risk_analysis(state),
            "Aggressive Analyst",
        )


if __name__ == "__main__":
    unittest.main()
