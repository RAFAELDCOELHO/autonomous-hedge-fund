"""P3.12: macro_report should influence only Bull/Bear prompts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableLambda

from tradingagents.agents.managers.research_manager import create_research_manager
from tradingagents.agents.researchers.bear_researcher import create_bear_researcher
from tradingagents.agents.researchers.bull_researcher import create_bull_researcher
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph


BULL_ABSENT_PROMPT_SHA256 = "ab467475e46d012f1c52d75323c16321960b8acebf26b6e05af3c11461717676"
BEAR_ABSENT_PROMPT_SHA256 = "7e387d43925152981b6c96ed49ec08708d3415d0084d7fd0c2acff19bc4df810"
MACRO_TOKEN = "MACRO-REPORT-TOKEN-P3-12"


class _NoMemory:
    def get_memories(self, *_args: Any, **_kwargs: Any) -> list[dict[str, str]]:
        return []


@dataclass
class _CaptureInvokeLLM:
    prompts: list[str] = field(default_factory=list)

    def invoke(self, prompt: str) -> AIMessage:
        self.prompts.append(prompt)
        return AIMessage(content="stub-response", id=f"capture-{len(self.prompts)}")


@dataclass
class _RecordingLLM:
    """LLM stub for full graph execution that records every prompt payload."""

    macro_token: str
    prompts: list[str] = field(default_factory=list)

    def bind_tools(self, _tools: list[Any]) -> RunnableLambda:
        return RunnableLambda(lambda payload: self.invoke(payload))

    def invoke(self, payload: Any) -> AIMessage:
        prompt = self._payload_to_text(payload)
        self.prompts.append(prompt)

        lower_prompt = prompt.lower()
        if "specializing in the brazilian economy" in lower_prompt:
            content = self.macro_token
        elif "extracts the trading decision" in lower_prompt:
            content = "HOLD"
        else:
            content = "stub-response"
        return AIMessage(content=content, id=f"record-{len(self.prompts)}")

    def _payload_to_text(self, payload: Any) -> str:
        messages: list[Any]
        if hasattr(payload, "to_messages"):
            messages = payload.to_messages()
        elif isinstance(payload, list):
            messages = payload
        else:
            messages = [payload]

        rendered: list[str] = []
        for message in messages:
            if isinstance(message, BaseMessage):
                rendered.append(f"[{message.type}] {message.content}")
            elif isinstance(message, tuple) and len(message) == 2:
                rendered.append(f"[{message[0]}] {message[1]}")
            elif isinstance(message, dict):
                rendered.append(f"[{message.get('role', 'unknown')}] {message.get('content', '')}")
            else:
                rendered.append(str(message))
        return "\n".join(rendered)


def _base_debate_state() -> dict[str, Any]:
    return {
        "investment_debate_state": {
            "history": "HIST",
            "bull_history": "BULL_HIST",
            "bear_history": "BEAR_HIST",
            "current_response": "LAST",
            "count": 0,
        },
        "market_report": "MKT",
        "sentiment_report": "SENT",
        "news_report": "NEWS",
        "fundamentals_report": "FUND",
    }


def test_bull_and_bear_macro_prompt_diff_and_absent_golden_hash():
    memory = _NoMemory()

    bull_llm_absent = _CaptureInvokeLLM()
    bull_node = create_bull_researcher(bull_llm_absent, memory)
    bull_node(_base_debate_state())
    bull_absent_prompt = bull_llm_absent.prompts[-1]

    bull_llm_present = _CaptureInvokeLLM()
    bull_node_with_macro = create_bull_researcher(bull_llm_present, memory)
    state_with_macro = _base_debate_state()
    state_with_macro["macro_report"] = MACRO_TOKEN
    bull_node_with_macro(state_with_macro)
    bull_present_prompt = bull_llm_present.prompts[-1]

    assert bull_present_prompt != bull_absent_prompt
    assert MACRO_TOKEN in bull_present_prompt
    assert MACRO_TOKEN not in bull_absent_prompt
    assert (
        hashlib.sha256(bull_absent_prompt.encode("utf-8")).hexdigest()
        == BULL_ABSENT_PROMPT_SHA256
    )
    for empty_macro in ("", "   ", None):
        llm = _CaptureInvokeLLM()
        node = create_bull_researcher(llm, memory)
        state_empty = _base_debate_state()
        state_empty["macro_report"] = empty_macro
        node(state_empty)
        assert llm.prompts[-1] == bull_absent_prompt

    bear_llm_absent = _CaptureInvokeLLM()
    bear_node = create_bear_researcher(bear_llm_absent, memory)
    bear_node(_base_debate_state())
    bear_absent_prompt = bear_llm_absent.prompts[-1]

    bear_llm_present = _CaptureInvokeLLM()
    bear_node_with_macro = create_bear_researcher(bear_llm_present, memory)
    state_with_macro = _base_debate_state()
    state_with_macro["macro_report"] = MACRO_TOKEN
    bear_node_with_macro(state_with_macro)
    bear_present_prompt = bear_llm_present.prompts[-1]

    assert bear_present_prompt != bear_absent_prompt
    assert MACRO_TOKEN in bear_present_prompt
    assert MACRO_TOKEN not in bear_absent_prompt
    assert (
        hashlib.sha256(bear_absent_prompt.encode("utf-8")).hexdigest()
        == BEAR_ABSENT_PROMPT_SHA256
    )
    for empty_macro in ("", "   ", None):
        llm = _CaptureInvokeLLM()
        node = create_bear_researcher(llm, memory)
        state_empty = _base_debate_state()
        state_empty["macro_report"] = empty_macro
        node(state_empty)
        assert llm.prompts[-1] == bear_absent_prompt


def test_research_manager_prompt_excludes_macro_report():
    llm = _CaptureInvokeLLM()
    node = create_research_manager(llm, _NoMemory())
    state = _base_debate_state()
    state["company_of_interest"] = "PETR4.SA"
    state["macro_report"] = MACRO_TOKEN
    node(state)

    assert llm.prompts, "research_manager should call llm.invoke once"
    assert MACRO_TOKEN not in llm.prompts[-1]


def test_real_graph_propagation_macro_visibility_by_arm(monkeypatch, tmp_path: Path):
    def run_arm(selected_analysts: list[str]) -> list[str]:
        recording_llm = _RecordingLLM(macro_token=MACRO_TOKEN)

        class _FakeClient:
            def get_llm(self) -> _RecordingLLM:
                return recording_llm

        monkeypatch.setattr(
            "tradingagents.graph.trading_graph.create_llm_client",
            lambda **_kwargs: _FakeClient(),
        )

        config = DEFAULT_CONFIG.copy()
        config["results_dir"] = str(tmp_path / "results")
        config["data_cache_dir"] = str(tmp_path / "cache")
        config["max_recur_limit"] = 60

        graph = TradingAgentsGraph(
            selected_analysts=selected_analysts,
            debug=False,
            config=config,
        )
        graph.propagate("PETR4.SA", "2024-01-10")
        return recording_llm.prompts

    absent_prompts = run_arm(["market", "social", "news", "fundamentals"])
    assert absent_prompts, "absent arm should still produce LLM prompts"
    assert all(MACRO_TOKEN not in prompt for prompt in absent_prompts)
    absent_bull_prompts = [
        prompt
        for prompt in absent_prompts
        if "You are a Bull Analyst advocating for investing in the stock." in prompt
    ]
    absent_bear_prompts = [
        prompt
        for prompt in absent_prompts
        if "You are a Bear Analyst making the case against investing in the stock." in prompt
    ]
    assert absent_bull_prompts and absent_bear_prompts

    present_prompts = run_arm(["market", "social", "news", "fundamentals", "macro"])
    assert present_prompts, "present arm should produce LLM prompts"

    bull_prompts = [
        prompt
        for prompt in present_prompts
        if "You are a Bull Analyst advocating for investing in the stock." in prompt
    ]
    bear_prompts = [
        prompt
        for prompt in present_prompts
        if "You are a Bear Analyst making the case against investing in the stock." in prompt
    ]
    assert bull_prompts and bear_prompts
    assert all(MACRO_TOKEN in prompt for prompt in bull_prompts)
    assert all(MACRO_TOKEN in prompt for prompt in bear_prompts)
    assert bull_prompts[0] != absent_bull_prompts[0]
    assert bear_prompts[0] != absent_bear_prompts[0]

    research_manager_prompts = [
        prompt
        for prompt in present_prompts
        if "As the portfolio manager and debate facilitator" in prompt
    ]
    trader_prompts = [
        prompt
        for prompt in present_prompts
        if "You are a trading agent analyzing market data to make investment decisions." in prompt
    ]
    risk_prompts = [
        prompt
        for prompt in present_prompts
        if (
            "As the Aggressive Risk Analyst" in prompt
            or "As the Conservative Risk Analyst" in prompt
            or "As the Neutral Risk Analyst" in prompt
        )
    ]
    assert research_manager_prompts and trader_prompts and risk_prompts
    assert all(MACRO_TOKEN not in prompt for prompt in research_manager_prompts)
    assert all(MACRO_TOKEN not in prompt for prompt in trader_prompts)
    assert all(MACRO_TOKEN not in prompt for prompt in risk_prompts)
