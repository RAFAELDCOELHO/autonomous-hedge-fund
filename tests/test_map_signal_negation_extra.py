"""map_signal: negation window spans commas and a few words, never a sentence."""

from __future__ import annotations

import pytest

from tradingagents.backtest.agent_integration import map_signal


@pytest.mark.parametrize(
    "raw",
    [
        "never, ever BUY",
        "Never, ever BUY this name.",
        "do not, repeat, BUY",
        "Do not, I repeat, BUY",
        "we do not recommend that you BUY",
        "We do NOT think you should BUY",
        "never, under any circumstances, SELL",
    ],
)
def test_widened_negation_maps_to_hold(raw):
    assert map_signal(raw) == "HOLD"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("BUY, never SELL", "BUY"),
        ("not only BUY", "BUY"),
        ("Do not panic. BUY.", "BUY"),
        ("Do not panic! SELL", "SELL"),
        ("Never mind; BUY", "BUY"),
        ("Do not wait\nBUY", "BUY"),
        ("No doubt, BUY", "BUY"),
        ("never, ever BUY. SELL", "SELL"),
        ("We do not recommend that you BUY; HOLD instead", "HOLD"),
        ("not SELL, BUY", "BUY"),
    ],
)
def test_widened_window_keeps_non_negated_labels(raw, expected):
    assert map_signal(raw) == expected


def test_window_is_bounded():
    """Negation more than five words before the label does not reach it."""
    assert map_signal("Do not one two three four five six BUY") == "BUY"
