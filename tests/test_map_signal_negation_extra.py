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


@pytest.mark.parametrize(
    "raw",
    [
        # Capataz review of PR #89: a negation whose clause closes at a comma
        # does not reach a label in the next clause.
        "No question, BUY",
        "Don't hesitate, BUY",
        "No matter what, BUY",
        "No doubt about it, BUY",
        "Not surprisingly, we BUY",
        "Never been more confident, BUY",
        # probes
        "Nothing to lose, BUY",
        "No surprise here, BUY",
        "Without a doubt, BUY",
        "Not bad at all, BUY",
        "Do not wait, BUY",
        "No reason to wait, BUY now",
        "Never a better time, BUY",
        "Not a sell, BUY",
    ],
)
def test_closed_negation_clause_does_not_reach_next_clause(raw):
    assert map_signal(raw) == "BUY"


@pytest.mark.parametrize(
    "raw",
    [
        "Never, ever, under any circumstances, SELL",
        "Do not, I say, BUY",
        "Never, ever again BUY",
        "do not, repeat, do not BUY",
    ],
)
def test_intensifier_parentheticals_keep_the_negation(raw):
    assert map_signal(raw) == "HOLD"
