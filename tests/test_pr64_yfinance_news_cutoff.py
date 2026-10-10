"""PR #64 follow-up: strict UTC cutoff for the yfinance news tools.

The tools receive the data cutoff D-1 (``end_date`` for ticker news,
``curr_date`` for global news, i.e. the graph's ``trade_date``). The decision
date D is the next calendar day, and an article is shown only if it was
published strictly before D 00:00:00 UTC:

1. strict ``pub < D 00:00 UTC``; aware timestamps are CONVERTED to UTC;
2. flat items are filtered by ``providerPublishTime`` (epoch s, UTC), and
   missing / unparseable values are dropped;
3. ``content`` items without a verifiable ``pubDate`` are dropped;
4. global news applies ``[:limit]`` AFTER the date filter;
5. the ``get_global_news`` tool caps a model-supplied ``curr_date`` at the
   injected ``trade_date`` (like ``get_news`` caps ``end_date``).

Expected to FAIL on 5bf7fe3 and PASS once the fix lands. Offline: yfinance
Ticker/Search are faked (tests/_news_fakes.py). Assumed names, kept here:
``get_news_yfinance(ticker, start_date, end_date)``,
``get_global_news_yfinance(curr_date, look_back_days, limit)`` (unchanged
signatures) and the tool ``get_global_news`` gaining
``trade_date: Annotated[Optional[str], InjectedState("trade_date")] = None``.
"""

from __future__ import annotations

import time

import pytest

from _news_fakes import FakeNews, content_item, epoch, flat_item, set_process_tz

CUTOFF = "2024-01-03"   # D-1 = date passed to the tools (trade_date)
D = "2024-01-04"        # decision day; cutoff instant = 2024-01-04T00:00:00Z
START = "2023-12-27"    # ticker news start_date (all valid items are later)
TOOLS = ["ticker", "global"]


@pytest.fixture(autouse=True)
def _local_tz_not_utc(monkeypatch):
    # Run with a non-UTC process timezone (UTC-3, like the box / Brazil), so
    # naive local-time conversions cannot pass by accident.
    set_process_tz(monkeypatch, "Etc/GMT+3")
    yield
    monkeypatch.undo()
    time.tzset()


def _fetch(monkeypatch, tool, items, limit=50):
    from tradingagents.dataflows import yfinance_news as yfn

    FakeNews(items).install(monkeypatch)
    if tool == "ticker":
        out = yfn.get_news_yfinance("AAPL", START, CUTOFF)
    else:
        out = yfn.get_global_news_yfinance(CUTOFF, 7, limit)
    assert "Error fetching" not in out, out
    return out


def _assert_shown(out, *tokens):
    for t in tokens:
        assert t in out, f"{t} should be shown:\n{out}"


def _assert_hidden(out, *tokens):
    for t in tokens:
        assert t not in out, f"{t} leaked:\n{out}"


# 1. Boundary --------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["content", "flat"])
@pytest.mark.parametrize("tool", TOOLS)
def test_item_at_d_midnight_utc_hidden_and_last_second_of_d_minus_1_shown(monkeypatch, tool, fmt):
    if fmt == "content":
        items = [
            content_item("TOK_C_AT_D_0000Z", f"{D}T00:00:00Z"),
            content_item("TOK_C_D1_235959Z", f"{CUTOFF}T23:59:59Z"),
        ]
    else:
        items = [
            flat_item("TOK_F_AT_D_0000Z", epoch(f"{D}T00:00:00")),
            flat_item("TOK_F_D1_235959Z", epoch(f"{CUTOFF}T23:59:59")),
        ]
    out = _fetch(monkeypatch, tool, items)
    _assert_hidden(out, f"TOK_{fmt[0].upper()}_AT_D_0000Z")
    _assert_shown(out, f"TOK_{fmt[0].upper()}_D1_235959Z")


# 2. Offsets are converted to UTC, not stripped ----------------------------


@pytest.mark.parametrize("tool", TOOLS)
def test_non_utc_offsets_are_converted_to_utc(monkeypatch, tool):
    items = [
        content_item("TOK_TZ_M0300_IS_D_0030Z", f"{CUTOFF}T21:30:00-03:00"),     # = D 00:30Z
        content_item("TOK_TZ_M0500_IS_D_0430Z", f"{CUTOFF}T23:30:00-05:00"),     # = D 04:30Z
        content_item("TOK_TZ_M0300_IS_D1_235959Z", f"{CUTOFF}T20:59:59-03:00"),  # = D-1 23:59:59Z
        content_item("TOK_TZ_P0200_IS_D1_2300Z", f"{D}T01:00:00+02:00"),         # = D-1 23:00Z
    ]
    out = _fetch(monkeypatch, tool, items)
    _assert_hidden(out, "TOK_TZ_M0300_IS_D_0030Z", "TOK_TZ_M0500_IS_D_0430Z")
    _assert_shown(out, "TOK_TZ_M0300_IS_D1_235959Z", "TOK_TZ_P0200_IS_D1_2300Z")


@pytest.mark.parametrize("tzname", ["Etc/GMT+3", "Asia/Tokyo"])
@pytest.mark.parametrize("tool", TOOLS)
def test_cutoff_does_not_depend_on_process_timezone(monkeypatch, tool, tzname):
    """Epoch seconds are UTC (no naive fromtimestamp), and an offset-less
    pubDate is read as UTC (no astimezone() on a naive value)."""
    set_process_tz(monkeypatch, tzname)
    items = [
        flat_item("TOK_EPOCH_D_0100Z", epoch(f"{D}T01:00:00")),
        flat_item("TOK_EPOCH_D1_2200Z", epoch(f"{CUTOFF}T22:00:00")),
        content_item("TOK_NAIVE_D_0030", f"{D}T00:30:00"),
        content_item("TOK_NAIVE_D1_2200", f"{CUTOFF}T22:00:00"),
    ]
    out = _fetch(monkeypatch, tool, items)
    _assert_hidden(out, "TOK_EPOCH_D_0100Z", "TOK_NAIVE_D_0030")
    _assert_shown(out, "TOK_EPOCH_D1_2200Z", "TOK_NAIVE_D1_2200")


# 3. Items without a verifiable date are dropped ---------------------------


@pytest.mark.parametrize("tool", TOOLS)
def test_items_without_verifiable_date_are_dropped(monkeypatch, tool):
    items = [
        content_item("TOK_NODATE_C_MISSING"),
        content_item("TOK_NODATE_C_EMPTY", ""),
        content_item("TOK_NODATE_C_NONE", None),
        content_item("TOK_NODATE_C_GARBAGE", "yesterday-ish"),
        flat_item("TOK_NODATE_F_MISSING"),
        flat_item("TOK_NODATE_F_NONE", None),
        flat_item("TOK_NODATE_F_GARBAGE", "not-a-time"),
        content_item("TOK_VALID_CONTROL_C", f"{CUTOFF}T12:00:00Z"),
        flat_item("TOK_VALID_CONTROL_F", epoch(f"{CUTOFF}T12:00:00")),
    ]
    out = _fetch(monkeypatch, tool, items)
    _assert_hidden(out, *[i["id"] if "id" in i else i["uuid"] for i in items if "NODATE" in str(i)])
    _assert_shown(out, "TOK_VALID_CONTROL_C", "TOK_VALID_CONTROL_F")


# 4. Sentinels after D -----------------------------------------------------


@pytest.mark.parametrize("tool", TOOLS)
def test_future_sentinels_never_appear(monkeypatch, tool):
    items = [
        content_item("TOK_FUTURE_C_D_0930Z_X7Q", f"{D}T09:30:00Z"),
        content_item("TOK_FUTURE_C_2026_X7Q", "2026-10-07T12:00:00Z"),
        flat_item("TOK_FUTURE_F_D_1430Z_X7Q", epoch(f"{D}T14:30:00")),
        flat_item("TOK_FUTURE_F_2026_X7Q", epoch("2026-10-07T12:00:00")),
        content_item("TOK_PAST_CONTROL_C", "2024-01-02T15:00:00Z"),
        flat_item("TOK_PAST_CONTROL_F", epoch("2024-01-02T16:00:00")),
    ]
    out = _fetch(monkeypatch, tool, items)
    assert "X7Q" not in out, out
    _assert_shown(out, "TOK_PAST_CONTROL_C", "TOK_PAST_CONTROL_F")


# 5. Global news: limit counts kept items only -----------------------------


def test_global_limit_is_applied_after_the_date_filter(monkeypatch):
    future = [content_item(f"TOK_LIM_FUT_C{i}", f"2026-10-0{i + 1}T12:00:00Z") for i in range(3)]
    future += [flat_item(f"TOK_LIM_FUT_F{i}", epoch(f"2026-10-0{i + 4}T12:00:00")) for i in range(3)]
    old = [content_item(f"TOK_LIM_OLD_C{i}", f"2024-01-0{i + 1}T10:00:00Z") for i in range(3)]
    old += [flat_item(f"TOK_LIM_OLD_F{i}", epoch(f"2024-01-0{i + 1}T11:00:00")) for i in range(2)]
    out = _fetch(monkeypatch, "global", future + old, limit=3)
    _assert_hidden(out, *[f"TOK_LIM_FUT_C{i}" for i in range(3)], *[f"TOK_LIM_FUT_F{i}" for i in range(3)])
    shown = [line for line in out.splitlines() if line.startswith("### ")]
    assert len(shown) == 3, out
    assert all("TOK_LIM_OLD_" in line for line in shown), shown


# 6. get_global_news tool: model curr_date capped at trade_date ------------


def _invoke_global_tool(model_curr_date, trade_date=CUTOFF):
    from langchain_core.messages import AIMessage
    from langgraph.prebuilt import ToolNode

    from tradingagents.agents.utils import news_data_tools

    call = {"name": "get_global_news", "id": "g1",
            "args": {"curr_date": model_curr_date, "look_back_days": 7, "limit": 5}}
    result = ToolNode([news_data_tools.get_global_news]).invoke(
        {"messages": [AIMessage(content="", tool_calls=[call])], "trade_date": trade_date}
    )
    return result["messages"][-1].content


@pytest.mark.parametrize("model_curr_date", ["2024-01-15", D])
def test_global_news_tool_caps_model_curr_date_at_trade_date(monkeypatch, model_curr_date):
    from tradingagents.agents.utils import news_data_tools

    assert "trade_date" not in news_data_tools.get_global_news.tool_call_schema.model_json_schema()["properties"]
    FakeNews([
        content_item("TOK_GCAP_AT_D_0000Z", f"{D}T00:00:00Z"),
        content_item("TOK_GCAP_JAN10", "2024-01-10T12:00:00Z"),
        content_item("TOK_GCAP_2026", "2026-10-07T12:00:00Z"),
        content_item("TOK_GCAP_CONTROL", f"{CUTOFF}T12:00:00Z"),
    ]).install(monkeypatch)

    out = _invoke_global_tool(model_curr_date)

    _assert_hidden(out, "TOK_GCAP_AT_D_0000Z", "TOK_GCAP_JAN10", "TOK_GCAP_2026")
    _assert_shown(out, "TOK_GCAP_CONTROL")
    header = next(line for line in out.splitlines() if line.startswith("## "))
    assert header.endswith(f"to {CUTOFF}:"), header
    assert model_curr_date not in header


@pytest.mark.parametrize(("model_curr_date", "expected"), [("2024-01-15", CUTOFF), (D, CUTOFF)])
def test_global_news_tool_routes_capped_curr_date(monkeypatch, model_curr_date, expected):
    """Mirrors test_p49_style_get_news_end_date_capped_at_graph_trade_date."""
    from tradingagents.agents.utils import news_data_tools

    seen = []
    monkeypatch.setattr(news_data_tools, "route_to_vendor", lambda *args: seen.append(args) or "news")
    _invoke_global_tool(model_curr_date)
    _invoke_global_tool("2024-01-02")  # below the ceiling: unchanged
    assert seen == [("get_global_news", expected, 7, 5), ("get_global_news", "2024-01-02", 7, 5)]
