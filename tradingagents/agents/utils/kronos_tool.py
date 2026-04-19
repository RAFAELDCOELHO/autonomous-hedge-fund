"""LangChain tool wrapping Kronos forecast for the market_analyst agent."""

from typing import Annotated

from langchain_core.tools import tool

from tradingagents.dataflows.kronos_analyst import get_kronos_signal
from tradingagents.dataflows.stockstats_utils import load_ohlcv


@tool
def get_kronos_forecast(
    symbol: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "current trading date, YYYY-mm-dd"],
    pred_len: Annotated[int, "forecast horizon in business days"] = 5,
) -> str:
    """Obtain a Kronos short-horizon directional forecast for the ticker.

    Call this once per analysis as a complementary input to classical
    indicators. Kronos is a time-series foundation model; its output is a
    directional signal (BUY/HOLD/SELL) with an expected return. If the
    model is unavailable for any reason, the tool returns a message
    saying so and the analyst should proceed with indicators only.
    """
    df = load_ohlcv(symbol, curr_date)
    result = get_kronos_signal(df, symbol, pred_len=pred_len)
    if result is None:
        return "Kronos model unavailable for this run."
    return (
        f"Kronos forecast ({result['pred_len']}d, device={result['device']}): "
        f"{result['signal']} — expected return {result['forecast_return_pct']}% "
        f"(predicted close {result['predicted_close']} vs current {result['current_close']})."
    )
