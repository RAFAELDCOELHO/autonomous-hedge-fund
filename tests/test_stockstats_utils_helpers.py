import pandas as pd

from tradingagents.dataflows.stockstats_utils import filter_financials_by_date


def test_filter_financials_by_date_keeps_only_columns_on_or_before_cutoff():
    data = pd.DataFrame(
        {
            "2024-12-31": [10.0],
            "2025-03-31": [11.0],
            "2025-06-30": [12.0],
        },
        index=["Revenue"],
    )

    filtered = filter_financials_by_date(data, "2025-03-31")

    assert list(filtered.columns) == ["2024-12-31", "2025-03-31"]
    assert filtered.loc["Revenue", "2024-12-31"] == 10.0
    assert filtered.loc["Revenue", "2025-03-31"] == 11.0


def test_filter_financials_by_date_returns_input_when_cutoff_missing():
    data = pd.DataFrame(
        {
            "2024-12-31": [10.0],
            "2025-03-31": [11.0],
        },
        index=["Revenue"],
    )

    result = filter_financials_by_date(data, "")

    assert result is data


def test_filter_financials_by_date_returns_input_when_dataframe_empty():
    empty = pd.DataFrame()

    result = filter_financials_by_date(empty, "2025-03-31")

    assert result is empty
