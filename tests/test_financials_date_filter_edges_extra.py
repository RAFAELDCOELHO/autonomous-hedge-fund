"""Additional edge-case coverage for the financial statements date filter."""

from __future__ import annotations

import pandas as pd

from tradingagents.dataflows.stockstats_utils import filter_financials_by_date


def test_quarterly_period_exactly_on_cutoff_is_excluded_with_tz_aware_column():
    data = pd.DataFrame([[1.0]], columns=[pd.Timestamp("2024-03-31T00:00:00Z")])

    out_on_cutoff = filter_financials_by_date(
        data,
        "2024-05-15",
        freq="quarterly",
        annual_period_ends=["2023-12-31"],
    )
    out_after_cutoff = filter_financials_by_date(
        data,
        "2024-05-16",
        freq="quarterly",
        annual_period_ends=["2023-12-31"],
    )

    assert list(out_on_cutoff.columns) == []
    assert len(out_after_cutoff.columns) == 1


def test_non_date_columns_fail_closed_to_empty_frame():
    data = pd.DataFrame([[100.0, 200.0]], columns=["Assets", "Liabilities"])

    out = filter_financials_by_date(
        data,
        "2024-05-16",
        freq="quarterly",
        annual_period_ends=["2023-12-31"],
    )

    assert list(out.columns) == []
    assert out.shape[0] == data.shape[0]


def test_empty_financials_frame_is_returned_unchanged():
    data = pd.DataFrame()

    out = filter_financials_by_date(
        data,
        "2024-05-16",
        freq="quarterly",
        annual_period_ends=["2023-12-31"],
    )

    assert out is data
    assert out.empty
