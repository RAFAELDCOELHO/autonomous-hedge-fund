"""CLI for running the academic backtest comparison.

Compares Buy & Hold, MACD(12/26/9), SMA(50/200), and TradingAgents arms
over a single-ticker window. TradingAgents runs two harness labels:
`baseline` (mapped from `no_macro` in scripts/headline_arena_arms.py) and
`macro`. Use `--arms` to choose which arms to run (default: baseline,macro)
or `--skip-agents` to run only classical baselines. Prints a rich table
of CR / AR / Sharpe / MDD. The TradingAgents arms' cash earns the market's
daily rf (CDI for .SA tickers, DTB3 otherwise) from data/rf/, so the agent
window must lie inside 2023-12-01..2024-04-30 (daily_rf raises otherwise).
The table shows Sharpe at the flat exploratory rf for every strategy, and the
agent arm's H1 Sharpe (excess over the daily rf) beside it.
``--cells-out`` appends one PREREGISTRATION §4 row per agent arm
(baseline → absent, macro → present; B3 tickers stored without ``.SA``),
plus ``start``/``end`` and the five config columns
(``deep_think_llm``, ``quick_think_llm``, ``temperature``,
``max_debate_rounds``, ``max_risk_discuss_rounds``).
It also writes ``sharpe_flat`` for exploratory PREREGISTRATION §7 analyses
and leaves ``sharpe_flat`` empty on failed rows.
Before any API key check, download, or LLM call, it preflights ticker/market,
the fixed preregistered window (2024-01-02..2024-03-28), existing header
validity, duplicate (ticker, arm, seed) keys, and rows already logged with a
different config; if rejected, it exits 2 without touching the file.
If an arm raises, it records ``status=failed`` and continues with the next arm,
and any failed arm makes the CLI exit non-zero.

Usage:
    uv run python run_backtest.py --ticker AAPL --start 2024-01-02 --end 2024-03-28
    uv run python run_backtest.py --ticker AAPL --start 2023-01-01 --end 2024-01-01 --skip-agents
    uv run python run_backtest.py --ticker AAPL --start 2024-01-02 --end 2024-03-28 --arms macro
    uv run python run_backtest.py --ticker PETR4.SA --start 2024-01-02 --end 2024-03-28 --cells-out cells.csv --seed 0
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from pathlib import Path
from runpy import run_path
import sys
import tempfile

from dotenv import load_dotenv

from tradingagents.backtest import (
    BuyAndHold,
    MACDStrategy,
    SMACrossStrategy,
    print_comparison,
    run_strategy,
    run_agent_strategy,
    run_buy_and_hold_at_open,
)
from tradingagents.backtest.agent_integration import make_decide_fn
from tradingagents.backtest.cells import (
    COLUMNS,
    PREREG_TICKERS,
    PREREG_WINDOW_END,
    PREREG_WINDOW_START,
    SHARPE_FLAT_FIELD,
    append_cells,
    bare_ticker,
    make_cell_row,
    prereg_arm,
)
from tradingagents.backtest.risk_free import market_of
from tradingagents.default_config import DEFAULT_CONFIG

CELLS_EXTRA_COLUMNS = (
    "start",
    "end",
    "deep_think_llm",
    "quick_think_llm",
    "temperature",
    "max_debate_rounds",
    "max_risk_discuss_rounds",
    "data_cutoff",
)
CELLS_CONFIG_COLUMNS = (
    "deep_think_llm",
    "quick_think_llm",
    "temperature",
    "max_debate_rounds",
    "max_risk_discuss_rounds",
)


def _load_headline_arena_arms() -> dict[str, dict[str, object]]:
    script = Path(__file__).resolve().parent / "scripts" / "headline_arena_arms.py"
    return run_path(str(script))["ARMS"]


def _selected_analysts_by_arm() -> dict[str, list[str]]:
    arms_data = _load_headline_arena_arms()
    # no_macro (arms file) == baseline (harness label) == macro absent.
    baseline_arm = arms_data.get("no_macro")
    macro_arm = arms_data.get("macro")
    arms: dict[str, list[str]] = {}
    if baseline_arm and "selected_analysts" in baseline_arm:
        arms["baseline"] = list(baseline_arm["selected_analysts"])
    if macro_arm and "selected_analysts" in macro_arm:
        arms["macro"] = list(macro_arm["selected_analysts"])
    return arms


def _run_agent_decider(
    ticker: str,
    start: str,
    end: str,
    capital: float,
    run_config: dict[str, object],
):
    """Run TradingAgents once per trading day and return an equity curve.

    Falls back to None if the pipeline cannot be constructed.
    """
    try:
        decide_fn = make_decide_fn(ticker=ticker, config=run_config)
    except Exception as e:
        logging.warning("TradingAgents pipeline unavailable (%s)", e)
        return None

    return run_agent_strategy(decide_fn, ticker, start, end, capital, market=market_of(ticker))


def _build_run_config(selected_analysts: list[str] | None = None) -> dict[str, object]:
    """Build one run config object consumed by both run and cells.csv logging."""
    config = DEFAULT_CONFIG.copy()
    if selected_analysts is not None:
        config["selected_analysts"] = list(selected_analysts)
    return config


def _parse_arms_csv(value: str) -> list[str]:
    allowed = {"baseline", "macro"}
    parsed = [token.strip().lower() for token in value.split(",") if token.strip()]
    if not parsed:
        raise ValueError("--arms must include at least one of: baseline,macro")
    invalid = [token for token in parsed if token not in allowed]
    if invalid:
        raise ValueError(
            f"--arms received invalid arm(s): {','.join(invalid)}. Allowed: baseline,macro"
        )
    deduped: list[str] = []
    for token in parsed:
        if token not in deduped:
            deduped.append(token)
    return deduped


def _planned_cells_keys(ticker: str, arms: list[str], seed: int) -> set[tuple[str, str, str]]:
    bare = bare_ticker(ticker)
    return {(bare, prereg_arm(arm), str(seed)) for arm in arms}


def _read_existing_cells_keys(path: Path) -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        _validate_cells_header(path, fieldnames)
        for row in reader:
            keys.add((row.get("ticker", ""), row.get("arm", ""), row.get("seed", "")))
    return keys


def _validate_cells_header(path: Path, fieldnames: list[str]) -> None:
    missing_required = [column for column in COLUMNS if column not in fieldnames]
    if missing_required:
        raise ValueError(f"{path} is missing required columns: {missing_required}")
    has_start = "start" in fieldnames
    has_end = "end" in fieldnames
    if has_start != has_end:
        raise ValueError(f"{path} must include both start and end columns together")
    has_any_config = any(name in fieldnames for name in CELLS_CONFIG_COLUMNS)
    has_all_config = all(name in fieldnames for name in CELLS_CONFIG_COLUMNS)
    if has_any_config and not has_all_config:
        raise ValueError(
            f"{path} must include all config columns together or none: "
            f"{list(CELLS_CONFIG_COLUMNS)}"
        )


def _ensure_cells_extra_columns(path: Path, extra_columns: tuple[str, ...]) -> None:
    """Ensure cells.csv has extra_columns, appended in order; existing rows get them empty."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        fieldnames = [*COLUMNS, *extra_columns]
        rows: list[dict[str, str]] = []
    else:
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fieldnames = list(reader.fieldnames or [])
            _validate_cells_header(path, fieldnames)
            extras_to_add = [name for name in extra_columns if name not in fieldnames]
            if not extras_to_add:
                return
            rows = list(reader)
            fieldnames = [*fieldnames, *extras_to_add]

    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({name: row.get(name, "") for name in fieldnames})
        os.replace(tmp_name, path)
    except Exception:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def _cells_config_values(config: dict[str, object]) -> dict[str, str]:
    # The runtime graph path does not currently read `temperature`; we still
    # log exactly what exists on the run_config object passed into execution
    # so recorded metadata cannot drift from the executed config snapshot.
    raw_temperature = config.get("temperature")
    if raw_temperature in (None, ""):
        temperature = "provider-default"
    else:
        temperature = str(raw_temperature)
    return {
        "deep_think_llm": str(config.get("deep_think_llm", "")),
        "quick_think_llm": str(config.get("quick_think_llm", "")),
        "temperature": temperature,
        "max_debate_rounds": str(config.get("max_debate_rounds", "")),
        "max_risk_discuss_rounds": str(config.get("max_risk_discuss_rounds", "")),
    }


assert tuple(_cells_config_values(DEFAULT_CONFIG).keys()) == CELLS_CONFIG_COLUMNS, (
    "CELLS_CONFIG_COLUMNS fora de sincronia com _cells_config_values"
)


def _row_has_logged_config(row: dict[str, str]) -> bool:
    return any((row.get(column, "") or "").strip() != "" for column in CELLS_CONFIG_COLUMNS)


def _find_config_mismatch(
    rows: list[dict[str, str]], expected_config: dict[str, str]
) -> tuple[dict[str, str], str, str] | None:
    for row in rows:
        if not _row_has_logged_config(row):
            continue
        for column, expected in expected_config.items():
            actual = (row.get(column, "") or "").strip()
            if actual != expected:
                key = (
                    row.get("ticker", ""),
                    row.get("arm", ""),
                    row.get("seed", ""),
                )
                return row, column, repr(key)
    return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run academic backtest comparison.")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--skip-agents", action="store_true",
                        help="Do not run the TradingAgents pipeline (baselines only)")
    parser.add_argument(
        "--arms",
        default="baseline,macro",
        help="Comma-separated TradingAgents arms to run: baseline,macro (default: baseline,macro)",
    )
    parser.add_argument(
        "--cells-out",
        type=Path,
        default=None,
        help="Append one cells.csv row per TradingAgents arm (PREREGISTRATION §4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Replicate index written with --cells-out (integer >= 0, default: 0)",
    )
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("--seed must be >= 0")
    try:
        requested_arms = _parse_arms_csv(args.arms)
    except ValueError as e:
        parser.error(str(e))
    if args.cells_out is not None:
        run_config_values = _cells_config_values(_build_run_config())
        bare = bare_ticker(args.ticker)
        expected_market = PREREG_TICKERS.get(bare)
        if expected_market is None:
            parser.error(
                f"--cells-out requires a pre-registered ticker; got {bare!r}. "
                "Use one from tradingagents.backtest.cells.PREREG_TICKERS."
            )
        actual_market = market_of(args.ticker)
        if actual_market != expected_market:
            parser.error(
                f"--cells-out ticker {args.ticker!r} resolves to market {actual_market}, "
                f"but preregistered {bare} market is {expected_market}. "
                "Use the Yahoo symbol expected by market_of (B3 tickers require .SA)."
            )
        if args.start != PREREG_WINDOW_START or args.end != PREREG_WINDOW_END:
            parser.error(
                f"--cells-out requires the preregistered H1 window "
                f"{PREREG_WINDOW_START}..{PREREG_WINDOW_END}; got {args.start}..{args.end}"
            )
        if args.skip_agents:
            arms_to_write = []
        else:
            available_arms = set(_selected_analysts_by_arm())
            arms_to_write = [arm for arm in requested_arms if arm in available_arms]
        planned = _planned_cells_keys(args.ticker, arms_to_write, args.seed)
        if planned and args.cells_out.exists() and args.cells_out.stat().st_size > 0:
            try:
                with args.cells_out.open(newline="", encoding="utf-8") as fh:
                    reader = csv.DictReader(fh)
                    fieldnames = list(reader.fieldnames or [])
                    _validate_cells_header(args.cells_out, fieldnames)
                    existing_rows = list(reader)
            except ValueError as exc:
                parser.error(str(exc))
            existing = {
                (row.get("ticker", ""), row.get("arm", ""), row.get("seed", ""))
                for row in existing_rows
            }
            duplicates = sorted(planned & existing)
            if duplicates:
                parser.error(
                    "--cells-out already contains (ticker, arm, seed) key(s) for this run: "
                    + ", ".join(repr(item) for item in duplicates)
                )
            has_config_columns = all(name in fieldnames for name in CELLS_CONFIG_COLUMNS)
            if has_config_columns:
                mismatch = _find_config_mismatch(existing_rows, run_config_values)
                if mismatch is not None:
                    row, column, key = mismatch
                    parser.error(
                        "--cells-out contains rows logged with a different run config: "
                        f"{column}={row.get(column, '')!r} at {key}; expected "
                        f"{run_config_values[column]!r} for this run"
                    )

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if not args.skip_agents and requested_arms:
        load_dotenv()
        if not os.getenv("ANTHROPIC_API_KEY"):
            logging.error(
                "ANTHROPIC_API_KEY is required when running agents; set it or use --skip-agents."
            )
            return 2

    curves = {}
    for strat in (BuyAndHold(), MACDStrategy(), SMACrossStrategy()):
        curves[strat.name] = run_strategy(strat, args.ticker, args.start, args.end, args.capital)

    if not args.skip_agents:
        # Same convention as the agent arms (decide before the open, fill and mark at opens).
        curves["Buy & Hold (open)"] = run_buy_and_hold_at_open(
            args.ticker, args.start, args.end, args.capital, market=market_of(args.ticker)
        )
        arms = _selected_analysts_by_arm()
        failed_arms: list[str] = []
        if not arms:
            logging.warning("No selected_analysts arms configured; skipping TradingAgents run")
        for arm_name in requested_arms:
            selected_analysts = arms.get(arm_name)
            if selected_analysts is None:
                logging.warning(
                    "Requested arm '%s' is unavailable in harness config; skipping", arm_name
                )
                continue
            arm_status = "failed"
            agent_curve = None
            run_config = _build_run_config(selected_analysts=selected_analysts)
            try:
                agent_curve = _run_agent_decider(
                    args.ticker,
                    args.start,
                    args.end,
                    args.capital,
                    run_config=run_config,
                )
            except Exception:
                logging.exception("TradingAgents arm '%s' failed; recording status=failed", arm_name)
            if agent_curve is not None:
                curves[f"TradingAgents ({arm_name})"] = agent_curve
                arm_status = "ok"
            else:
                failed_arms.append(arm_name)
            # Append each finished arm immediately so a later arm keeps it.
            if args.cells_out is not None:
                try:
                    _ensure_cells_extra_columns(
                        args.cells_out, (*CELLS_EXTRA_COLUMNS, SHARPE_FLAT_FIELD)
                    )
                    row = make_cell_row(
                        args.ticker,
                        arm_name,
                        args.seed,
                        equity=agent_curve,
                        status=arm_status,
                    )
                    row["start"] = args.start
                    row["end"] = args.end
                    row.update(_cells_config_values(run_config))
                    # start/end are decision dates; the agent saw closes up to data_cutoff.
                    row["data_cutoff"] = (
                        "" if agent_curve is None else agent_curve.attrs.get("data_cutoff", "")
                    )
                    append_cells(args.cells_out, [row])
                except ValueError as exc:
                    logging.error("%s", exc)
                    return 2
        if failed_arms:
            logging.error("Failed TradingAgents arms: %s", ",".join(failed_arms))
            print_comparison(curves, market=market_of(args.ticker))
            return 1

    print_comparison(curves, market=market_of(args.ticker))
    return 0


if __name__ == "__main__":
    sys.exit(main())
