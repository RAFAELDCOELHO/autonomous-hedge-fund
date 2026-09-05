# NeurIPS 2026 Datasets & Benchmarks Checklist (BrazilBench)

This checklist is filled for the repository artifact at:
<https://github.com/RAFAELDCOELHO/autonomous-hedge-fund>

Offline reproduction command (no API key, no network):

```bash
make reproduce
```

Scope note: the zero-cost path reports **classical baselines only** (Buy & Hold, MACD, SMA) on committed fixtures. LLM-agent results are explicitly future work in `docs/brazilbench.tex` (Limitations).

## Checklist

| Item | Yes/No/NA | Repo-grounded justification |
|---|---|---|
| Claims are clearly stated and scoped | Yes | `README.md` and `docs/brazilbench.tex` state this artifact is classical-only on the $0 path; no claimed LLM results. |
| Limitations are described | Yes | `docs/brazilbench.tex` Section `Limitations and Future Work` lists no-LLM-in-this-draft scope, hand-typed table caveats, and significance-test gaps. |
| Reproducibility instructions are provided | Yes | `make bench` and `make reproduce` are documented in `README.md`; outputs are tied to committed fixtures and CI checks (`tests/test_reproduce.py`). |
| Data provenance is documented | Yes | Inputs are committed local fixtures under `benchmark/prices/` (and `benchmark/prices/paper/`), with no online download in the reproduce path. |
| Baselines are appropriate and implemented | Yes | Classical baselines are implemented and documented (`docs/brazilbench_baselines.md`, `scripts/run_brazilbench.py`). |
| Statistical uncertainty for the random baseline is addressed | Yes | Random null distribution is generated over N=100 seeds (`docs/paper_random_n100.tex`, `benchmark/results/random_n100/*`). |
| LLM/API-dependent evaluations are reproducible from this artifact | No | Intentionally out of scope for the $0 artifact; README and paper state LLM runs need paid API or GPU inference and are future work. |
| Human subject data / personally identifiable information | No | No human-subject data is collected or evaluated in this repository artifact. |
| External paid services required for core artifact reproduction | No | `make bench` and `make reproduce` run offline from committed files and do not require paid APIs. |
| License and citation metadata are available | Yes | Apache-2.0 `LICENSE` is present; `CITATION.cff` is included for software and paper-artifact citation guidance. |
| Compute/environment requirements are documented | Yes | Environment lock is documented (`uv.lock`, `.python-version`, `Dockerfile.bench`) and README covers offline commands. |
| Potential misuse or risk discussion | Yes | Paper discusses scope limits and emphasizes descriptive (not significance-proven) claims; no trading-performance guarantees are made. |

## Out-of-scope for this artifact

- Live multi-agent runs (`uv run python main.py`) require `ANTHROPIC_API_KEY`.
- Headline Arena is a separate forward-only path and not part of the backtest reproducibility contract.
