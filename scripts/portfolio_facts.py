"""Write docs/portfolio/FACTS.md: what this repository lets one claim, read at a pinned
commit (PIN), in plain Portuguese, each fact with its type, denominator and source.

Types:

* prospectivo: a forecast locked before the outcome existed (there are none here);
* retrospectivo: a backtest on past data, classical baselines on committed fixtures;
* estado: a count of what exists in the repository at PIN.

Every number is computed from a file read with ``git show PIN:<path>`` (or from git
itself at PIN), never typed. Offline and stdlib only: no network, no LLM call.

    python scripts/portfolio_facts.py          # (re)write docs/portfolio/FACTS.md
    python scripts/portfolio_facts.py --check  # exit 1 if the committed file differs
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "portfolio" / "FACTS.md"
PIN = "7740909"  # origin/main of 2026-11-02 (#91), the state the inventory was read at
TYPES = ("prospectivo", "retrospectivo", "estado")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True
    ).stdout


def show(path: str) -> str:
    return git("show", f"{PIN}:{path}")


def rows(path: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(show(path))))


def find(pattern: str, text: str, what: str) -> re.Match[str]:
    m = re.search(pattern, text, re.DOTALL)
    if m is None:  # a source changed shape: fail loud, never guess
        raise SystemExit(f"portfolio_facts: {what} not found ({pattern!r})")
    return m


def num(x: float, nd: int = 2) -> str:
    """Portuguese decimal comma, explicit sign kept by the caller."""
    return f"{x:.{nd}f}".replace(".", ",")


def pct(x: float, nd: int = 2) -> str:
    return ("+" if x > 0 else "−" if x < 0 else "") + num(abs(x), nd) + " %"


def h1_not_run() -> tuple[str, str]:
    files = git("ls-tree", "-r", "--name-only", PIN).split()
    results = [
        f for f in files if f.endswith("cells.csv") and not f.startswith("tests/")
    ]
    roadmap = show("ROADMAP.md")
    phase2 = find(r"## Phase 2.*?(?=\n## )", roadmap, "ROADMAP Phase 2").group(0)
    items = re.findall(r"^- \[([ x])\] (.*)$", phase2, re.MULTILINE)
    steps = [
        mark for mark, txt in items if not txt.startswith("P3.")
    ]  # P3.x = preparation
    open_items, total_steps = steps.count(" "), len(steps)
    done_items = sum(1 for mark, txt in items if txt.startswith("P3.") and mark == "x")
    prereg = show("docs/PREREGISTRATION.md")
    find(r"No H1 factorial results \(`cells.csv`\) are committed", prereg, "adendo")
    value = (
        f"não rodou: {len(results)} arquivos `cells.csv` de resultado versionados; "
        f"{open_items} de {total_steps} passos de execução da Fase 2 abertos; "
        f"{done_items} itens de preparação (P3.x) feitos"
    )
    return (
        value,
        "`git ls-tree` em PIN; `ROADMAP.md` (Fase 2); `docs/PREREGISTRATION.md` (adendo de 2026-11-02)",
    )


def prereg() -> tuple[str, str, str, str]:
    text = show("docs/PREREGISTRATION.md")
    m = find(r"\*\*signed off, (\d{4}-\d{2}-\d{2}), by", text, "status")
    sha = find(
        r"signed version is the content at commit ([0-9a-f]{40})", text, "sha"
    ).group(1)
    runs = find(
        r"Total backtest runs \| (\d+) = (\d+) × (\d+) × (\d+)", text, "corridas"
    )
    groups = find(
        r"\| US \| ([^|]+) \|.*?\| BR macro-sensitive \| ([^|]+) \|", text, "grupos"
    )
    n_us = len(groups.group(1).split(","))
    n_br = len(groups.group(2).split(","))
    labelings = math.comb(n_us + n_br, n_us)
    find(r"\| α \(primary\) \| 0\.05, one-sided \|", text, "α")
    find(r"exact one-sided permutation test", text, "teste exato")
    find(r"No dollar figure is claimed here", text, "sem valor em dólar")
    in_main = (
        subprocess.run(
            ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", sha, PIN],
            check=False,
        ).returncode
        == 0
    )
    calls = find(
        r"(?:about|roughly) ([\d,]+) decisions, or about (\d+)k–(\d+)k LLM calls",
        text,
        "chamadas",
    )
    signed = (
        f"assinado em {m.group(1)} (versão `{sha[:7]}`"
        f"{'' if in_main else ', fora do histórico da main'}), "
        f"antes de qualquer corrida do H1; "
        f"{runs.group(1)} corridas planejadas ({runs.group(2)} ativos × {runs.group(3)} braços "
        f"× {runs.group(4)} sementes)"
    )
    test = (
        f"permutação exata sobre {labelings} rotulações ({n_br} ativos BR × {n_us} EUA), "
        f"p mínimo 1/{labelings} = {num(1 / labelings, 3)}, α = 0,05 unilateral"
    )
    cost = (
        f"cerca de {calls.group(1).replace(',', '.')} decisões, ou {calls.group(2)} mil a "
        f"{calls.group(3)} mil chamadas de LLM pago; nenhum valor em dólar no documento"
    )
    return signed, test, cost, "`docs/PREREGISTRATION.md` (status, §2, §6.1)"


def audit() -> tuple[str, str]:
    text = show("docs/TEMPORAL_AUDIT.md")
    leaks = re.findall(r"^\| (L\d+) \| (\w+) \|", text, re.MULTILINE)
    high = sum(1 for _, s in leaks if s == "High")
    if {s for _, s in leaks} - {"High", "Medium"}:
        raise SystemExit(
            "portfolio_facts: a leak severity other than High/Medium; recount"
        )
    roadmap = show("ROADMAP.md")
    items = re.findall(r"^- \[([ x])\] (P4\.(\d+))\b", roadmap, re.MULTILINE)
    fixes = [
        (mark, name) for mark, name, n in items if int(n) >= 2
    ]  # P4.1 is the audit
    done = [name for mark, name in fixes if mark == "x"]
    open_ = [name for mark, name in fixes if mark == " "]
    leak_value = (
        f"{len(leaks)} vazamentos achados ({high} graves, {len(leaks) - high} médios)"
    )
    fix_value = f"{len(done)} de {len(fixes)} itens de correção feitos; abertos: {', '.join(open_)}"
    return leak_value, fix_value


def timing_rule() -> str:
    git("merge-base", "--is-ancestor", "c6d990d", PIN)  # raises if #64 is not in PIN
    sha, day, subject = (
        git("log", "-1", "--format=%h|%cs|%s", "c6d990d").strip().split("|", 2)
    )
    find(r"decides before the open of D", subject, "assunto do #64")
    return f"decide com dados até o fechamento de D−1 e executa na abertura de D, desde {day} (`{sha}`, #64)"


def brazilbench_tables() -> str:
    tex = show("docs/brazilbench.tex")
    scope = find(r"\\label\{sec:scope\}(.*?)\\subsection", tex, "escopo").group(1)
    first, second = [p for p in scope.split("\n\n") if p.strip()][:2]
    regen = re.findall(r"\\ref\{(tab:[^}]+)\}", first)
    typed = re.findall(r"\\ref\{(tab:[^}]+)\}", second)
    files = set(git("ls-tree", "-r", "--name-only", PIN).split())
    if "\\graphicspath" in tex:
        raise SystemExit(
            "portfolio_facts: \\graphicspath changes figure paths; recount"
        )
    figs = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", tex)
    missing = [f for f in figs if f"docs/{f}" not in files]
    return (
        f"{len(regen)} de {len(regen) + len(typed)} tabelas de resultado regeneradas byte a "
        f"byte por `make reproduce` ({', '.join(regen)}); {len(typed)} digitadas à mão de um "
        f"download anterior ({', '.join(typed)}); {len(missing)} de {len(figs)} figuras citadas "
        f"não estão no repositório"
    )


def macd_covid() -> str:
    m = {
        (r["strategy"], r["ticker"], r["regime"]): r
        for r in rows("benchmark/results/brazilbench/matrix.csv")
    }
    find(
        r"frictionless",
        show("tradingagents/backtest/brazilbench.py"),
        "matriz sem custos",
    )
    macd = m[("MACD(12,26,9)", "PETR4", "crisis_2020")]
    bh = m[("Buy & Hold", "PETR4", "crisis_2020")]
    return (
        f"MACD {pct(100 * float(macd['cr']))} contra {pct(100 * float(bh['cr']))} do Buy & Hold "
        f"(PETR4, crise de 2020, {macd['n_days']} pregões, sem custos)"
    )


def momentum_gap() -> str:
    cells = {
        (r["agent"], r["regime"], r["ticker"]): float(r["total_return_pct"])
        for r in rows("benchmark/results/baselines/per_cell_returns.csv")
    }
    bvsp = cells[("momentum", "hi_rates_2022", "^BVSP")]
    br = [
        v
        for (a, g, t), v in cells.items()
        if a == "momentum" and g == "hi_rates_2022" and t != "^BVSP"
    ]
    tex = show("docs/brazilbench.tex")
    typed = find(
        r"Momentum lost\s*\$(-[\d.]+)\\%\$ on the Ibovespa", tex, "Momentum digitado"
    ).group(1)
    return (
        f"regenerado: Momentum {pct(bvsp)} no ^BVSP em juros altos (2022) e de {pct(min(br))} a "
        f"{pct(max(br))} nos {len(br)} ativos B3; o paper digitado diz {pct(float(typed), 1)} no Ibovespa"
    )


def random_floor() -> str:
    tex = show("docs/paper_random_n100.tex")
    cells = re.findall(
        r"& ([\d.-]+)\$\\pm\$([\d.]+) & [\d.-]+ & [\d.-]+ & [\d.-]+ & ([\d.]+) \\\\",
        tex,
    )
    if len(cells) != len(
        re.findall(r"\$\\pm\$\d", tex)
    ):  # every number±number row parsed
        raise SystemExit("portfolio_facts: a row of the random table did not parse")
    find(r"over seeds \$0\$--\$99\$", tex, "100 sementes")
    find(r"frictionless", tex, "sem custos")
    sds = [float(sd) for _, sd, _ in cells]
    tails = sum(1 for *_, p in cells if float(p) < 5 or float(p) > 95)
    below = sum(1 for s in sds if s < 10)
    return (
        f"desvio-padrão por célula de {num(min(sds), 1)} a {num(max(sds), 1)} pp ({below} de "
        f"{len(sds)} abaixo de 10 pp); a semente 42 cai na cauda (abaixo de 5 % ou acima de 95 %) "
        f"em {tails} de {len(sds)} células; 100 sementes por célula, sem custos"
    )


def hmm() -> str:
    regimes = [
        r
        for r in rows("benchmark/results/hmm_regimes/alignment.csv")
        if not r["regime"].startswith("(")
    ]
    days = sum(int(r["days"]) for r in regimes)
    purity = sum(int(r["days"]) * float(r["purity"]) for r in regimes) / days
    base = next(
        r
        for r in rows("benchmark/results/hmm_regimes/alignment.csv")
        if r["regime"] == "(all days)"
    )
    calm = sum(1 for r in regimes if r["majority_state"] == "calm")
    crisis = next(r for r in regimes if r["regime"] == "crisis_2020")
    ari = find(
        r"2-way HMM path: (\d+\.\d+)",
        show("benchmark/results/hmm_regimes/README.md"),
        "ARI",
    ).group(1)
    return (
        f"o estado majoritário é o calmo em {calm} de {len(regimes)} janelas, inclusive a crise de 2020 "
        f"({num(100 * float(crisis['share_turbulent']), 0)} % dos dias no estado turbulento); pureza "
        f"ponderada {num(purity, 3)} nas janelas ({days} pregões), abaixo da base de todos os dias "
        f"({num(float(base['purity']), 3)}); índice de Rand ajustado {ari.replace('.', ',')}"
    )


def survivorship() -> str:
    gaps = [
        float(r["gap_pp"]) for r in rows("benchmark/results/survivorship/summary.csv")
    ]
    ahead = sum(1 for g in gaps if g > 0)
    names = sorted(
        {
            r["ticker"]
            for r in rows("benchmark/results/survivorship/per_cell.csv")
            if r["universe"] == "distressed"
        }
    )
    return (
        f"líquidas menos em dificuldade: {pct(sum(gaps) / len(gaps)).replace(' %', ' pp')} em média, "
        f"em {len(gaps)} células, mas as líquidas ficam à frente em só {ahead} de {len(gaps)} "
        f"(de {pct(min(gaps)).replace(' %', ' pp')} a {pct(max(gaps)).replace(' %', ' pp')}); "
        f"{len(names)} nomes em dificuldade ({', '.join(names)}); sem teste de significância"
    )


def chronos() -> str:
    r = {
        x["agent"] + "/" + x["ticker"]: x
        for x in rows("benchmark/results/chronos/summary.csv")
    }["chronos/PETR4"]
    find(
        r"frictionless",
        show("benchmark/results/chronos/README.md"),
        "chronos sem custos",
    )
    regimes = {
        x["regime"]
        for x in rows("benchmark/results/chronos/per_cell.csv")
        if x["agent"] == "chronos" and x["ticker"] == "PETR4"
    }
    return (
        f"Chronos-t5-tiny na PETR4: retorno médio {pct(float(r['mean_cr_pct']))}, "
        f"{pct(float(r['gap_vs_bh_pct'])).replace(' %', ' pp')} contra o Buy & Hold, nos {len(regimes)} regimes, sem custos"
    )


def qwen() -> str:
    r = rows("benchmark/results/qwen_n10/summary.csv")
    agree = sum(1 for x in r if x["n_unique_responses"] == "1")
    meta = json.loads(show("benchmark/results/qwen_n10/summary.json"))
    temp = meta["options"]["temperature"]
    if meta["protocol"] != "cold-start":
        raise SystemExit("portfolio_facts: qwen protocol is not cold-start")
    find(
        r"qwen_coldstart_n10\.py:84,94` - the prompt's `close` is `closes\[idx\]` \(close of D\)",
        show("docs/TEMPORAL_AUDIT.md"),
        "look-ahead do qwen",
    )
    n = {x["n"] for x in r}
    signals = sorted({x["majority_signal"] for x in r})
    return (
        f"{meta['model']} local, temperatura {temp}: {len(r)} datas × {'/'.join(sorted(n))} partidas a frio, "
        f"respostas idênticas em {agree} de {len(r)} datas (todas {', '.join(signals)}); "
        f"o prompt usa o fechamento do dia da decisão (look-ahead)"
    )


def reliability() -> str:
    r = rows("benchmark/results/reliability/bins.csv")
    if len(r) != 1:
        raise SystemExit(
            "portfolio_facts: more than one confidence bin; recount prompts"
        )
    n, prompts = int(r[0]["n"]), int(r[0]["n_unique_prompts"])
    recs = [
        json.loads(ln)
        for ln in show("benchmark/results/reliability/decisions.jsonl").splitlines()
        if ln.strip()
    ]
    models = sorted({x["model"] for x in recs if x.get("confidence") is not None})
    return f"1 faixa de confiança, {n} decisões de {', '.join(models)}, {prompts} prompt único: não é calibração"


def authorship() -> str:
    commits = int(git("rev-list", "--count", PIN))
    per_commit = git(
        "log", PIN, "--format=%(trailers:key=Co-authored-by,valueonly)%x1e"
    ).split("\x1e")
    cursor = sum(
        1 for t in per_commit if "Cursor Agent" in t
    )  # commits, not trailer lines
    claude = sum(1 for t in per_commit if re.search(r"^Claude", t, re.MULTILINE))
    return (
        f"{commits} commits na main em `{PIN}`: {cursor} com coautoria do Cursor Agent e "
        f"{claude} do Claude (trailer `Co-authored-by`)"
    )


def code_size() -> str:
    files = git("ls-tree", "-r", "--name-only", PIN).split()
    out = []
    for top in ("tradingagents", "scripts", "tests"):
        py = [f for f in files if f.startswith(top + "/") and f.endswith(".py")]
        out.append(f"{top}: {len(py)} arquivos .py")
    counts = git("grep", "-c", "def test_", PIN, "--", "tests/*.py").splitlines()
    tests = sum(int(ln.rsplit(":", 1)[1]) for ln in counts)
    return "; ".join(out) + f"; {tests} funções `def test_` (contagem estática)"


def brazilfi() -> str:
    lock = show("uv.lock")
    m = find(
        r'name = "brazilfi"\nversion = "([^"]+)"\nsource = \{ registry = "([^"]+)" \}',
        lock,
        "brazilfi",
    )
    return f"brazilfi {m.group(1)}, instalado de {m.group(2)}"


def arena() -> str:
    files = [
        f
        for f in git("ls-tree", "-r", "--name-only", PIN).split()
        if f.startswith("benchmark/results/headline_arena/")
    ]
    find(r"No live arena results yet\.", show("README.md"), "Headline Arena")
    if (
        json.loads(show("benchmark/results/headline_arena/dry_run.json"))["mode"]
        != "dry_run"
    ):
        raise SystemExit("portfolio_facts: headline arena file is not a dry run")
    return f"0 resultados ao vivo; a pasta tem {len(files)} arquivos ({', '.join(Path(f).name for f in files)}), um ensaio offline"


def facts() -> list[tuple[str, str, str, str, str]]:
    h1, h1_src = h1_not_run()
    signed, test, cost, pre_src = prereg()
    leaks, fixes = audit()
    return [
        (
            "O experimento principal (H1)",
            "estado",
            h1,
            "O teste da hipótese (o agente macro ajuda mais no Brasil que nos EUA?) nunca foi executado. Não existe nenhum resultado de agente de IA operando.",
            h1_src,
        ),
        (
            "Pré-registro do H1",
            "estado",
            signed,
            "A hipótese, o teste estatístico e as regras de exclusão foram escritos e assinados antes de qualquer corrida do H1.",
            pre_src,
        ),
        (
            "Teste estatístico do H1",
            "estado",
            test,
            "O ativo é a unidade; por isso há poucas rotulações e o teste só rejeita com separação quase completa.",
            pre_src,
        ),
        (
            "Tamanho do H1",
            "estado",
            cost,
            "Rodar o H1 com o modelo pré-registrado (Claude, decisão 7) exige API paga.",
            pre_src,
        ),
        (
            "Auditoria temporal",
            "estado",
            leaks,
            "Lugares em que as ferramentas de dados dos agentes podiam ver o futuro num backtest, achados antes de rodar o experimento.",
            "`docs/TEMPORAL_AUDIT.md` §2",
        ),
        (
            "Correção dos vazamentos",
            "estado",
            fixes,
            "O P4.2 (fundamentos com a data certa) é pré-requisito do H1: hoje a ferramenta devolve os dados atuais da empresa.",
            "`ROADMAP.md` (F4); `docs/TEMPORAL_AUDIT.md` §10.2",
        ),
        (
            "Regra de decisão do backtest",
            "estado",
            timing_rule(),
            "Antes disso o runner decidia e executava no fechamento do mesmo dia, o que a auditoria classificou como vazamento (L9).",
            "`git log` (c6d990d)",
        ),
        (
            "Reprodutibilidade do BrazilBench",
            "estado",
            brazilbench_tables(),
            "Só parte do paper sai do código; o resto foi digitado.",
            "`docs/brazilbench.tex` (`sec:scope`)",
        ),
        (
            "MACD na crise da COVID",
            "retrospectivo",
            macd_covid(),
            "Um ativo, um regime, sem teste de significância. A explicação (uma saída e uma volta, 3 operações) vem da tabela digitada; o arquivo regenerado só tem os retornos.",
            "`benchmark/results/brazilbench/matrix.csv`",
        ),
        (
            "Momentum em juros altos (o lado que perde)",
            "retrospectivo",
            momentum_gap(),
            "A Finding 2 do paper não se reproduz pelo código: a regra (venda abaixo de −5 %) e os custos (25 bps) do texto não estão no código da main.",
            "`benchmark/results/baselines/per_cell_returns.csv`; `docs/brazilbench.tex` (Finding 2)",
        ),
        (
            "Ruído de uma política aleatória",
            "retrospectivo",
            random_floor(),
            "Quanto um agente sem informação nenhuma varia só por sorte; uma única semente não basta como referência.",
            "`docs/paper_random_n100.tex`",
        ),
        (
            "Regimes à mão contra um HMM",
            "retrospectivo",
            hmm(),
            "A volatilidade sozinha não separa os regimes escolhidos à mão: um modelo estatístico de 2 estados chama quase todos os dias de calmos. Nenhuma estratégia foi rodada de novo.",
            "`benchmark/results/hmm_regimes/alignment.csv` e `README.md`",
        ),
        (
            "Viés de sobrevivência",
            "retrospectivo",
            survivorship(),
            "Quanto o resultado muda ao trocar ações líquidas por empresas em dificuldade; com 3 nomes é ilustração, não prova.",
            "`benchmark/results/survivorship/`",
        ),
        (
            "Comparador Chronos",
            "retrospectivo",
            chronos(),
            "Um modelo de previsão pequeno, usado só pelo sinal da previsão.",
            "`benchmark/results/chronos/summary.csv`",
        ),
        (
            "LLM local: determinismo",
            "retrospectivo",
            qwen(),
            "Mostra só que a temperatura 0 repete a resposta; não mede acerto.",
            "`benchmark/results/qwen_n10/summary.csv`; `docs/TEMPORAL_AUDIT.md` §10",
        ),
        (
            "LLM local: confiança",
            "retrospectivo",
            reliability(),
            "Um esqueleto de análise, sem dados para concluir nada.",
            "`benchmark/results/reliability/bins.csv`",
        ),
        (
            "Headline Arena (braço prospectivo)",
            "estado",
            arena(),
            "O único caminho com previsões travadas antes do resultado ainda não rodou.",
            "`benchmark/results/headline_arena/`; `README.md`",
        ),
        (
            "Autoria e uso de IA",
            "estado",
            authorship(),
            "Commits com trailer de coautoria de IA; um commit sem trailer também pode ter usado IA.",
            "`git log` em PIN",
        ),
        (
            "Tamanho do código",
            "estado",
            code_size(),
            "O pacote `tradingagents` vem do TradingAgents (Xiao et al., 2024); a divisão entre o código novo e o herdado não é medida aqui.",
            "`git ls-tree` em PIN",
        ),
        (
            "SDK de dados brasileiros",
            "estado",
            brazilfi(),
            "O agente macro lê SELIC, IPCA, PIB e câmbio por esse pacote, que o PAPER atribui ao mesmo autor.",
            "`uv.lock`; `PAPER.md` §3.1",
        ),
    ]


def build() -> str:
    log = git(
        "log", PIN, "--format=%s%x1f%(trailers:key=Co-authored-by,valueonly)%x1e"
    ).split("\x1e")
    entries = [e.strip("\n").split("\x1f", 1) for e in log if e.strip()]
    commits_n = len(entries)
    test_entries = [t for s, t in entries if re.match(r"tests?\b", s)]
    tests_n, tests_cursor = (
        len(test_entries),
        sum(1 for t in test_entries if "Cursor Agent" in t),
    )
    queue = show("docs/archive/PR_QUEUE.md")
    upgrades = find(
        r'None of the "Adds \(upgrade\)" deliverables reached `main`\.',
        queue,
        "PR_QUEUE",
    ).group(0)
    find(
        r"Transaction costs, seeds,.*?bootstrap/deflated\s*>?\s*Sharpe/Holm",
        queue,
        "lista dos upgrades",
    )
    workflow = show(".github/workflows/tests.yml")
    find(r"actions/checkout@", workflow, "checkout do CI")
    ci_note = (
        ""
        if "fetch-depth: 0" in workflow
        else (
            " (localmente; o CI faz clone raso, sem `fetch-depth: 0` em `.github/workflows/tests.yml`, "
            "então lá o commit fixado não existe e o teste é pulado)"
        )
    )
    day = git("log", "-1", "--format=%cs", PIN).strip()
    table = facts()
    prospective = sum(1 for _, kind, *_ in table if kind == "prospectivo")
    lines = [
        "# Fatos do portfólio: Autonomous Hedge Fund (trading multiagente)",
        "",
        f"Gerado por `scripts/portfolio_facts.py` a partir do commit `{PIN}` ({day}). Não edite à mão: "
        "`tests/test_portfolio_facts.py` falha se este arquivo não for a regeneração exata"
        + ci_note
        + ".",
        "",
        (
            "Tipos: **prospectivo** = previsão gravada antes do fato e conferida depois; **retrospectivo** = "
            "backtest num período já visto (descreve, não prova); **estado** = contagem do que existe no "
            "repositório nesse commit. Toda fração vem com o denominador."
        ),
        "",
        (
            "**O lado que perde, primeiro:** o H1 nunca rodou, não há nenhum resultado de agente de IA "
            f"operando, e fatos prospectivos: {prospective}."
        ),
        "",
        "| # | fato | valor | o que significa | tipo | fonte |",
        "|---|---|---|---|---|---|",
    ]
    for i, (title, kind, value, meaning, source) in enumerate(table, 1):
        lines.append(f"| {i} | {title} | {value} | {meaning} | {kind} | {source} |")
    lines += [
        "",
        "## O que não está aqui, e por quê",
        "",
        '- **Resultado de agente de IA, alfa ou "o agente macro ajuda":** não existe; é a hipótese, não um resultado.',
        "- **Quantos testes passam:** o resultado do CI não fica gravado no repositório; aqui só a contagem estática.",
        "- **Quanto do código é novo:** o repositório não tem o histórico do upstream para comparar.",
        f"- **Os upgrades de setembro (`up/17`…`up/42`: custos, sementes, bootstrap, PSR/DSR, Holm):** ficaram em branches locais, fora da main; `docs/archive/PR_QUEUE.md` diz: {upgrades}",
        f"- **O número de PRs como mérito:** {tests_n} de {commits_n} commits da main têm assunto `test`/`tests` ({tests_cursor} com coautoria do Cursor Agent); a contagem de PRs não mede trabalho de pesquisa.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if the file differs")
    args = ap.parse_args(argv)
    text = build()
    if args.check:
        same = OUT.is_file() and OUT.read_text(encoding="utf-8") == text
        if not same:
            print(
                f"{OUT} is stale or hand-edited: run python scripts/portfolio_facts.py",
                file=sys.stderr,
            )
        return 0 if same else 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
