#!/usr/bin/env python3
"""
BrazilBench Determinism Test — Mistral 7B
Protocolo fiel ao BrazilBench: 80 dias consecutivos de PETR4/crisis_2020.
3 runs sem cold-start vs 3 runs com cold-start.
"""

import json, subprocess, time, urllib.request, hashlib, sys, os

try:
    import yfinance as yf
    import numpy as np
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "yfinance", "numpy", "-q"])
    import yfinance as yf
    import numpy as np

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral:7b"

SYSTEM_PROMPT = """You are a quantitative trading analyst evaluating Brazilian and US equities on behalf of an institutional portfolio.

For each market state, output EXACTLY this JSON and nothing else (no preamble, no markdown fences):
{"signal": "BUY"|"SELL"|"HOLD", "confidence": 0.0-1.0, "reasoning": "<= 80 words>"}

=== DECISION RULES ===
- BUY only when you expect positive risk-adjusted return over ~20 trading days relative to SELIC.
- SELL when: (R1) unrealized return < -8% from entry price, OR (R2) position is up > +15% AND 20d return has turned negative.
- HOLD is the default. Prefer HOLD under high uncertainty.
- For BR tickers: SELIC > 12% historically pressures equities; IPCA > 6% signals inflation regime; USD/BRL > 5.50 adds FX risk.
- Never base decisions on news outside the provided data.
- Output ONLY the JSON object."""

def get_petr4_data():
    print("Baixando dados PETR4.SA...")
    df = yf.download("PETR4.SA", start="2019-06-01", end="2020-06-01",
                     auto_adjust=True, progress=False)
    df = df.dropna()
    # Fix yfinance MultiIndex columns
    if isinstance(df.columns, object) and hasattr(df.columns, 'droplevel'):
        try:
            df.columns = df.columns.droplevel(1)
        except:
            pass
    prices  = [float(x) for x in df["Close"].values]
    volumes = [float(x) for x in df["Volume"].values]
    dates   = df.index.strftime("%Y-%m-%d").tolist()
    print(f"  {len(prices)} dias carregados. Range: {dates[0]} → {dates[-1]}")
    return prices, volumes, dates

def compute_stats(prices, volumes, idx):
    close      = prices[idx]
    lk20       = prices[max(0, idx-20):idx]
    lk60       = prices[max(0, idx-60):idx]
    ret_20d    = (close / lk20[0]  - 1) * 100 if len(lk20)  > 1 else 0.0
    ret_60d    = (close / lk60[0]  - 1) * 100 if len(lk60)  > 1 else 0.0
    if len(lk20) > 1:
        dr      = np.diff(lk20) / np.array(lk20[:-1])
        vol_20d = float(np.std(dr) * np.sqrt(252) * 100)
    else:
        vol_20d = 0.0
    high_20d   = max(lk20) if lk20 else close
    low_20d    = min(lk20) if lk20 else close
    vols_20    = volumes[max(0, idx-20):idx]
    vol_zscore = float((volumes[idx] - np.mean(vols_20)) / (np.std(vols_20) + 1e-9)) if len(vols_20) > 1 else 0.0
    return dict(close=close, ret_20d=ret_20d, ret_60d=ret_60d,
                vol_20d=vol_20d, high_20d=high_20d, low_20d=low_20d,
                vol_zscore=vol_zscore)

def build_prompt(date, prices_20, s, macro):
    price_str = ", ".join(f"{p:.2f}" for p in prices_20)
    return f"""TICKER: PETR4 (Brazil/B3) DATE: {date}

RECENT PRICES (last 20 trading days, oldest to newest):
{price_str}

PRICE STATISTICS:
- close: {s['close']:.2f}
- 20d return: {s['ret_20d']:.1f}% | 60d return: {s['ret_60d']:.1f}%
- 20d realized vol (annualized): {s['vol_20d']:.1f}%
- 20d high/low: {s['high_20d']:.2f} / {s['low_20d']:.2f}
- volume z-score (20d): {s['vol_zscore']:+.2f}

POSITION: None (flat)

MACRO CONTEXT:
- SELIC: {macro['selic']}%
- IPCA (monthly, last): {macro['ipca']}%
- USD/BRL: {macro['usdbrl']}

Output the JSON decision now."""

def call_ollama(user_prompt):
    payload = json.dumps({
        "model": MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": user_prompt,
        "options": {"temperature": 0},
        "stream": False
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read()).get("response", "").strip()

def run_80_days(prices, volumes, dates, label):
    macro = {"selic": 4.25, "ipca": 0.42, "usdbrl": 4.87}
    # Acha inicio do regime crisis_2020
    regime_start = next((i for i, d in enumerate(dates) if d >= "2020-02-03"), None)
    if regime_start is None:
        print("  ERRO: data de início não encontrada")
        return []
    signals = []
    for offset in range(80):
        idx = regime_start + offset
        if idx >= len(prices): break
        prices_20 = prices[max(0, idx-20):idx]
        stats     = compute_stats(prices, volumes, idx)
        prompt    = build_prompt(dates[idx], prices_20, stats, macro)
        try:
            resp   = call_ollama(prompt)
            signal = json.loads(resp).get("signal", "?")
        except:
            signal = "ERR"
        signals.append(signal)
        if (offset + 1) % 20 == 0:
            print(f"    {label}: {offset+1}/80 dias | último: {signal}")
    return signals

def kill_ollama():
    subprocess.run(["pkill", "ollama"], capture_output=True)
    time.sleep(3)

def start_cold():
    env = {**os.environ, "OLLAMA_KEEP_ALIVE": "0"}
    subprocess.Popen(["ollama", "serve"], env=env,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(6)

def wait_ollama():
    for _ in range(20):
        try:
            urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
            return True
        except: time.sleep(1)
    return False

def agreement(runs):
    n = min(len(r) for r in runs)
    days_agree = sum(len({r[d] for r in runs}) == 1 for d in range(n))
    return days_agree / n * 100

# ── MAIN ──────────────────────────────────────────────────────────
prices, volumes, dates = get_petr4_data()

print(f"\n{'='*60}")
print(f"MODELO: {MODEL} | PROTOCOLO: PETR4/crisis_2020 80 dias | temp=0")
print(f"{'='*60}")

# FASE 1: sem cold-start
print("\n[FASE 1] Sem cold-start")
print("-" * 40)
wait_ollama()
warm_runs = []
for i in range(1, 4):
    print(f"\n  Run {i}/3:")
    s = run_80_days(prices, volumes, dates, f"W{i}")
    warm_runs.append(s)
    h = hashlib.md5(str(s).encode()).hexdigest()[:8]
    print(f"  hash={h} | signals: {s[:5]}")

pct_warm = agreement(warm_runs)
print(f"\n  Acordo: {pct_warm:.1f}% | {'✅ DETERMINÍSTICO' if pct_warm==100 else '❌ NÃO DETERMINÍSTICO'}")

# FASE 2: com cold-start
print("\n[FASE 2] Com cold-start")
print("-" * 40)
cold_runs = []
for i in range(1, 4):
    kill_ollama(); start_cold(); wait_ollama()
    print(f"\n  Run {i}/3:")
    s = run_80_days(prices, volumes, dates, f"C{i}")
    cold_runs.append(s)
    h = hashlib.md5(str(s).encode()).hexdigest()[:8]
    print(f"  hash={h} | signals: {s[:5]}")

pct_cold = agreement(cold_runs)
print(f"\n  Acordo: {pct_cold:.1f}% | {'✅ DETERMINÍSTICO' if pct_cold==100 else '❌ NÃO DETERMINÍSTICO'}")

# SUMÁRIO
print(f"\n{'='*60}")
print(f"SUMÁRIO — {MODEL}")
print(f"  Sem cold-start: {pct_warm:.1f}% agreement")
print(f"  Com cold-start: {pct_cold:.1f}% agreement")
if pct_warm < 100 and pct_cold == 100:
    print("  ✅ ACHADO REPLICADO: cold-start restaura determinismo")
elif pct_warm == 100:
    print("  ⚠️  Mistral determinístico sem cold-start: near-tie tokens insuficientes")
print(f"{'='*60}")

with open("brazilbench_mistral_results.json","w") as f:
    json.dump({"model":MODEL,"warm_pct":pct_warm,"cold_pct":pct_cold,
               "warm_runs":warm_runs,"cold_runs":cold_runs}, f, indent=2)
print("\nResultados em brazilbench_mistral_results.json")
