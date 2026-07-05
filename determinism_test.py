#!/usr/bin/env python3
"""
Ollama Determinism Test
Testa se temperature=0 garante reprodutibilidade no Ollama.
Protocolo: 3 runs sem cold-start vs 3 runs com cold-start.
"""

import json
import subprocess
import time
import urllib.request
import urllib.error
import hashlib
import sys

OLLAMA_URL = "http://localhost:11434/api/generate"

# Prompt financeiro complexo (mesmo formato do BrazilBench)
# Valores reais de PETR4 em fev-mar 2020 (crisis_2020)
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

USER_PROMPT = """TICKER: PETR4 (Brazil/B3) DATE: 2020-03-06

RECENT PRICES (last 20 trading days, oldest to newest):
30.12, 29.87, 30.45, 31.02, 30.78, 29.54, 28.93, 29.21, 28.45, 27.89,
26.34, 25.67, 24.89, 23.45, 22.78, 21.34, 20.56, 19.87, 18.45, 17.23

PRICE STATISTICS:
- close: 17.23
- 20d return: -42.8% | 60d return: -38.2%
- 20d realized vol (annualized): 87.4%
- 20d high/low: 31.02 / 17.23
- volume z-score (20d): +3.42

POSITION: None (flat)

MACRO CONTEXT:
- SELIC: 4.25%
- IPCA (monthly, last): 0.42%
- USD/BRL: 4.78

Output the JSON decision now."""


def call_ollama(model: str) -> dict:
    """Faz uma chamada à API do Ollama e retorna response + hash."""
    payload = json.dumps({
        "model": model,
        "system": SYSTEM_PROMPT,
        "prompt": USER_PROMPT,
        "options": {"temperature": 0},
        "stream": False
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
            response_text = data.get("response", "")
            response_hash = hashlib.md5(response_text.encode()).hexdigest()[:8]
            return {"text": response_text, "hash": response_hash}
    except Exception as e:
        return {"text": f"ERROR: {e}", "hash": "ERROR"}


def kill_ollama():
    subprocess.run(["pkill", "ollama"], capture_output=True)
    time.sleep(3)


def start_ollama_cold():
    proc = subprocess.Popen(
        ["ollama", "serve"],
        env={**__import__("os").environ, "OLLAMA_KEEP_ALIVE": "0"},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(6)
    return proc


def wait_for_ollama():
    for _ in range(20):
        try:
            urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
            return True
        except:
            time.sleep(1)
    return False


def run_experiment(model: str):
    print(f"\n{'='*60}")
    print(f"MODELO: {model}")
    print(f"PROMPT: PETR4/crisis_2020 (~500 tokens, financial reasoning)")
    print(f"TEMPERATURE: 0")
    print(f"{'='*60}")

    # ── Fase 1: sem cold-start ──────────────────────────────────
    print("\n[FASE 1] Sem cold-start (servidor persistente)")
    print("-" * 40)

    if not wait_for_ollama():
        print("Ollama não está rodando. Iniciando...")
        start_ollama_cold()

    results_warm = []
    for i in range(1, 4):
        print(f"  Run {i}/3 ...", end=" ", flush=True)
        r = call_ollama(model)
        results_warm.append(r)
        print(f"hash={r['hash']} | {r['text'][:60].strip()}...")

    hashes_warm = [r["hash"] for r in results_warm]
    all_same_warm = len(set(hashes_warm)) == 1
    print(f"\n  Resultado: {'✅ IDÊNTICAS' if all_same_warm else '❌ DIVERGENTES'}")
    print(f"  Hashes: {hashes_warm}")

    # ── Fase 2: com cold-start ──────────────────────────────────
    print("\n[FASE 2] Com cold-start (servidor restartado entre runs)")
    print("-" * 40)

    results_cold = []
    for i in range(1, 4):
        kill_ollama()
        start_ollama_cold()
        if not wait_for_ollama():
            print(f"  Run {i}: Ollama não respondeu")
            continue
        print(f"  Run {i}/3 ...", end=" ", flush=True)
        r = call_ollama(model)
        results_cold.append(r)
        print(f"hash={r['hash']} | {r['text'][:60].strip()}...")

    hashes_cold = [r["hash"] for r in results_cold]
    all_same_cold = len(set(hashes_cold)) == 1
    print(f"\n  Resultado: {'✅ IDÊNTICAS' if all_same_cold else '❌ DIVERGENTES'}")
    print(f"  Hashes: {hashes_cold}")

    # ── Sumário ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"SUMÁRIO — {model}")
    print(f"  Sem cold-start: {'determinístico' if all_same_warm else 'NÃO determinístico'}")
    print(f"  Com cold-start: {'determinístico' if all_same_cold else 'NÃO determinístico'}")

    if not all_same_warm and all_same_cold:
        print(f"  ✅ ACHADO CONFIRMADO: KV-cache causa não-determinismo")
    elif all_same_warm and all_same_cold:
        print(f"  ⚠️  Ambos determinísticos: prompt sem near-tie tokens para este modelo")
    elif not all_same_warm and not all_same_cold:
        print(f"  🔴 Não determinístico mesmo com cold-start: outro mecanismo")
    print(f"{'='*60}")

    return {
        "model": model,
        "warm_deterministic": all_same_warm,
        "cold_deterministic": all_same_cold,
        "warm_hashes": hashes_warm,
        "cold_hashes": hashes_cold,
        "warm_outputs": [r["text"] for r in results_warm],
        "cold_outputs": [r["text"] for r in results_cold],
    }


if __name__ == "__main__":
    models = sys.argv[1:] if len(sys.argv) > 1 else ["mistral:7b"]

    all_results = []
    for model in models:
        result = run_experiment(model)
        all_results.append(result)

    # Salva resultados em JSON para o paper
    with open("determinism_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResultados salvos em determinism_results.json")
