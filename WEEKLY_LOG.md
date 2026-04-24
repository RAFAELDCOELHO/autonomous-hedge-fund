# Weekly Log

## Week of 2026-04-22 to 2026-04-24

### Shipped
- Macro Economist Agent integrado ao TradingAgents: tools brazilfi
  (SELIC/IPCA/GDP/câmbio), agent factory, setup.py, `AgentState.macro_report`.
  Exportado via `tradingagents.agents`.
- `tradingagents/backtest/agent_integration.py` (140 linhas):
  `map_signal` (5→3 classes), `make_decide_fn` (factory+closure),
  `run_tradingagents_backtest` (wrapper).
- Validação: mock BUY = BuyAndHold em AAPL Jan 2024. CR/Sharpe/MDD
  idênticos. Prova que a camada de integração é neutra.
- `CONTRIBUTION.md` expandido com seção Implementation Architecture:
  Macro Economist Agent, Backtest Infrastructure, TradingAgents
  Integration Layer, Validation.
- yfinance BR: verificado. `.SA` funciona sem adaptação. ITUB4.SA
  baixou 692 linhas, range 2021-04-26 a 2024-01-31, filtro de
  look-ahead preservado.
- 8 commits na branch main. Working tree limpo.

### Learned
- **`iloc[:i+1]` em `run_agent_strategy`:** por que incluir o índice
  `i` (presente) e não só `:i` (passado). O agente real tem o preço
  de fechamento quando decide — esconder isso subestima a estratégia.
  Look-ahead seria `:i+2` (ver amanhã). Regra: presente sim, futuro não.
- **Factory + closure pra injeção de dependência:** `make_decide_fn`
  fecha sobre `propagate_fn` e retorna função com assinatura do
  runner. Isso permite trocar o grafo sem mexer no runner e
  mockar o agent trivialmente.

### Blocked
- **Anthropic API billing bloqueado desde 21/04.** 4 follow-ups
  enviados até 24/04 — vou pausar contato até 01/05 (over-contact
  desprioriza ticket). Plano B: usar créditos de API key do meu pai
  como medida temporária até destravar. Smoke test do 2x2 factorial
  depende disso.

### Next Week
1. Auditar `tradingagents/backtest/baselines.py` pra verificar como
   SMA/MACD tratam dias iniciais sem histórico suficiente. Critério:
   documentar comportamento no CONTRIBUTION.md e confirmar que não
   vicia comparação com agents.
2. Conversar com meu pai sobre acesso temporário à API key. Se ok,
   rodar smoke test com 1 ticker (AAPL ou ITUB4.SA) e 1 agent.
3. Começar coleta de prices para os 9 tickers do 2x2 (3 US + 6 BR).
   Cache preenchido = experimento mais rápido quando destravar.

### Meta
Semana empolgada e produtiva. 4 dias intensos, 8 commits, dois
conceitos novos consolidados (iloc, closure). O bloqueio de billing
é frustrante mas não está me travando — tem trabalho técnico
adjacente suficiente.
