# Research Log — Autonomous Hedge Fund Research

## 2026-04-21 — Semana 1, Dia 1 (Terça)

### Objetivo do dia
Reproduzir o baseline TradingAgents (arXiv:2412.20138) no meu Mac usando Claude como backbone via API Anthropic.

### Setup completado
- Python 3.12.13 via uv
- Clone do repo oficial: github.com/TauricResearch/TradingAgents
- Reorganização de filesystem: brazilfi, tradingagents-original e autonomous-hedge-fund como repos irmãos em ~/projetos/
- venv recriada após mv (pyenv shim quebrou paths)
- 108 pacotes instalados

### Arquitetura entendida
- TradingAgents é model-agnostic via langchain wrapper
- 4 analistas (Market, Social, News, Fundamentals) → debate Bull/Bear → Research Manager → Trader → debate de risco (3 agentes) → Portfolio Manager → Signal Processor
- Orquestração: LangGraph StateGraph em tradingagents/graph/setup.py
- ~17-25 chamadas LLM por dia de trading
- Paper original usava Finnhub/EODHD/Reddit; repo atual usa Yahoo Finance + Alpha Vantage
- Não existe harness de backtest — propagate() roda 1 dia, loop é do usuário

### Bloqueios resolvidos
1. KeyError data_cache_dir: config precisa estender DEFAULT_CONFIG, não substituir.
2. Naming de modelos: aliases curtos do catálogo interno não funcionam via API; precisa ID completo com date-stamp.
3. pyenv shim quebrou venv após mv; fix: recriar venv.
4. Workspace errado da API key: primeira key criada antes do credit purchase.

### Bloqueio ativo (3+ horas)
Credit balance block persistente na API Anthropic.
- Saldo confirmado: $20.13 em Rafael's Individual Org (plano API)
- Monthly spend: $3.85 de $100
- API funciona parcialmente: models.list() OK, messages.create() falha
- Tentativas falhadas com múltiplos modelos: sonnet-4-5, opus-4
- Ticket Fin não escala para humano
- Email enviado pra support@anthropic.com
- Bloqueio persiste além de 2h após compra de créditos

### Aprendizado estratégico (importante)
Setup de infra para research com APIs pagas tem custos ocultos significativos:
- Billing, workspaces, organizations — cada camada tem pegadinha
- Bot de suporte não resolve casos não-padrão, precisa forçar escalação
- Documentar cada obstáculo é parte real do trabalho

### Progresso no plano de 8 semanas
- [x] Clone do TradingAgents original
- [x] Setup Python/venv/dependências
- [x] Entender arquitetura de alto nível
- [x] Script smoke_test.py criado
- [x] Reorganização de repositórios
- [ ] Smoke test rodando (BLOQUEADO: billing Anthropic)
- [ ] Harness de backtest
- [ ] Backtest AAPL Jan 2024 completo

### Próximos passos
1. Aguardar resolução do billing (via email support@anthropic.com)
2. Enquanto isso: leitura estruturada do código do TradingAgents
3. Esboçar design do Macro Economist Agent no papel
4. Começar draft do PAPER.md com seções que não dependem de resultados
