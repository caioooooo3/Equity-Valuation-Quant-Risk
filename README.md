# Equity Valuation & Quant Risk Lab

Projeto de portfólio em Python para **análise de ações, valuation e risco quantitativo**, combinando dados de mercado, métricas estatísticas, DCF e simulação de Monte Carlo.

> Projeto educacional. Não constitui recomendação de investimento.

## Objetivos

- analisar preços e retornos históricos;
- calcular retorno anualizado, volatilidade, beta, Sharpe, drawdown e VaR histórico;
- estimar valor intrínseco por um DCF simplificado;
- testar sensibilidade a WACC e crescimento;
- simular cenários de preço com Monte Carlo;
- manter um fluxo reproduzível, modular e testável.

## Stack

`Python` · `Pandas` · `NumPy` · `Matplotlib` · `yfinance` · `scikit-learn` (ecossistema de estudo) · `pytest`

## Estrutura

```text
.
├── main.py
├── requirements.txt
├── src/
│   ├── data.py
│   ├── metrics.py
│   ├── valuation.py
│   ├── monte_carlo.py
│   └── reporting.py
├── notebooks/
│   └── 01_equity_valuation_and_risk.ipynb
├── tests/
│   └── test_core.py
├── docs/
│   └── METHODOLOGY.md
└── outputs/
```

## Métricas quantitativas

- retorno anualizado;
- volatilidade anualizada;
- beta contra benchmark;
- Sharpe Ratio;
- maximum drawdown;
- VaR histórico diário de 95%.

## Valuation

O DCF parte do Free Cash Flow mais recente disponível e usa:

- crescimento explícito do FCF;
- WACC;
- crescimento na perpetuidade;
- caixa;
- dívida;
- ações em circulação.

Também é gerada uma matriz de sensibilidade **WACC × crescimento do FCF**.

## Monte Carlo

A simulação utiliza momentos históricos dos log-retornos em um processo do tipo GBM. São calculados:

- percentil 5%;
- mediana;
- percentil 95%;
- probabilidade de preço terminal abaixo do preço inicial;
- perda associada ao percentil inferior da simulação.

## Instalação

```bash
python -m venv .venv
pip install -r requirements.txt
```

## Execução

Exemplo para uma ação dos EUA:

```bash
python main.py --ticker AAPL --benchmark '^GSPC' --period 5y
```

Exemplo para Brasil:

```bash
python main.py --ticker PETR4.SA --benchmark '^BVSP' --period 5y
```

Com premissas explícitas:

```bash
python main.py --ticker AAPL --forecast-growth 0.06 --wacc 0.09 --terminal-growth 0.025 --mc-sims 10000 --mc-days 252
```

Se algum dado fundamental não vier da fonte, é possível fornecer manualmente `--fcf`, `--cash`, `--debt` e `--shares`.

## Saídas

Para cada ticker são gerados arquivos em `outputs/<ticker>/`, incluindo:

- `metrics.csv`;
- `summary.json`;
- `dcf_sensitivity.csv`, quando o DCF pode ser calculado;
- histórico de preços;
- gráfico de drawdown;
- caminhos de Monte Carlo;
- distribuição terminal;
- heatmap de sensibilidade do DCF.

## Testes

```bash
pytest -q
```

## Limitações

O projeto simplifica diversos pontos de um modelo profissional. Retornos históricos não determinam retornos futuros; o DCF é sensível às premissas; e o Monte Carlo assume distribuição e dinâmica simplificadas. As limitações estão detalhadas em `docs/METHODOLOGY.md`.

## Próximos passos

- CAPM e construção completa do WACC;
- cenários de receita, margem e reinvestimento;
- análise de múltiplos comparáveis;
- GARCH e volatilidade condicional;
- bootstrap;
- stress testing;
- otimização de portfólio;
- geração de investment memo.

## Autor

**Caio Vinicius Paes da Silva**  
Sistemas de Informação — UNICAMP · Data Science — FIAP · Dados & Automação — Itaú