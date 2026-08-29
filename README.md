# Trady

Research pipeline for finding unusual Polymarket wallet activity and testing whether those signals would have been useful historically.

It ingests public market and trade data, builds wallet profiles, groups related wallets, produces heuristic signals, and can train an XGBoost score from the resulting features. A high score means "worth investigating". It does not prove that a wallet belongs to an insider.

## Current status

**MVP, backtest-only.** Data ingestion, wallet analysis, clustering, rule-based signals, and ML scoring are implemented as local pipelines. The repository does not place orders, manage a live portfolio, or run a production trading service.

The backtest engine and terminal dashboard described in `action_plan/` are design specs, not completed modules. Treat any result as research output until it has been tested on clean out-of-sample data and reviewed for leakage.

## Pipeline

```text
Polymarket Gamma / Data API / Goldsky
                 |
                 v
      Parquet market + trade data
                 |
                 v
 freshness, behaviour, funding and wallet clustering
                 |
                 v
 rule-based trade signals and position-size suggestions
                 |
                 v
      optional XGBoost scoring
```

The default working directory is `data/processed/`. Intermediate datasets are kept as Parquet files so each stage can be inspected independently.

## Run

Requires Python 3.10 or newer.

```bash
git clone https://github.com/vedanth-jadhav/trady.git
cd trady
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Open the interactive CLI:

```bash
trady
```

Or run each stage directly:

```bash
trady-data --help
trady-analysis --help
trady-signals --help
trady-ml --help
```

A typical research run is:

```bash
trady-data run-pipeline
trady-analysis run-analysis
trady-signals run-pipeline
trady-ml train
trady-ml score
trady-ml suspects
```

The commands expose flags for market filters, date ranges, input paths, and output paths. Use each command's `--help` before a larger data pull.

## Verify

```bash
pip install -e '.[dev]'
pytest
```

## What is not built

- live order execution or exchange authentication
- a live portfolio, risk engine, or position reconciliation
- the backtest engine specified in `action_plan/05_backtest_engine/`
- the terminal dashboard specified in `action_plan/06_terminal_dashboard/`
- deployment, scheduling, monitoring, or alerting
- proof that a flagged wallet has non-public information

## Repository map

- `src/data/` - Polymarket clients, ingestion, validation, and Parquet storage
- `src/analysis/` - wallet freshness, behaviour, funding, profiling, and clustering
- `src/signals/` - signal detection, aggregation, filters, market context, and sizing
- `src/ml/` - features, labels, XGBoost model, overrides, and scoring CLI
- `tests/` - data, signal, and ML tests
- `action_plan/` - project design notes, including modules that are not implemented yet

This is experimental trading research, not financial advice.
