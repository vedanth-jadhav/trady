# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

**Trady** is a Python trading bot that detects and follows potential insider activity on Polymarket prediction markets. It identifies suspicious trading patterns (fresh wallets, unusual sizing, timing anomalies, funding patterns) to generate signals for automated trade execution.

Core philosophy: *"If an insider knows an event will happen, they won't exit at 90%—they'll ride it to 100%."*

**Current Status**: Phases 1 (Data Ingestion), 2 (Wallet Analysis), and 3 (Signal Detection) are implemented. `src/ml/` also exists with a train/score CLI and a saved model artifact, but it should be treated as experimental rather than fully validated. Phases 5-6 (Backtest Engine, Terminal Dashboard) are still planned.

**Reality Check**:
- The repository currently has heuristic signal detection and wallet profiling running end-to-end.
- The repository also has an ML scoring pipeline (`trady-ml`) that can train and score trades, but the project does not currently store a clean, validated benchmark report for real-world insider-detection accuracy.
- Existing `final_score` / insider-likelihood outputs are heuristic or model scores, not proof of achieved detection accuracy.
- Do not claim a confirmed production accuracy, precision, recall, or win rate unless you recompute and show the evaluation from current data.

## Commands

### Installation
```bash
pip install -e ".[dev]"
```

### Running Tests
```bash
# Run all tests
pytest

# Run a single test file
pytest tests/test_data.py

# Run with verbose output
pytest -v

# Run a specific test
pytest tests/test_data.py::TestWalletIndexer::test_build_index_basic
```

### Linting and Formatting
```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/
```

### CLI Tools

**Data Ingestion** (`trady-data`):
```bash
# Run complete data ingestion pipeline
trady-data run-pipeline --count 100 --days 90

# Individual steps
trady-data fetch-markets
trady-data select-markets --count 100
trady-data fetch-trades --days 90
trady-data build-wallet-index

# Validate stored data
trady-data validate

# Show data info
trady-data info
```

**Wallet Analysis** (`trady-analysis`):
```bash
# Run complete analysis pipeline
trady-analysis run-analysis

# Individual steps
trady-analysis analyze-freshness
trady-analysis analyze-behavior
trady-analysis cluster-wallets
trady-analysis build-profiles
```

**Signal Detection** (`trady-signals`):
```bash
# Run complete signal detection pipeline
trady-signals run-pipeline

# Individual steps
trady-signals detect-signals --min-score 0.3
trady-signals show-suspicious --min-score 0.5 --top 20
trady-signals analyze-signal <trade_id>
```

**ML Scoring** (`trady-ml`):
```bash
# Train the experimental ML model
trady-ml train

# Score trades with the saved model
trady-ml score --high-signal

# Show top suspicious wallets from scored trades
trady-ml suspects --top 20 --min-trades 2
```

## Architecture

### Development Phases

```
Phase 1: Data Ingestion    → src/data/      [IMPLEMENTED]
Phase 2: Wallet Analysis   → src/analysis/  [IMPLEMENTED]
Phase 3: Signal Detection  → src/signals/   [IMPLEMENTED]
Phase 4: ML Scoring        → src/ml/        [EXPERIMENTAL]
Phase 5: Backtest Engine   → src/backtest/  [PLANNED]
Phase 6: Terminal Dashboard→ src/dashboard/ [PLANNED]
```

### Key Modules

**`src/data/`** - Data ingestion from Polymarket API:
- `client.py`: Async API client with rate limiting (RateLimiter, PolymarketClient)
- `fetcher.py`: Market selection and trade fetching (MarketSelector, TradeFetcher, WalletIndexer)
- `storage.py`: Parquet file I/O with consistent schemas (DataStorage)
- `validator.py`: Data quality validation (DataValidator)
- `cli.py`: Typer CLI for data commands

**`src/analysis/`** - Wallet profiling and clustering:
- `freshness.py`: Wallet freshness scoring (FreshnessAnalyzer)
- `behavior.py`: Trading behavior profiling (BehaviorProfiler)
- `clusterer.py`: Wallet clustering for coordinated activity (WalletClusterer)
- `funding.py`: Funding source tracking (FundingTracker)
- `profiler.py`: Complete wallet profile builder (WalletProfileBuilder, NegativeSignalDetector)
- `types.py`: Dataclasses for analysis types (BehaviorProfile, etc.)
- `cli.py`: Typer CLI for analysis commands

**`src/signals/`** - Signal detection for insider activity:
- `types.py`: Signal dataclasses (Signal, AggregatedSignal, Trade, Market, TradeSignal)
- `freshness.py`: Freshness signal detector (FreshnessSignalDetector)
- `timing.py`: Timing signal detector (TimingSignalDetector)
- `sizing.py`: Sizing signal detector (SizingSignalDetector)
- `funding.py`: Funding signal detector (FundingSignalDetector)
- `cluster.py`: Cluster signal detector (ClusterSignalDetector)
- `market_context.py`: Market context analyzer (MarketContextAnalyzer)
- `negative_filter.py`: Negative signal filter (NegativeSignalFilter)
- `aggregator.py`: Signal aggregator (SignalAggregator)
- `detector.py`: Main orchestrator (InsiderSignalDetector)
- `cli.py`: Typer CLI for signal commands

**`src/ml/`** - Experimental ML scoring pipeline:
- `features.py`: Feature engineering for trade, wallet, market, and signal data
- `labels.py`: Hybrid ground-truth label generation from resolved/proxy outcomes
- `model.py`: XGBoost ensemble with calibration and evaluation helpers
- `service.py`: Trade scoring service and confidence tiers
- `overrides.py`: Rule-based overrides for edge cases
- `cli.py`: Typer CLI for ML train/score/suspects commands

### Data Flow

```
Polymarket API → Data Ingestion → Parquet Storage → Wallet Analysis → Signal Detection
                      ↓                   ↓                ↓                ↓
               selected_markets.parquet   trades.parquet   wallet_profiles  signals.parquet
               wallets.parquet                             .parquet
```

### Storage Schema

Data is stored in `data/processed/` as Parquet files:
- `all_markets.parquet` / `selected_markets.parquet`: Market metadata with volume tiers
- `trades.parquet`: Trade history (trade_id, market_id, timestamp, maker/taker addresses, side, size, price, notional)
- `wallets.parquet`: Wallet index (address, first_seen, last_seen, total_trades, total_volume, is_whale)
- `wallet_profiles.parquet`: Complete analysis profiles with insider scores
- `signals.parquet`: Trade-level signals with confidence scores and category breakdowns
- `scored_trades.parquet`: Experimental ML-scored trade outputs
- `suspicious_wallets.parquet`: Summary of wallets with high insider likelihood

## Accuracy Notes

- The planning docs mention aspirational targets such as `>80%` cluster accuracy, `>70%` signal precision, and `>60%` win rate on high-confidence signals, but these are targets rather than confirmed achieved results.
- The ML model code can compute `auc_roc`, `precision`, `recall`, `f1`, `log_loss`, and `brier_score`, but the current training CLI does not persist a benchmark report automatically.
- If asked about "accuracy", verify whether the question refers to:
  - heuristic insider-likelihood scores
  - ML classification metrics on generated labels
  - trading win rate from a backtest
- Treat those as different things and avoid conflating them.

## Insider Detection Signals

The system combines multiple signals to score insider likelihood:

**Positive Signals** (increase suspicion):
- Wallet freshness (new wallets, new to Polymarket, recently funded)
- Trade sizing (relative to market volume and wallet history)
- Timing patterns (pre-news, off-hours, before price spikes)
- Funding source (privacy tools > tracked wallets > bridges > CEX)
- Wallet clusters (coordinated activity across multiple wallets)

**Negative Signals** (decrease suspicion):
- Long trading history
- Known identity/public entity
- Retail-like behavior (many small diverse bets)
- Frequent early exits

## API Details

Uses Polymarket Gamma API (`gamma-api.polymarket.com`) for market/event data with volume information, and Data API for trade history. Key endpoints:
- `/events` - Fetch events with nested markets
- `/markets` - Individual market details
- `/trades` - Trade history by market or user
