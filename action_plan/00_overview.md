# Trady - Polymarket Insider Detection Bot

## Project Overview

**Trady** is a Python-based trading bot that detects and follows potential insider activity on Polymarket prediction markets. The bot identifies suspicious trading patterns—fresh wallets, unusual sizing, timing anomalies, and funding patterns—to generate high-confidence signals for automated trade execution.

---

## Core Philosophy

> "If an insider knows an event will happen, they won't exit at 90%—they'll ride it to 100%."

The bot follows this insight: true insiders don't take early profits. They enter positions at favorable odds and hold to resolution. Trady detects these patterns and mirrors their conviction.

---

## Target Scope

| Attribute | Value |
|-----------|-------|
| Platform | Polymarket only |
| Language | Python |
| Data Source | Polymarket Official API (`/trades`, `/markets`) |
| Storage | Parquet files |
| Deployment | Railway (future), Local (MVP) |
| Interface | Terminal dashboard (htop-style) |
| MVP Focus | **Backtest-only** (no live trading) |

---

## Development Phases

```
Phase 1: Data Ingestion          → Scrape & store market + trade data
Phase 2: Wallet Analysis         → Profile wallets, cluster related addresses
Phase 3: Signal Detection        → Identify insider-like patterns
Phase 4: ML Scoring              → XGBoost ensemble for confidence scoring
Phase 5: Backtest Engine         → Simulate strategy on historical data
Phase 6: Terminal Dashboard      → Real-time monitoring interface
```

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         TRADY ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Polymarket │───▶│    Data      │───▶│   Parquet    │       │
│  │     API      │    │   Ingestion  │    │   Storage    │       │
│  └──────────────┘    └──────────────┘    └──────┬───────┘       │
│                                                  │               │
│                                                  ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Wallet     │◀───│   Signal     │◀───│   Feature    │       │
│  │   Profiler   │    │   Detector   │    │   Engineer   │       │
│  └──────┬───────┘    └──────┬───────┘    └──────────────┘       │
│         │                   │                                    │
│         ▼                   ▼                                    │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Wallet     │    │   XGBoost    │───▶│   Backtest   │       │
│  │   Clusters   │    │   Scorer     │    │    Engine    │       │
│  └──────────────┘    └──────────────┘    └──────┬───────┘       │
│                                                  │               │
│                                                  ▼               │
│                                          ┌──────────────┐       │
│                                          │   Terminal   │       │
│                                          │   Dashboard  │       │
│                                          └──────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Signal Framework

### Positive Signals (Increase Insider Likelihood)

#### Wallet Freshness (Multi-Factor)
- Zero transaction history (brand new wallet)
- No prior Polymarket activity
- Recently funded (funding arrived shortly before trade)
- Combination scoring across all factors

#### Trade Sizing (Dynamic)
- Size relative to market's typical volume (% of market)
- Not absolute thresholds—adapts to market liquidity

#### Timing Patterns
- Pre-news timing (trades before announcements)
- Off-hours activity (when retail is less active)
- Before price spikes (trades preceding unusual movement)
- Rapid succession trades (multiple trades in quick bursts)

#### Funding Source (Prioritized)
| Priority | Source | Weight |
|----------|--------|--------|
| Critical | Privacy tools (Tornado Cash, etc.) | 1.0 |
| High | From known tracked wallets | 0.8 |
| Medium-High | Cross-chain bridge | 0.6 |
| Medium | CEX withdrawals | 0.4 |

### Negative Signals (Decrease Insider Likelihood)

- Long trading history on Polymarket
- Known identity (public figures, funds)
- Retail-like behavior (many small diverse bets)
- Frequent early exits before resolution

---

## Wallet Clustering Methods

To detect insiders splitting activity across multiple wallets:

1. **Funding Source Clustering**: Wallets funded from the same source
2. **Behavioral Clustering**: Wallets trading at similar times/patterns
3. **Position Correlation**: Wallets with correlated position changes

---

## Risk Management Framework

### Multi-Layered Limits
- **Portfolio Limit**: Hard cap on total capital at risk
- **Per-Market Limit**: Maximum exposure per individual market
- **Per-Wallet Limit**: Cap positions following any single wallet

### Position Sizing
- Confidence-weighted (based on signal score)
- Risk-adjusted (Kelly criterion or similar)

### Conflict Resolution
When two insider-like wallets bet opposite directions:
→ **Bet on higher confidence signal**

### Market Focus
- Prioritize high-insider categories (political, corporate)
- Adaptive thresholds by market context

---

## Backtest Requirements

### Data Scope
- **Duration**: 90 days historical
- **Markets**: ~100 markets (mix of low/medium/high volume)
- **Filter**: Active markets only

### Ground Truth Labeling (Hybrid)
1. Profitable early bets (traded at low probability, market resolved in their favor)
2. Manual pattern review
3. High-signal clustering analysis

### Output Metrics
- Performance metrics (P&L, ROI, returns curves)
- Signal-by-signal breakdown (each signal's profitability)
- Feature importance (which signals contributed most)
- Trade visualization (charts of trades vs. market movement)

---

## File Structure

```
polymarket/
├── action_plan/
│   ├── 00_overview.md              # This file
│   ├── 01_data_ingestion/
│   │   └── spec.md                 # Data scraping & storage
│   ├── 02_wallet_analysis/
│   │   └── spec.md                 # Wallet profiling & clustering
│   ├── 03_signal_detection/
│   │   └── spec.md                 # Pattern detection logic
│   ├── 04_ml_scoring/
│   │   └── spec.md                 # XGBoost ensemble training
│   ├── 05_backtest_engine/
│   │   └── spec.md                 # Historical simulation
│   └── 06_terminal_dashboard/
│       └── spec.md                 # Monitoring interface
├── src/
│   ├── data/                       # Data ingestion modules
│   ├── analysis/                   # Wallet analysis
│   ├── signals/                    # Signal detection
│   ├── ml/                         # ML scoring
│   ├── backtest/                   # Backtest engine
│   └── dashboard/                  # Terminal UI
├── data/
│   ├── raw/                        # Raw API responses
│   ├── processed/                  # Parquet files
│   └── models/                     # Trained models
├── tests/
├── config/
└── requirements.txt
```

---

## API Endpoints Used

### Markets
```
GET /markets
GET /markets/{market_id}
```

### Trades
```
GET /trades?market={market_id}&limit=500&offset=0
GET /trades?user={wallet_address}&limit=500&offset=0
```

---

## Success Criteria

### Backtest MVP
- [ ] Successfully scrape 90 days of data for 100 markets
- [ ] Profile all unique wallets with activity
- [ ] Cluster related wallets with >80% accuracy
- [ ] Generate insider likelihood scores
- [ ] Run backtest with complete metrics output
- [ ] Terminal dashboard showing backtest results

### Performance Targets
- Win rate: >60% on high-confidence signals
- Sharpe ratio: >1.5
- Max drawdown: <20%
- Signal precision: >70% (true positives vs false positives)

---

## Next Steps

1. **Read Phase 1 spec**: `01_data_ingestion/spec.md`
2. Implement data ingestion pipeline
3. Progress through each phase sequentially
4. Each phase builds on the previous

---

*Spec Version: 1.0*
*Created: January 2025*
*Project: Trady*
