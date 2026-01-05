# Phase 5: Backtest Engine

## Objective

Build a comprehensive backtesting engine that simulates the insider-following trading strategy on historical data, producing detailed performance metrics and trade-by-trade analysis.

---

## Scope

| Item | Details |
|------|---------|
| Input | Scored trades from Phase 4, market resolution data |
| Output | Performance metrics, trade log, visualizations |
| Strategy | Follow high-confidence insider signals, hold to resolution |
| Execution | Virtual trades (paper trading simulation) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      BACKTEST ENGINE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Scored     │───▶│   Signal     │───▶│   Position   │       │
│  │   Trades     │    │   Filter     │    │   Manager    │       │
│  └──────────────┘    └──────────────┘    └──────┬───────┘       │
│                                                  │               │
│                                                  ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Risk       │◀───│   Execution  │◀───│   Order      │       │
│  │   Manager    │    │   Simulator  │    │   Generator  │       │
│  └──────┬───────┘    └──────────────┘    └──────────────┘       │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Portfolio  │───▶│   Resolution │───▶│   Metrics    │       │
│  │   Tracker    │    │   Handler    │    │   Calculator │       │
│  └──────────────┘    └──────────────┘    └──────┬───────┘       │
│                                                  │               │
│                                                  ▼               │
│                                          ┌──────────────┐       │
│                                          │   Report     │       │
│                                          │   Generator  │       │
│                                          └──────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Backtest Configuration

```python
@dataclass
class BacktestConfig:
    """
    Configuration for backtest run.
    """
    # Capital
    initial_capital: float = 10_000.0
    max_capital_per_trade: float = 500.0

    # Signal thresholds
    min_confidence_score: float = 0.6
    confidence_tiers_allowed: List[str] = field(
        default_factory=lambda: ["very_high", "high"]
    )

    # Position sizing
    sizing_method: str = "confidence_weighted"  # or "fixed", "kelly"
    base_position_size: float = 100.0
    confidence_multiplier: float = 5.0  # Max 5x base for very_high

    # Risk limits
    max_portfolio_exposure: float = 0.8  # 80% of capital max
    max_position_per_market: float = 1000.0
    max_positions_per_wallet: float = 2000.0
    max_concurrent_positions: int = 50

    # Execution
    slippage_bps: float = 50.0  # 0.5% slippage assumption
    fee_bps: float = 0.0  # Polymarket has no fees

    # Strategy
    hold_to_resolution: bool = True  # Core strategy assumption
    exit_on_high_prob: float = None  # Optional: exit at 95%

    # Time
    start_date: datetime = None
    end_date: datetime = None
```

---

### 2. Signal Filter

```python
class SignalFilter:
    """
    Filters scored trades to generate trading signals.
    """

    def __init__(self, config: BacktestConfig):
        self.config = config

    def filter_signals(
        self,
        scored_trades: List[ScoredTrade]
    ) -> List[TradingSignal]:
        """
        Filter scored trades into actionable trading signals.

        Criteria:
        - Score >= min_confidence_score
        - Confidence tier in allowed tiers
        - Market is still active (not resolved)
        - Within time window
        """
        signals = []

        for scored in scored_trades:
            # Score threshold
            if scored.final_score < self.config.min_confidence_score:
                continue

            # Confidence tier
            if scored.confidence_tier not in self.config.confidence_tiers_allowed:
                continue

            # Time window
            if self.config.start_date and scored.trade.timestamp < self.config.start_date:
                continue
            if self.config.end_date and scored.trade.timestamp > self.config.end_date:
                continue

            signals.append(TradingSignal(
                timestamp=scored.trade.timestamp,
                market_id=scored.trade.market_id,
                direction=self._get_direction(scored.trade),
                confidence=scored.final_score,
                confidence_tier=scored.confidence_tier,
                source_trade=scored.trade,
                source_wallet=scored.trade.wallet,
            ))

        return sorted(signals, key=lambda x: x.timestamp)

    def _get_direction(self, trade: Trade) -> str:
        """
        Determine direction to follow.

        If insider bought Yes, we buy Yes.
        If insider sold Yes (bought No), we buy No.
        """
        if trade.side == "BUY":
            return trade.outcome  # "Yes" or "No"
        else:
            # Selling Yes = Buying No
            return "No" if trade.outcome == "Yes" else "Yes"


@dataclass
class TradingSignal:
    timestamp: datetime
    market_id: str
    direction: str  # "Yes" or "No"
    confidence: float
    confidence_tier: str
    source_trade: Trade
    source_wallet: str
```

---

### 3. Position Sizer

```python
class PositionSizer:
    """
    Determines position size based on configuration and portfolio state.
    """

    def __init__(self, config: BacktestConfig):
        self.config = config

    def calculate_size(
        self,
        signal: TradingSignal,
        current_portfolio: Portfolio,
        market_price: float
    ) -> float:
        """
        Calculate position size for a signal.

        Methods:
        - fixed: Always use base_position_size
        - confidence_weighted: Scale by confidence
        - kelly: Kelly criterion based on confidence as edge estimate
        """
        available_capital = self._get_available_capital(current_portfolio)

        if self.config.sizing_method == "fixed":
            raw_size = self.config.base_position_size

        elif self.config.sizing_method == "confidence_weighted":
            # Scale from 1x to multiplier based on confidence
            multiplier = 1 + (signal.confidence - 0.5) * 2 * (
                self.config.confidence_multiplier - 1
            )
            raw_size = self.config.base_position_size * multiplier

        elif self.config.sizing_method == "kelly":
            # Estimate edge from confidence
            # Assume confidence roughly equals win probability
            win_prob = signal.confidence
            # Odds based on current price
            odds = (1 - market_price) / market_price if market_price < 1 else 1
            kelly_fraction = (win_prob * odds - (1 - win_prob)) / odds
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
            raw_size = available_capital * kelly_fraction

        else:
            raise ValueError(f"Unknown sizing method: {self.config.sizing_method}")

        # Apply limits
        size = min(
            raw_size,
            self.config.max_capital_per_trade,
            available_capital,
            self._get_market_limit(signal.market_id, current_portfolio),
            self._get_wallet_limit(signal.source_wallet, current_portfolio),
        )

        return max(0, size)

    def _get_available_capital(self, portfolio: Portfolio) -> float:
        """Calculate available capital after exposure limits."""
        max_exposure = portfolio.total_capital * self.config.max_portfolio_exposure
        current_exposure = portfolio.total_position_value
        return max(0, max_exposure - current_exposure)

    def _get_market_limit(self, market_id: str, portfolio: Portfolio) -> float:
        """Get remaining limit for this market."""
        current = portfolio.get_market_exposure(market_id)
        return max(0, self.config.max_position_per_market - current)

    def _get_wallet_limit(self, wallet: str, portfolio: Portfolio) -> float:
        """Get remaining limit for positions following this wallet."""
        current = portfolio.get_wallet_exposure(wallet)
        return max(0, self.config.max_positions_per_wallet - current)
```

---

### 4. Execution Simulator

```python
class ExecutionSimulator:
    """
    Simulates trade execution with realistic assumptions.
    """

    def __init__(
        self,
        config: BacktestConfig,
        markets_df: pd.DataFrame,
        trades_df: pd.DataFrame
    ):
        self.config = config
        self.markets = markets_df.set_index("market_id")
        self.trades = trades_df
        self._build_price_series()

    def _build_price_series(self):
        """
        Build price time series for each market.

        Used to determine execution prices.
        """
        self.price_series = {}
        for market_id, group in self.trades.groupby("market_id"):
            # Build OHLC from trades
            group = group.sort_values("timestamp")
            self.price_series[market_id] = group[["timestamp", "price"]].set_index("timestamp")

    def execute(
        self,
        signal: TradingSignal,
        size: float,
        portfolio: Portfolio
    ) -> Optional[ExecutedTrade]:
        """
        Simulate execution of a trade.

        Applies:
        - Slippage
        - Fees
        - Price at signal time
        """
        if size <= 0:
            return None

        # Get price at signal time
        base_price = self._get_price_at_time(
            signal.market_id,
            signal.direction,
            signal.timestamp
        )

        if base_price is None:
            return None

        # Apply slippage (adverse movement)
        slippage = self.config.slippage_bps / 10000
        exec_price = base_price * (1 + slippage)  # Buying at slightly higher
        exec_price = min(exec_price, 0.99)  # Cap at 99 cents

        # Calculate shares
        shares = size / exec_price

        # Apply fees
        fees = size * (self.config.fee_bps / 10000)

        return ExecutedTrade(
            trade_id=f"bt_{signal.market_id}_{signal.timestamp.timestamp()}",
            timestamp=signal.timestamp,
            market_id=signal.market_id,
            direction=signal.direction,
            shares=shares,
            entry_price=exec_price,
            notional=size,
            fees=fees,
            signal=signal,
            source_wallet=signal.source_wallet,
        )

    def _get_price_at_time(
        self,
        market_id: str,
        direction: str,
        timestamp: datetime
    ) -> Optional[float]:
        """Get market price at specific time."""
        series = self.price_series.get(market_id)
        if series is None or len(series) == 0:
            return None

        # Find closest price before or at timestamp
        mask = series.index <= timestamp
        if not mask.any():
            return series.iloc[0]["price"]

        return series[mask].iloc[-1]["price"]


@dataclass
class ExecutedTrade:
    trade_id: str
    timestamp: datetime
    market_id: str
    direction: str
    shares: float
    entry_price: float
    notional: float
    fees: float
    signal: TradingSignal
    source_wallet: str

    # Filled after resolution
    exit_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    pnl: Optional[float] = None
    return_pct: Optional[float] = None
    won: Optional[bool] = None
```

---

### 5. Portfolio Tracker

```python
class Portfolio:
    """
    Tracks portfolio state during backtest.
    """

    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.total_capital = initial_capital
        self.positions: Dict[str, Position] = {}  # market_id -> Position
        self.closed_trades: List[ExecutedTrade] = []
        self.trade_log: List[Dict] = []

        # Tracking by wallet
        self.wallet_exposure: Dict[str, float] = defaultdict(float)

    @property
    def total_position_value(self) -> float:
        """Total value of open positions at cost."""
        return sum(p.notional for p in self.positions.values())

    @property
    def n_positions(self) -> int:
        """Number of open positions."""
        return len(self.positions)

    def open_position(self, trade: ExecutedTrade):
        """Open a new position from executed trade."""
        if trade.market_id in self.positions:
            # Add to existing position
            pos = self.positions[trade.market_id]
            pos.add(trade)
        else:
            # New position
            self.positions[trade.market_id] = Position.from_trade(trade)

        # Update cash and exposure
        self.cash -= (trade.notional + trade.fees)
        self.wallet_exposure[trade.source_wallet] += trade.notional

        # Log
        self.trade_log.append({
            "action": "open",
            "timestamp": trade.timestamp,
            "market_id": trade.market_id,
            "direction": trade.direction,
            "shares": trade.shares,
            "price": trade.entry_price,
            "notional": trade.notional,
            "cash_after": self.cash,
            "n_positions": self.n_positions,
        })

    def close_position(
        self,
        market_id: str,
        exit_price: float,
        exit_timestamp: datetime,
        resolution: str
    ) -> Optional[ExecutedTrade]:
        """Close position on market resolution."""
        if market_id not in self.positions:
            return None

        position = self.positions.pop(market_id)

        # Determine if won
        won = (position.direction == resolution)

        # Calculate PnL
        if won:
            exit_value = position.shares * 1.0  # Resolution pays $1 per share
        else:
            exit_value = 0.0

        pnl = exit_value - position.notional - position.total_fees
        return_pct = pnl / position.notional if position.notional > 0 else 0

        # Update position with exit info
        closed_trade = ExecutedTrade(
            trade_id=position.trade_id,
            timestamp=position.entry_timestamp,
            market_id=market_id,
            direction=position.direction,
            shares=position.shares,
            entry_price=position.avg_entry_price,
            notional=position.notional,
            fees=position.total_fees,
            signal=position.signal,
            source_wallet=position.source_wallet,
            exit_price=exit_price,
            exit_timestamp=exit_timestamp,
            pnl=pnl,
            return_pct=return_pct,
            won=won,
        )

        # Update cash
        self.cash += exit_value
        self.total_capital = self.cash + self.total_position_value

        # Update wallet exposure
        self.wallet_exposure[position.source_wallet] -= position.notional

        self.closed_trades.append(closed_trade)

        # Log
        self.trade_log.append({
            "action": "close",
            "timestamp": exit_timestamp,
            "market_id": market_id,
            "direction": position.direction,
            "resolution": resolution,
            "won": won,
            "pnl": pnl,
            "return_pct": return_pct,
            "cash_after": self.cash,
            "total_capital": self.total_capital,
        })

        return closed_trade

    def get_market_exposure(self, market_id: str) -> float:
        """Get current exposure in a market."""
        if market_id in self.positions:
            return self.positions[market_id].notional
        return 0.0

    def get_wallet_exposure(self, wallet: str) -> float:
        """Get total exposure following a wallet."""
        return self.wallet_exposure.get(wallet, 0.0)

    def get_snapshot(self, timestamp: datetime) -> Dict:
        """Get portfolio snapshot at point in time."""
        return {
            "timestamp": timestamp,
            "cash": self.cash,
            "position_value": self.total_position_value,
            "total_capital": self.total_capital,
            "n_positions": self.n_positions,
            "return_pct": (self.total_capital - self.initial_capital) / self.initial_capital,
        }


@dataclass
class Position:
    trade_id: str
    market_id: str
    direction: str
    shares: float
    notional: float
    avg_entry_price: float
    total_fees: float
    entry_timestamp: datetime
    signal: TradingSignal
    source_wallet: str

    @classmethod
    def from_trade(cls, trade: ExecutedTrade) -> "Position":
        return cls(
            trade_id=trade.trade_id,
            market_id=trade.market_id,
            direction=trade.direction,
            shares=trade.shares,
            notional=trade.notional,
            avg_entry_price=trade.entry_price,
            total_fees=trade.fees,
            entry_timestamp=trade.timestamp,
            signal=trade.signal,
            source_wallet=trade.source_wallet,
        )

    def add(self, trade: ExecutedTrade):
        """Add to position (averaging in)."""
        total_notional = self.notional + trade.notional
        self.avg_entry_price = (
            (self.notional * self.avg_entry_price + trade.notional * trade.entry_price)
            / total_notional
        )
        self.shares += trade.shares
        self.notional = total_notional
        self.total_fees += trade.fees
```

---

### 6. Resolution Handler

```python
class ResolutionHandler:
    """
    Handles market resolution events.
    """

    def __init__(self, markets_df: pd.DataFrame):
        self.markets = markets_df
        self._build_resolution_timeline()

    def _build_resolution_timeline(self):
        """Build timeline of resolution events."""
        resolved = self.markets[self.markets["resolved"] == True].copy()
        resolved["resolution_timestamp"] = pd.to_datetime(resolved["resolution_timestamp"])
        self.resolution_events = resolved.sort_values("resolution_timestamp").to_dict("records")

    def get_resolutions_in_window(
        self,
        start: datetime,
        end: datetime
    ) -> List[Dict]:
        """Get all resolutions in time window."""
        return [
            e for e in self.resolution_events
            if start <= e["resolution_timestamp"] <= end
        ]

    def get_resolution(self, market_id: str) -> Optional[Dict]:
        """Get resolution for specific market."""
        market = self.markets[self.markets["market_id"] == market_id]
        if len(market) == 0 or not market.iloc[0]["resolved"]:
            return None

        row = market.iloc[0]
        return {
            "market_id": market_id,
            "resolution": row["resolution"],
            "resolution_timestamp": row["resolution_timestamp"],
        }
```

---

### 7. Backtest Engine

```python
class BacktestEngine:
    """
    Main backtest orchestrator.
    """

    def __init__(
        self,
        config: BacktestConfig,
        scored_trades: List[ScoredTrade],
        markets_df: pd.DataFrame,
        trades_df: pd.DataFrame
    ):
        self.config = config
        self.scored_trades = scored_trades
        self.markets = markets_df
        self.trades = trades_df

        # Initialize components
        self.signal_filter = SignalFilter(config)
        self.sizer = PositionSizer(config)
        self.executor = ExecutionSimulator(config, markets_df, trades_df)
        self.resolution_handler = ResolutionHandler(markets_df)

        # State
        self.portfolio = None
        self.signals_processed = 0
        self.trades_executed = 0

    def run(self, progress_callback: Callable = None) -> BacktestResult:
        """
        Run full backtest.
        """
        # Initialize
        self.portfolio = Portfolio(self.config.initial_capital)
        portfolio_history = []

        # Filter and sort signals
        signals = self.signal_filter.filter_signals(self.scored_trades)

        # Build event timeline (signals + resolutions)
        events = self._build_event_timeline(signals)

        # Process events chronologically
        for i, event in enumerate(events):
            if event["type"] == "signal":
                self._process_signal(event["data"])
            elif event["type"] == "resolution":
                self._process_resolution(event["data"])

            # Record portfolio snapshot
            portfolio_history.append(
                self.portfolio.get_snapshot(event["timestamp"])
            )

            if progress_callback and i % 100 == 0:
                progress_callback(i, len(events))

        # Close any remaining positions (unresolved markets)
        self._close_unresolved_positions()

        # Calculate metrics
        metrics = MetricsCalculator(self.portfolio, self.config).calculate()

        return BacktestResult(
            config=self.config,
            portfolio=self.portfolio,
            portfolio_history=portfolio_history,
            metrics=metrics,
            closed_trades=self.portfolio.closed_trades,
            signals_processed=self.signals_processed,
            trades_executed=self.trades_executed,
        )

    def _build_event_timeline(self, signals: List[TradingSignal]) -> List[Dict]:
        """Build chronological event timeline."""
        events = []

        # Add signals
        for signal in signals:
            events.append({
                "type": "signal",
                "timestamp": signal.timestamp,
                "data": signal,
            })

        # Add resolutions
        for res in self.resolution_handler.resolution_events:
            events.append({
                "type": "resolution",
                "timestamp": res["resolution_timestamp"],
                "data": res,
            })

        return sorted(events, key=lambda x: x["timestamp"])

    def _process_signal(self, signal: TradingSignal):
        """Process a trading signal."""
        self.signals_processed += 1

        # Check position limits
        if self.portfolio.n_positions >= self.config.max_concurrent_positions:
            return

        # Check if market already has position
        if signal.market_id in self.portfolio.positions:
            return  # Already have position, skip

        # Get price
        price = self.executor._get_price_at_time(
            signal.market_id, signal.direction, signal.timestamp
        )
        if price is None:
            return

        # Calculate size
        size = self.sizer.calculate_size(signal, self.portfolio, price)
        if size <= 0:
            return

        # Execute
        trade = self.executor.execute(signal, size, self.portfolio)
        if trade:
            self.portfolio.open_position(trade)
            self.trades_executed += 1

    def _process_resolution(self, resolution: Dict):
        """Process a market resolution."""
        market_id = resolution["market_id"]

        if market_id in self.portfolio.positions:
            self.portfolio.close_position(
                market_id=market_id,
                exit_price=1.0 if resolution["resolution"] else 0.0,
                exit_timestamp=resolution["resolution_timestamp"],
                resolution=resolution["resolution"]
            )

    def _close_unresolved_positions(self):
        """Close positions in unresolved markets at current price."""
        for market_id in list(self.portfolio.positions.keys()):
            # Get last known price
            series = self.executor.price_series.get(market_id)
            if series is not None and len(series) > 0:
                last_price = series.iloc[-1]["price"]
            else:
                last_price = 0.5  # Assume 50/50

            # Mark as unresolved loss
            self.portfolio.close_position(
                market_id=market_id,
                exit_price=last_price,
                exit_timestamp=datetime.now(),
                resolution="UNRESOLVED"
            )


@dataclass
class BacktestResult:
    config: BacktestConfig
    portfolio: Portfolio
    portfolio_history: List[Dict]
    metrics: Dict
    closed_trades: List[ExecutedTrade]
    signals_processed: int
    trades_executed: int
```

---

### 8. Metrics Calculator

```python
class MetricsCalculator:
    """
    Calculates comprehensive performance metrics.
    """

    def __init__(self, portfolio: Portfolio, config: BacktestConfig):
        self.portfolio = portfolio
        self.config = config

    def calculate(self) -> Dict:
        """Calculate all metrics."""
        trades = self.portfolio.closed_trades

        if not trades:
            return self._empty_metrics()

        return {
            # Returns
            "total_return": self._total_return(),
            "total_pnl": self._total_pnl(trades),
            "annualized_return": self._annualized_return(),

            # Win/Loss
            "win_rate": self._win_rate(trades),
            "n_wins": sum(1 for t in trades if t.won),
            "n_losses": sum(1 for t in trades if not t.won),
            "n_trades": len(trades),

            # Risk metrics
            "sharpe_ratio": self._sharpe_ratio(),
            "sortino_ratio": self._sortino_ratio(),
            "max_drawdown": self._max_drawdown(),
            "calmar_ratio": self._calmar_ratio(),

            # Trade metrics
            "avg_win": self._avg_win(trades),
            "avg_loss": self._avg_loss(trades),
            "profit_factor": self._profit_factor(trades),
            "expectancy": self._expectancy(trades),

            # Position metrics
            "avg_position_size": self._avg_position_size(trades),
            "avg_hold_time_days": self._avg_hold_time(trades),
            "max_concurrent_positions": self._max_concurrent(),

            # Signal metrics
            "signal_precision": self._signal_precision(trades),
            "signals_per_day": self._signals_per_day(),

            # By confidence tier
            "metrics_by_tier": self._metrics_by_tier(trades),

            # By market category
            "metrics_by_category": self._metrics_by_category(trades),
        }

    def _total_return(self) -> float:
        """Total percentage return."""
        return (
            (self.portfolio.total_capital - self.portfolio.initial_capital)
            / self.portfolio.initial_capital
        )

    def _total_pnl(self, trades: List[ExecutedTrade]) -> float:
        """Total P&L in dollars."""
        return sum(t.pnl for t in trades if t.pnl is not None)

    def _win_rate(self, trades: List[ExecutedTrade]) -> float:
        """Percentage of winning trades."""
        if not trades:
            return 0
        wins = sum(1 for t in trades if t.won)
        return wins / len(trades)

    def _sharpe_ratio(self) -> float:
        """Sharpe ratio from daily returns."""
        daily_returns = self._get_daily_returns()
        if len(daily_returns) < 2:
            return 0

        mean_return = np.mean(daily_returns)
        std_return = np.std(daily_returns)

        if std_return == 0:
            return 0

        return (mean_return / std_return) * np.sqrt(252)  # Annualized

    def _sortino_ratio(self) -> float:
        """Sortino ratio (downside risk only)."""
        daily_returns = self._get_daily_returns()
        if len(daily_returns) < 2:
            return 0

        mean_return = np.mean(daily_returns)
        downside = daily_returns[daily_returns < 0]
        downside_std = np.std(downside) if len(downside) > 0 else 1

        if downside_std == 0:
            return 0

        return (mean_return / downside_std) * np.sqrt(252)

    def _max_drawdown(self) -> float:
        """Maximum peak-to-trough drawdown."""
        capitals = [h["total_capital"] for h in self.portfolio.trade_log if "total_capital" in h]
        if not capitals:
            return 0

        peak = capitals[0]
        max_dd = 0

        for capital in capitals:
            if capital > peak:
                peak = capital
            dd = (peak - capital) / peak
            max_dd = max(max_dd, dd)

        return max_dd

    def _profit_factor(self, trades: List[ExecutedTrade]) -> float:
        """Gross profit / Gross loss."""
        gross_profit = sum(t.pnl for t in trades if t.pnl and t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl and t.pnl < 0))

        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0

        return gross_profit / gross_loss

    def _expectancy(self, trades: List[ExecutedTrade]) -> float:
        """Expected value per trade."""
        if not trades:
            return 0
        return sum(t.pnl for t in trades if t.pnl) / len(trades)

    def _avg_win(self, trades: List[ExecutedTrade]) -> float:
        """Average winning trade P&L."""
        wins = [t.pnl for t in trades if t.won and t.pnl]
        return np.mean(wins) if wins else 0

    def _avg_loss(self, trades: List[ExecutedTrade]) -> float:
        """Average losing trade P&L (negative)."""
        losses = [t.pnl for t in trades if not t.won and t.pnl]
        return np.mean(losses) if losses else 0

    def _metrics_by_tier(self, trades: List[ExecutedTrade]) -> Dict:
        """Break down metrics by confidence tier."""
        tiers = {}
        for tier in ["very_high", "high", "medium", "low"]:
            tier_trades = [t for t in trades if t.signal.confidence_tier == tier]
            if tier_trades:
                tiers[tier] = {
                    "n_trades": len(tier_trades),
                    "win_rate": self._win_rate(tier_trades),
                    "total_pnl": self._total_pnl(tier_trades),
                    "avg_return": np.mean([t.return_pct for t in tier_trades if t.return_pct]),
                }
        return tiers

    def _get_daily_returns(self) -> np.ndarray:
        """Calculate daily returns from portfolio history."""
        if len(self.portfolio.trade_log) < 2:
            return np.array([])

        capitals = []
        for entry in self.portfolio.trade_log:
            if "total_capital" in entry:
                capitals.append({
                    "date": entry["timestamp"].date(),
                    "capital": entry["total_capital"]
                })

        df = pd.DataFrame(capitals)
        daily = df.groupby("date")["capital"].last().pct_change().dropna()
        return daily.values

    def _empty_metrics(self) -> Dict:
        """Return empty metrics when no trades."""
        return {
            "total_return": 0,
            "total_pnl": 0,
            "win_rate": 0,
            "n_trades": 0,
            "sharpe_ratio": 0,
            "max_drawdown": 0,
        }
```

---

### 9. Report Generator

```python
class ReportGenerator:
    """
    Generates backtest reports and visualizations.
    """

    def __init__(self, result: BacktestResult):
        self.result = result

    def generate_summary(self) -> str:
        """Generate text summary."""
        m = self.result.metrics
        lines = [
            "=" * 60,
            "TRADY BACKTEST REPORT",
            "=" * 60,
            "",
            "CONFIGURATION",
            f"  Initial Capital: ${self.result.config.initial_capital:,.2f}",
            f"  Min Confidence: {self.result.config.min_confidence_score}",
            f"  Sizing Method: {self.result.config.sizing_method}",
            "",
            "PERFORMANCE",
            f"  Total Return: {m['total_return']*100:.2f}%",
            f"  Total P&L: ${m['total_pnl']:,.2f}",
            f"  Final Capital: ${self.result.portfolio.total_capital:,.2f}",
            "",
            "RISK METRICS",
            f"  Sharpe Ratio: {m['sharpe_ratio']:.2f}",
            f"  Sortino Ratio: {m.get('sortino_ratio', 0):.2f}",
            f"  Max Drawdown: {m['max_drawdown']*100:.2f}%",
            "",
            "TRADE STATISTICS",
            f"  Total Trades: {m['n_trades']}",
            f"  Win Rate: {m['win_rate']*100:.2f}%",
            f"  Profit Factor: {m.get('profit_factor', 0):.2f}",
            f"  Expectancy: ${m.get('expectancy', 0):.2f}",
            f"  Avg Win: ${m.get('avg_win', 0):.2f}",
            f"  Avg Loss: ${m.get('avg_loss', 0):.2f}",
            "",
            "SIGNAL STATISTICS",
            f"  Signals Processed: {self.result.signals_processed}",
            f"  Trades Executed: {self.result.trades_executed}",
            f"  Execution Rate: {self.result.trades_executed/max(1,self.result.signals_processed)*100:.1f}%",
            "",
            "BY CONFIDENCE TIER",
        ]

        for tier, stats in m.get("metrics_by_tier", {}).items():
            lines.append(
                f"  {tier}: {stats['n_trades']} trades, "
                f"{stats['win_rate']*100:.1f}% win, "
                f"${stats['total_pnl']:.2f} P&L"
            )

        lines.extend(["", "=" * 60])
        return "\n".join(lines)

    def generate_trade_log(self) -> pd.DataFrame:
        """Generate detailed trade log."""
        records = []
        for trade in self.result.closed_trades:
            records.append({
                "trade_id": trade.trade_id,
                "timestamp": trade.timestamp,
                "market_id": trade.market_id,
                "direction": trade.direction,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "shares": trade.shares,
                "notional": trade.notional,
                "pnl": trade.pnl,
                "return_pct": trade.return_pct,
                "won": trade.won,
                "confidence": trade.signal.confidence,
                "confidence_tier": trade.signal.confidence_tier,
                "source_wallet": trade.source_wallet,
                "hold_days": (trade.exit_timestamp - trade.timestamp).days if trade.exit_timestamp else None,
            })
        return pd.DataFrame(records)

    def generate_equity_curve_data(self) -> pd.DataFrame:
        """Generate equity curve data."""
        return pd.DataFrame(self.result.portfolio_history)

    def save_report(self, output_dir: Path):
        """Save all report artifacts."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Summary
        with open(output_dir / "summary.txt", "w") as f:
            f.write(self.generate_summary())

        # Trade log
        self.generate_trade_log().to_parquet(output_dir / "trade_log.parquet")

        # Equity curve
        self.generate_equity_curve_data().to_parquet(output_dir / "equity_curve.parquet")

        # Metrics JSON
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(self.result.metrics, f, indent=2, default=str)

        # Config
        with open(output_dir / "config.json", "w") as f:
            json.dump(asdict(self.result.config), f, indent=2, default=str)
```

---

## CLI Interface

```python
@app.command()
def run_backtest(
    scored_trades_file: Path,
    markets_file: Path,
    trades_file: Path,
    output_dir: Path,
    initial_capital: float = 10000,
    min_confidence: float = 0.6,
    sizing_method: str = "confidence_weighted"
):
    """Run backtest with specified configuration."""
    pass

@app.command()
def optimize_thresholds(
    scored_trades_file: Path,
    markets_file: Path,
    trades_file: Path,
    output_file: Path
):
    """Grid search over confidence thresholds."""
    pass

@app.command()
def compare_strategies(
    scored_trades_file: Path,
    markets_file: Path,
    trades_file: Path,
    output_dir: Path
):
    """Compare different strategy configurations."""
    pass

@app.command()
def analyze_trade(
    trade_id: str,
    backtest_dir: Path
):
    """Deep dive analysis on specific trade."""
    pass
```

---

## Output Files

```
backtest_results/
├── summary.txt              # Text summary
├── metrics.json             # All metrics as JSON
├── config.json              # Backtest configuration
├── trade_log.parquet        # All trades with details
├── equity_curve.parquet     # Portfolio value over time
├── signals_by_tier.parquet  # Breakdown by confidence
└── feature_analysis.parquet # Which features predicted wins
```

---

## Success Criteria

- [ ] Backtest runs on 90 days of data without errors
- [ ] Portfolio tracking accurately reflects P&L
- [ ] All metrics calculate correctly
- [ ] Win rate >60% on high-confidence signals
- [ ] Sharpe ratio >1.5
- [ ] Max drawdown <20%
- [ ] Report generation produces all artifacts

---

## Next Phase

After completing Phase 5, proceed to:
→ **Phase 6: Terminal Dashboard** (`06_terminal_dashboard/spec.md`)

The backtest results will be displayed in:
- Real-time performance monitoring
- Signal visualization
- Trade history browser
