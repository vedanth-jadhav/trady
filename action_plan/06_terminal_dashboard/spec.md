# Phase 6: Terminal Dashboard

## Objective

Build a rich terminal-based dashboard (htop-style) for monitoring backtest results, viewing signals, and analyzing performance in real-time.

---

## Scope

| Item | Details |
|------|---------|
| Framework | Rich / Textual (Python TUI libraries) |
| Style | htop-inspired, multi-panel layout |
| Features | Live metrics, signal feed, position view, trade history |
| Mode | Backtest review (primary), Live monitoring (future) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      TERMINAL DASHBOARD                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    HEADER BAR                             │   │
│  │  TRADY v1.0 | Mode: Backtest | Capital: $12,450 (+24.5%) │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────┐  ┌─────────────────────────────────┐   │
│  │   METRICS PANEL     │  │        SIGNAL FEED              │   │
│  │                     │  │                                 │   │
│  │  Win Rate: 67%      │  │  [HIGH] 0x3f... BTC@100k Yes   │   │
│  │  Sharpe: 1.8        │  │  [MED]  0x7a... Election No    │   │
│  │  Max DD: 12%        │  │  [HIGH] 0x9c... ETH ETF Yes    │   │
│  │  P&L: +$2,450       │  │                                 │   │
│  │                     │  │                                 │   │
│  └─────────────────────┘  └─────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────┐  ┌─────────────────────────────────┐   │
│  │  POSITIONS PANEL    │  │       EQUITY CURVE              │   │
│  │                     │  │                                 │   │
│  │  Market   Dir  P&L  │  │    $12k ┤        ╭──────        │   │
│  │  BTC@100k Yes +$120 │  │         │    ╭───╯              │   │
│  │  Election No  -$45  │  │    $10k ┼────╯                  │   │
│  │  ETF      Yes +$200 │  │         │                       │   │
│  │                     │  │         └────────────────────   │   │
│  └─────────────────────┘  └─────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    TRADE HISTORY                          │   │
│  │  Time       Market          Dir  Entry  Exit   P&L  Tier │   │
│  │  12:30:45   BTC@100k        Yes  0.45   1.00  +$55  HIGH │   │
│  │  12:28:12   ETH Merge       No   0.30   0.00  -$30  MED  │   │
│  │  12:25:00   Fed Rate        Yes  0.60   1.00  +$40  HIGH │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  [Q]uit  [R]efresh  [F]ilter  [S]ort  [D]etails  [H]elp │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Technology Choice

### Textual Framework

```python
"""
Using Textual for rich TUI experience.

Textual provides:
- CSS-like styling
- Reactive components
- Built-in widgets (tables, trees, buttons)
- Async support
- Mouse and keyboard input
"""

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static, DataTable
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
```

---

## Component Specifications

### 1. Main Application

```python
class TradyDashboard(App):
    """
    Main Trady dashboard application.
    """

    CSS = """
    Screen {
        layout: grid;
        grid-size: 2 3;
        grid-gutter: 1;
    }

    #header {
        column-span: 2;
        height: 3;
        background: $primary;
    }

    #metrics {
        height: 100%;
    }

    #signals {
        height: 100%;
    }

    #positions {
        height: 100%;
    }

    #chart {
        height: 100%;
    }

    #history {
        column-span: 2;
        height: 100%;
    }

    .panel {
        border: solid $primary;
        padding: 1;
    }

    .panel-title {
        text-style: bold;
        color: $text;
    }

    .positive {
        color: green;
    }

    .negative {
        color: red;
    }

    .high-confidence {
        color: $success;
        text-style: bold;
    }

    .medium-confidence {
        color: $warning;
    }

    .low-confidence {
        color: $text-muted;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh"),
        ("f", "filter", "Filter"),
        ("s", "sort", "Sort"),
        ("d", "details", "Details"),
        ("h", "help", "Help"),
        ("1", "view_metrics", "Metrics"),
        ("2", "view_signals", "Signals"),
        ("3", "view_history", "History"),
    ]

    def __init__(self, backtest_result: BacktestResult):
        super().__init__()
        self.result = backtest_result
        self.current_view = "overview"

    def compose(self) -> ComposeResult:
        """Create dashboard layout."""
        yield Header(show_clock=True)
        yield HeaderPanel(self.result)

        with Horizontal():
            yield MetricsPanel(self.result, id="metrics")
            yield SignalFeed(self.result, id="signals")

        with Horizontal():
            yield PositionsPanel(self.result, id="positions")
            yield EquityCurve(self.result, id="chart")

        yield TradeHistory(self.result, id="history")
        yield Footer()

    def action_refresh(self):
        """Refresh all panels."""
        self.query_one("#metrics").refresh_data()
        self.query_one("#signals").refresh_data()
        self.query_one("#positions").refresh_data()
        self.query_one("#history").refresh_data()

    def action_filter(self):
        """Open filter dialog."""
        self.push_screen(FilterScreen())

    def action_details(self):
        """Show details for selected item."""
        history = self.query_one("#history", TradeHistory)
        if history.selected_trade:
            self.push_screen(TradeDetailScreen(history.selected_trade))
```

---

### 2. Header Panel

```python
class HeaderPanel(Static):
    """
    Top header showing key stats.
    """

    def __init__(self, result: BacktestResult, **kwargs):
        super().__init__(**kwargs)
        self.result = result

    def compose(self) -> ComposeResult:
        m = self.result.metrics
        portfolio = self.result.portfolio

        return_pct = m["total_return"] * 100
        return_class = "positive" if return_pct >= 0 else "negative"
        return_sign = "+" if return_pct >= 0 else ""

        yield Static(f"""
[bold]TRADY[/bold] v1.0 | Mode: Backtest Review
Capital: [bold]${portfolio.total_capital:,.2f}[/bold] ([{return_class}]{return_sign}{return_pct:.1f}%[/])
Win Rate: {m['win_rate']*100:.1f}% | Trades: {m['n_trades']} | Sharpe: {m['sharpe_ratio']:.2f}
        """.strip())
```

---

### 3. Metrics Panel

```python
class MetricsPanel(Static):
    """
    Panel showing key performance metrics.
    """

    def __init__(self, result: BacktestResult, **kwargs):
        super().__init__(**kwargs)
        self.result = result

    def compose(self) -> ComposeResult:
        yield Static("[bold]PERFORMANCE METRICS[/bold]", classes="panel-title")

        m = self.result.metrics

        # Create metrics table
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="dim")
        table.add_column("Value", justify="right")

        # Returns
        return_pct = m["total_return"] * 100
        return_style = "green" if return_pct >= 0 else "red"
        table.add_row("Total Return", f"[{return_style}]{return_pct:+.2f}%[/]")

        pnl = m["total_pnl"]
        pnl_style = "green" if pnl >= 0 else "red"
        table.add_row("Total P&L", f"[{pnl_style}]${pnl:+,.2f}[/]")

        # Risk
        table.add_row("Sharpe Ratio", f"{m['sharpe_ratio']:.2f}")
        table.add_row("Sortino Ratio", f"{m.get('sortino_ratio', 0):.2f}")

        dd = m["max_drawdown"] * 100
        dd_style = "red" if dd > 15 else "yellow" if dd > 10 else "green"
        table.add_row("Max Drawdown", f"[{dd_style}]{dd:.1f}%[/]")

        # Trade stats
        table.add_row("", "")  # Spacer
        table.add_row("Win Rate", f"{m['win_rate']*100:.1f}%")
        table.add_row("Profit Factor", f"{m.get('profit_factor', 0):.2f}")
        table.add_row("Expectancy", f"${m.get('expectancy', 0):.2f}")

        table.add_row("", "")
        table.add_row("Avg Win", f"[green]${m.get('avg_win', 0):.2f}[/]")
        table.add_row("Avg Loss", f"[red]${m.get('avg_loss', 0):.2f}[/]")

        yield Static(table)

    def refresh_data(self):
        """Refresh metrics display."""
        self.refresh()
```

---

### 4. Signal Feed

```python
class SignalFeed(Static):
    """
    Real-time feed of detected signals.
    """

    def __init__(self, result: BacktestResult, **kwargs):
        super().__init__(**kwargs)
        self.result = result
        self.signals = self._extract_signals()

    def _extract_signals(self) -> List[Dict]:
        """Extract signals from closed trades."""
        signals = []
        for trade in self.result.closed_trades[-50:]:  # Last 50
            signals.append({
                "timestamp": trade.timestamp,
                "market_id": trade.market_id[:20],
                "direction": trade.direction,
                "confidence": trade.signal.confidence,
                "tier": trade.signal.confidence_tier,
                "wallet": trade.source_wallet[:8],
                "won": trade.won,
            })
        return signals

    def compose(self) -> ComposeResult:
        yield Static("[bold]SIGNAL FEED[/bold]", classes="panel-title")

        table = DataTable(id="signal-table")
        table.add_column("Time", width=8)
        table.add_column("Tier", width=6)
        table.add_column("Wallet", width=10)
        table.add_column("Market", width=20)
        table.add_column("Dir", width=4)
        table.add_column("Result", width=6)

        for sig in reversed(self.signals[-20:]):  # Show last 20
            tier_style = self._get_tier_style(sig["tier"])
            result_style = "green" if sig["won"] else "red"
            result_text = "WIN" if sig["won"] else "LOSS"

            table.add_row(
                sig["timestamp"].strftime("%H:%M"),
                Text(sig["tier"][:4].upper(), style=tier_style),
                sig["wallet"] + "...",
                sig["market_id"][:18] + "..",
                sig["direction"],
                Text(result_text, style=result_style),
            )

        yield table

    def _get_tier_style(self, tier: str) -> str:
        styles = {
            "very_high": "bold green",
            "high": "green",
            "medium": "yellow",
            "low": "dim",
        }
        return styles.get(tier, "white")

    def refresh_data(self):
        self.refresh()
```

---

### 5. Positions Panel

```python
class PositionsPanel(Static):
    """
    Current open positions display.
    """

    def __init__(self, result: BacktestResult, **kwargs):
        super().__init__(**kwargs)
        self.result = result

    def compose(self) -> ComposeResult:
        yield Static("[bold]OPEN POSITIONS[/bold]", classes="panel-title")

        # In backtest review mode, show final state
        # For live mode, would show current positions
        positions = list(self.result.portfolio.positions.values())

        if not positions:
            yield Static("[dim]No open positions[/dim]")
            return

        table = DataTable()
        table.add_column("Market", width=25)
        table.add_column("Dir", width=5)
        table.add_column("Size", width=10)
        table.add_column("Entry", width=8)
        table.add_column("Current", width=8)
        table.add_column("P&L", width=10)

        for pos in positions:
            # Calculate unrealized P&L
            current_price = 0.5  # Placeholder
            unrealized = (current_price - pos.avg_entry_price) * pos.shares
            pnl_style = "green" if unrealized >= 0 else "red"

            table.add_row(
                pos.market_id[:23],
                pos.direction,
                f"${pos.notional:.0f}",
                f"{pos.avg_entry_price:.2f}",
                f"{current_price:.2f}",
                Text(f"${unrealized:+.0f}", style=pnl_style),
            )

        yield table
```

---

### 6. Equity Curve

```python
class EquityCurve(Static):
    """
    ASCII equity curve visualization.
    """

    def __init__(self, result: BacktestResult, **kwargs):
        super().__init__(**kwargs)
        self.result = result

    def compose(self) -> ComposeResult:
        yield Static("[bold]EQUITY CURVE[/bold]", classes="panel-title")

        # Get equity data
        history = self.result.portfolio_history
        if len(history) < 2:
            yield Static("[dim]Insufficient data[/dim]")
            return

        capitals = [h["total_capital"] for h in history]

        # Generate ASCII chart
        chart = self._generate_ascii_chart(capitals, width=40, height=10)
        yield Static(chart)

    def _generate_ascii_chart(
        self,
        values: List[float],
        width: int = 40,
        height: int = 10
    ) -> str:
        """Generate ASCII line chart."""
        if not values:
            return ""

        min_val = min(values)
        max_val = max(values)
        val_range = max_val - min_val or 1

        # Resample to width
        step = max(1, len(values) // width)
        sampled = values[::step][:width]

        # Build chart
        lines = []

        for row in range(height):
            threshold = max_val - (row / height) * val_range
            line = ""
            for val in sampled:
                if val >= threshold:
                    line += "█"
                else:
                    line += " "
            lines.append(f"│{line}")

        # Add axis labels
        lines.insert(0, f"${max_val:,.0f}")
        lines.append(f"${min_val:,.0f}")
        lines.append("└" + "─" * width)

        return "\n".join(lines)
```

---

### 7. Trade History

```python
class TradeHistory(Static):
    """
    Scrollable trade history table.
    """

    selected_trade = reactive(None)

    def __init__(self, result: BacktestResult, **kwargs):
        super().__init__(**kwargs)
        self.result = result
        self.trades = result.closed_trades
        self.sort_key = "timestamp"
        self.sort_reverse = True
        self.filter_tier = None

    def compose(self) -> ComposeResult:
        yield Static("[bold]TRADE HISTORY[/bold]", classes="panel-title")

        table = DataTable(id="trade-table", cursor_type="row")
        table.add_column("Time", key="timestamp", width=12)
        table.add_column("Market", key="market", width=25)
        table.add_column("Dir", key="direction", width=5)
        table.add_column("Entry", key="entry", width=8)
        table.add_column("Exit", key="exit", width=8)
        table.add_column("Size", key="size", width=10)
        table.add_column("P&L", key="pnl", width=10)
        table.add_column("Return", key="return", width=8)
        table.add_column("Tier", key="tier", width=6)
        table.add_column("Result", key="result", width=6)

        # Sort trades
        sorted_trades = sorted(
            self.trades,
            key=lambda t: getattr(t, self.sort_key, t.timestamp),
            reverse=self.sort_reverse
        )

        # Apply filter
        if self.filter_tier:
            sorted_trades = [
                t for t in sorted_trades
                if t.signal.confidence_tier == self.filter_tier
            ]

        for trade in sorted_trades[:100]:  # Limit for performance
            pnl_style = "green" if trade.pnl >= 0 else "red"
            result_style = "green bold" if trade.won else "red"

            table.add_row(
                trade.timestamp.strftime("%m/%d %H:%M"),
                trade.market_id[:23],
                trade.direction,
                f"{trade.entry_price:.2f}",
                f"{trade.exit_price:.2f}" if trade.exit_price else "-",
                f"${trade.notional:.0f}",
                Text(f"${trade.pnl:+.0f}", style=pnl_style),
                Text(f"{trade.return_pct*100:+.1f}%", style=pnl_style),
                trade.signal.confidence_tier[:4].upper(),
                Text("WIN" if trade.won else "LOSS", style=result_style),
                key=trade.trade_id,
            )

        yield table

    def on_data_table_row_selected(self, event: DataTable.RowSelected):
        """Handle row selection."""
        trade_id = event.row_key.value
        self.selected_trade = next(
            (t for t in self.trades if t.trade_id == trade_id),
            None
        )

    def sort_by(self, key: str):
        """Sort table by column."""
        if self.sort_key == key:
            self.sort_reverse = not self.sort_reverse
        else:
            self.sort_key = key
            self.sort_reverse = True
        self.refresh()

    def filter_by_tier(self, tier: Optional[str]):
        """Filter by confidence tier."""
        self.filter_tier = tier
        self.refresh()
```

---

### 8. Trade Detail Screen

```python
class TradeDetailScreen(Screen):
    """
    Detailed view of a single trade.
    """

    BINDINGS = [
        ("escape", "dismiss", "Back"),
        ("q", "dismiss", "Back"),
    ]

    def __init__(self, trade: ExecutedTrade, **kwargs):
        super().__init__(**kwargs)
        self.trade = trade

    def compose(self) -> ComposeResult:
        yield Header()

        with Container():
            yield Static(f"""
[bold]TRADE DETAILS[/bold]

[bold]Trade ID:[/bold] {self.trade.trade_id}

[bold]Market[/bold]
  ID: {self.trade.market_id}
  Direction: {self.trade.direction}

[bold]Execution[/bold]
  Entry Time: {self.trade.timestamp}
  Exit Time: {self.trade.exit_timestamp}
  Hold Duration: {self._hold_duration()}

[bold]Pricing[/bold]
  Entry Price: {self.trade.entry_price:.4f}
  Exit Price: {self.trade.exit_price:.4f}
  Shares: {self.trade.shares:.2f}

[bold]P&L[/bold]
  Notional: ${self.trade.notional:.2f}
  P&L: ${self.trade.pnl:+.2f}
  Return: {self.trade.return_pct*100:+.2f}%
  Result: {'WIN' if self.trade.won else 'LOSS'}

[bold]Signal[/bold]
  Confidence: {self.trade.signal.confidence:.3f}
  Tier: {self.trade.signal.confidence_tier}
  Source Wallet: {self.trade.source_wallet}

[bold]Top Features[/bold]
{self._format_features()}
            """)

        yield Footer()

    def _hold_duration(self) -> str:
        if not self.trade.exit_timestamp:
            return "N/A"
        delta = self.trade.exit_timestamp - self.trade.timestamp
        return f"{delta.days}d {delta.seconds // 3600}h"

    def _format_features(self) -> str:
        features = getattr(self.trade.signal, "top_features", [])
        if not features:
            return "  No feature data"
        lines = []
        for name, value in features[:5]:
            lines.append(f"  {name}: {value:.3f}")
        return "\n".join(lines)
```

---

### 9. Filter Screen

```python
class FilterScreen(Screen):
    """
    Filter configuration screen.
    """

    BINDINGS = [
        ("escape", "dismiss", "Cancel"),
        ("enter", "apply", "Apply"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()

        with Container():
            yield Static("[bold]FILTER OPTIONS[/bold]\n")

            yield Static("Confidence Tier:")
            yield RadioSet(
                RadioButton("All", id="all", value=True),
                RadioButton("Very High", id="very_high"),
                RadioButton("High", id="high"),
                RadioButton("Medium", id="medium"),
                RadioButton("Low", id="low"),
            )

            yield Static("\nResult:")
            yield RadioSet(
                RadioButton("All", id="result_all", value=True),
                RadioButton("Wins Only", id="wins"),
                RadioButton("Losses Only", id="losses"),
            )

            yield Static("\nDate Range:")
            yield Input(placeholder="Start date (YYYY-MM-DD)", id="start_date")
            yield Input(placeholder="End date (YYYY-MM-DD)", id="end_date")

            yield Button("Apply Filters", variant="primary", id="apply")
            yield Button("Clear Filters", variant="warning", id="clear")

        yield Footer()

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "apply":
            self.apply_filters()
        elif event.button.id == "clear":
            self.clear_filters()

    def apply_filters(self):
        # Get filter values and pass to main app
        self.dismiss(self.get_filter_config())

    def clear_filters(self):
        self.dismiss(None)
```

---

### 10. Help Screen

```python
class HelpScreen(Screen):
    """
    Help and keyboard shortcuts screen.
    """

    BINDINGS = [
        ("escape", "dismiss", "Close"),
        ("q", "dismiss", "Close"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()

        yield Static("""
[bold]TRADY DASHBOARD HELP[/bold]

[bold]Navigation[/bold]
  Tab / Shift+Tab    Move between panels
  Arrow keys         Navigate within panels
  Enter              Select item / Show details

[bold]Actions[/bold]
  Q                  Quit application
  R                  Refresh all data
  F                  Open filter dialog
  S                  Sort current table
  D                  Show details for selected item
  H                  Show this help screen

[bold]Views[/bold]
  1                  Focus metrics panel
  2                  Focus signals panel
  3                  Focus trade history

[bold]Trade History[/bold]
  Click column       Sort by column
  Enter on row       Show trade details

[bold]Signal Feed[/bold]
  Scroll             View more signals
  Enter              Show signal source

[bold]Equity Curve[/bold]
  No interactions    Display only

Press ESC or Q to close this help.
        """)

        yield Footer()
```

---

## Main Entry Point

```python
# src/dashboard/main.py

import typer
from pathlib import Path

app = typer.Typer()

@app.command()
def run(
    backtest_dir: Path = typer.Argument(..., help="Path to backtest results"),
    theme: str = typer.Option("dark", help="Color theme (dark/light)")
):
    """
    Launch Trady terminal dashboard.
    """
    # Load backtest results
    from src.backtest.engine import BacktestResult

    result = BacktestResult.load(backtest_dir)

    # Create and run app
    dashboard = TradyDashboard(result)

    if theme == "light":
        dashboard.dark = False

    dashboard.run()


@app.command()
def export_html(
    backtest_dir: Path,
    output_file: Path = Path("report.html")
):
    """
    Export dashboard as static HTML report.
    """
    from src.dashboard.export import HTMLExporter

    result = BacktestResult.load(backtest_dir)
    exporter = HTMLExporter(result)
    exporter.export(output_file)
    print(f"Report exported to {output_file}")


if __name__ == "__main__":
    app()
```

---

## CLI Interface

```bash
# Launch dashboard
python -m src.dashboard.main run ./backtest_results/

# With options
python -m src.dashboard.main run ./backtest_results/ --theme light

# Export to HTML
python -m src.dashboard.main export-html ./backtest_results/ -o report.html
```

---

## Dependencies

```python
# Additional requirements for Phase 6

textual>=0.47.0       # TUI framework
rich>=13.7.0          # Rich text rendering
plotext>=5.2.0        # ASCII plotting (optional)
```

---

## Success Criteria

- [ ] Dashboard launches without errors
- [ ] All panels render correctly
- [ ] Keyboard navigation works
- [ ] Trade details screen shows full info
- [ ] Filtering works correctly
- [ ] Sorting works on all columns
- [ ] Performance: <100ms render time
- [ ] Help screen documents all shortcuts

---

## Future Enhancements

For live trading mode (post-MVP):

1. **Live Signal Feed**: WebSocket connection for real-time signals
2. **Position Updates**: Live P&L updates
3. **Alert System**: Desktop notifications for high-confidence signals
4. **Trade Execution**: One-click trade confirmation
5. **Market Depth**: Show order book for selected market

---

## Complete Project Structure

After all phases:

```
polymarket/
├── action_plan/
│   ├── 00_overview.md
│   ├── 01_data_ingestion/spec.md
│   ├── 02_wallet_analysis/spec.md
│   ├── 03_signal_detection/spec.md
│   ├── 04_ml_scoring/spec.md
│   ├── 05_backtest_engine/spec.md
│   └── 06_terminal_dashboard/spec.md
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── client.py          # Polymarket API client
│   │   ├── fetcher.py         # Data fetching logic
│   │   ├── storage.py         # Parquet I/O
│   │   └── cli.py             # Data CLI
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── freshness.py       # Freshness analyzer
│   │   ├── funding.py         # Funding tracker
│   │   ├── behavior.py        # Behavior profiler
│   │   ├── clustering.py      # Wallet clusterer
│   │   └── cli.py             # Analysis CLI
│   ├── signals/
│   │   ├── __init__.py
│   │   ├── detectors.py       # Signal detectors
│   │   ├── aggregator.py      # Signal aggregation
│   │   ├── filters.py         # Negative filters
│   │   └── cli.py             # Signals CLI
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── features.py        # Feature engineering
│   │   ├── labeling.py        # Ground truth labeling
│   │   ├── model.py           # XGBoost model
│   │   ├── scoring.py         # Scoring service
│   │   └── cli.py             # ML CLI
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── config.py          # Backtest config
│   │   ├── engine.py          # Main engine
│   │   ├── portfolio.py       # Portfolio tracking
│   │   ├── metrics.py         # Metrics calculation
│   │   ├── report.py          # Report generation
│   │   └── cli.py             # Backtest CLI
│   └── dashboard/
│       ├── __init__.py
│       ├── app.py             # Main Textual app
│       ├── panels.py          # Panel widgets
│       ├── screens.py         # Modal screens
│       ├── styles.css         # Textual CSS
│       └── main.py            # Entry point
├── data/
│   ├── raw/
│   ├── processed/
│   └── models/
├── tests/
│   ├── test_data/
│   ├── test_analysis/
│   ├── test_signals/
│   ├── test_ml/
│   └── test_backtest/
├── config/
│   └── settings.yaml
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

*Phase 6 Complete - Full Specification Done*

**Total Development Phases: 6**
**Estimated Files: ~40 Python modules**
**Core Features: Data ingestion, wallet analysis, signal detection, ML scoring, backtesting, terminal dashboard**
