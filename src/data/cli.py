"""
CLI interface for data ingestion pipeline.

Commands:
- fetch-markets: Fetch and save all markets
- select-markets: Select markets for analysis
- fetch-trades: Fetch trades for selected markets
- build-wallet-index: Build wallet index from trades
- run-pipeline: Run complete data ingestion pipeline
- validate: Validate stored data
"""

import asyncio
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional
import logging

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn

from .client import PolymarketClient
from .fetcher import MarketSelector, TradeFetcher, WalletIndexer, GoldskyTradeFetcher
from .storage import DataStorage
from .validator import DataValidator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# CLI app
app = typer.Typer(
    name="trady-data",
    help="Trady data ingestion pipeline for Polymarket",
    add_completion=False,
)

console = Console()


def run_async(coro):
    """Run async coroutine in sync context."""
    return asyncio.run(coro)


@app.command()
def fetch_markets(
    output_dir: Path = typer.Option(
        Path("data/processed"),
        "--output", "-o",
        help="Output directory for Parquet files"
    ),
    min_volume: float = typer.Option(
        0,
        "--min-volume",
        help="Minimum volume filter"
    ),
):
    """Fetch and save all markets from Polymarket API."""
    console.print("[bold blue]Fetching markets from Polymarket API...[/bold blue]")

    storage = DataStorage(output_dir)

    async def _fetch():
        async with PolymarketClient() as client:
            selector = MarketSelector(client)
            markets = await selector.fetch_all_markets()
            return markets

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Fetching markets...", total=None)
        markets = run_async(_fetch())

    if markets.empty:
        console.print("[red]No markets fetched![/red]")
        raise typer.Exit(1)

    # Filter by volume
    if min_volume > 0:
        markets = markets[markets["volume"] >= min_volume]
        console.print(f"Filtered to {len(markets)} markets with volume >= {min_volume}")

    # Save
    path = storage.save_markets(markets, "all_markets.parquet")
    console.print(f"[green]✓ Saved {len(markets)} markets to {path}[/green]")

    # Show summary
    table = Table(title="Markets Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Total Markets", str(len(markets)))
    table.add_row("Active Markets", str(markets["is_active"].sum()))
    table.add_row("Resolved Markets", str(markets["is_resolved"].sum()))
    table.add_row("Total Volume", f"${markets['volume'].sum():,.0f}")

    if "category" in markets.columns:
        top_categories = markets["category"].value_counts().head(5)
        for cat, count in top_categories.items():
            table.add_row(f"Category: {cat}", str(count))

    console.print(table)


@app.command()
def select_markets(
    n_markets: int = typer.Option(
        100,
        "--count", "-n",
        help="Number of markets to select"
    ),
    output_dir: Path = typer.Option(
        Path("data/processed"),
        "--output", "-o",
        help="Output directory"
    ),
    include_resolved: bool = typer.Option(
        True,
        "--include-resolved/--no-resolved",
        help="Include resolved markets"
    ),
    min_volume: float = typer.Option(
        100,
        "--min-volume",
        help="Minimum volume threshold"
    ),
):
    """Select markets for analysis with balanced volume distribution."""
    console.print(f"[bold blue]Selecting {n_markets} markets for analysis...[/bold blue]")

    storage = DataStorage(output_dir)

    # Load markets
    markets = storage.load_markets("all_markets.parquet")
    if markets.empty:
        console.print("[yellow]No markets file found. Fetching from API...[/yellow]")

        async def _fetch():
            async with PolymarketClient() as client:
                selector = MarketSelector(client)
                return await selector.fetch_all_markets()

        markets = run_async(_fetch())
        storage.save_markets(markets, "all_markets.parquet")

    # Select markets
    async def _select():
        async with PolymarketClient() as client:
            selector = MarketSelector(client)
            return selector.select_markets(
                markets,
                n_total=n_markets,
                include_resolved=include_resolved,
                min_volume=min_volume,
            )

    selected = run_async(_select())

    if selected.empty:
        console.print("[red]No markets selected![/red]")
        raise typer.Exit(1)

    # Save
    path = storage.save_markets(selected, "selected_markets.parquet")
    console.print(f"[green]✓ Selected {len(selected)} markets, saved to {path}[/green]")

    # Show distribution
    table = Table(title="Selection Distribution")
    table.add_column("Volume Tier", style="cyan")
    table.add_column("Count", style="magenta")
    table.add_column("Avg Volume", style="green")

    for tier in ["low", "medium", "high", "resolved"]:
        tier_df = selected[selected["volume_tier"] == tier]
        if not tier_df.empty:
            table.add_row(
                tier.capitalize(),
                str(len(tier_df)),
                f"${tier_df['volume'].mean():,.0f}"
            )

    console.print(table)


@app.command()
def fetch_trades(
    market_file: Path = typer.Option(
        Path("data/processed/selected_markets.parquet"),
        "--markets", "-m",
        help="Path to selected markets file"
    ),
    output_dir: Path = typer.Option(
        Path("data/processed"),
        "--output", "-o",
        help="Output directory"
    ),
    lookback_days: int = typer.Option(
        90,
        "--days", "-d",
        help="Number of days to look back"
    ),
    max_concurrent: int = typer.Option(
        5,
        "--concurrent", "-c",
        help="Maximum concurrent API requests"
    ),
):
    """Fetch trades for selected markets."""
    console.print(f"[bold blue]Fetching trades for last {lookback_days} days...[/bold blue]")

    storage = DataStorage(output_dir)

    # Load selected markets
    if not market_file.exists():
        console.print(f"[red]Markets file not found: {market_file}[/red]")
        console.print("Run 'select-markets' first.")
        raise typer.Exit(1)

    markets = storage.load_markets(market_file.name)
    market_ids = markets["market_id"].tolist()
    console.print(f"Found {len(market_ids)} markets to fetch trades for")

    # Use Rich progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        trades_task = progress.add_task(
            f"[green]Fetching trades from {len(market_ids)} markets...",
            total=len(market_ids)
        )

        def progress_callback(current: int, total: int):
            progress.update(trades_task, completed=current)

        async def _fetch():
            async with PolymarketClient() as client:
                fetcher = TradeFetcher(client, lookback_days=lookback_days)
                return await fetcher.fetch_all_trades(
                    market_ids,
                    max_concurrent=max_concurrent,
                    progress_callback=progress_callback,
                )

        trades = run_async(_fetch())

    if trades.empty:
        console.print("[yellow]No trades fetched![/yellow]")
        return

    # Save
    path = storage.save_trades(trades)
    console.print(f"[green]✓ Saved {len(trades):,} trades to {path}[/green]")

    # Show summary
    table = Table(title="Trades Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Total Trades", f"{len(trades):,}")
    table.add_row("Unique Markets", str(trades["market_id"].nunique()))
    table.add_row("Unique Wallets (Makers)", str(trades["maker_address"].nunique()))
    table.add_row("Unique Wallets (Takers)", str(trades["taker_address"].nunique()))
    table.add_row("Total Volume", f"${trades['notional'].sum():,.0f}")
    table.add_row("Avg Trade Size", f"${trades['notional'].mean():,.2f}")

    if "timestamp" in trades.columns:
        table.add_row("Date Range", f"{trades['timestamp'].min().date()} to {trades['timestamp'].max().date()}")

    console.print(table)


@app.command()
def fetch_trades_goldsky(
    output_dir: Path = typer.Option(
        Path("data/processed"),
        "--output", "-o",
        help="Output directory"
    ),
    lookback_days: int = typer.Option(
        7,
        "--days", "-d",
        help="Number of days to look back"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Show detailed logging"
    ),
):
    """Fetch trades from Goldsky blockchain indexer (full historical data)."""
    # Suppress logging unless verbose
    if not verbose:
        logging.getLogger("src.data.goldsky").setLevel(logging.WARNING)
        logging.getLogger("src.data.fetcher").setLevel(logging.WARNING)

    console.print(f"[bold blue]Fetching trades from Goldsky for last {lookback_days} days...[/bold blue]")
    console.print("[dim]Goldsky provides complete historical data unlike the Data API[/dim]")
    console.print()

    storage = DataStorage(output_dir)

    # Use Rich Live display for better progress
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text
    from datetime import datetime, timezone, timedelta

    start_time = datetime.now()
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    # Track progress state
    state = {"count": 0, "pages": 0}

    def create_display():
        elapsed = datetime.now() - start_time
        elapsed_str = str(elapsed).split('.')[0]  # Remove microseconds

        rate = state["count"] / max(elapsed.total_seconds(), 1)

        text = Text()
        text.append("Goldsky Trade Fetcher\n\n", style="bold blue")
        text.append(f"  Trades fetched: ", style="dim")
        text.append(f"{state['count']:,}\n", style="bold green")
        text.append(f"  Pages fetched:  ", style="dim")
        text.append(f"{state['pages']:,}\n", style="cyan")
        text.append(f"  Rate:           ", style="dim")
        text.append(f"{rate:,.0f} trades/sec\n", style="yellow")
        text.append(f"  Elapsed:        ", style="dim")
        text.append(f"{elapsed_str}\n", style="white")
        text.append(f"\n  Lookback:       ", style="dim")
        text.append(f"{lookback_days} days (since {cutoff.strftime('%Y-%m-%d')})", style="dim")

        return Panel(text, title="[bold]Progress[/bold]", border_style="blue")

    def progress_callback(count: int):
        state["count"] = count
        state["pages"] = count // 1000

    async def _fetch():
        fetcher = GoldskyTradeFetcher(lookback_days=lookback_days)
        return await fetcher.fetch_all_trades(progress_callback=progress_callback)

    # Run with live display
    with Live(create_display(), refresh_per_second=4, console=console) as live:
        import asyncio

        async def fetch_with_updates():
            task = asyncio.create_task(_fetch())
            while not task.done():
                live.update(create_display())
                await asyncio.sleep(0.25)
            return await task

        trades = asyncio.run(fetch_with_updates())

    if trades.empty:
        console.print("[yellow]No trades fetched![/yellow]")
        return

    # Save
    path = storage.save_trades(trades)
    console.print(f"[green]✓ Saved {len(trades):,} trades to {path}[/green]")

    # Show summary
    table = Table(title="Trades Summary (Goldsky)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Total Trades", f"{len(trades):,}")
    table.add_row("Unique Markets", str(trades["market_id"].nunique()))
    table.add_row("Unique Wallets (Makers)", str(trades["maker_address"].nunique()))
    table.add_row("Unique Wallets (Takers)", str(trades["taker_address"].nunique()))
    table.add_row("Total Volume", f"${trades['notional'].sum():,.0f}")
    table.add_row("Avg Trade Size", f"${trades['notional'].mean():,.2f}")

    if "timestamp" in trades.columns:
        table.add_row("Date Range (Goldsky)", f"{trades['timestamp'].min().date()} to {trades['timestamp'].max().date()}")

    console.print(table)


@app.command()
def build_wallet_index(
    trades_file: Path = typer.Option(
        Path("data/processed/trades.parquet"),
        "--trades", "-t",
        help="Path to trades file"
    ),
    output_dir: Path = typer.Option(
        Path("data/processed"),
        "--output", "-o",
        help="Output directory"
    ),
    whale_percentile: float = typer.Option(
        0.95,
        "--whale-percentile",
        help="Percentile threshold for whale identification"
    ),
):
    """Build wallet index from trades data."""
    console.print("[bold blue]Building wallet index...[/bold blue]")

    storage = DataStorage(output_dir)

    # Load trades
    if not trades_file.exists():
        console.print(f"[red]Trades file not found: {trades_file}[/red]")
        console.print("Run 'fetch-trades' first.")
        raise typer.Exit(1)

    trades = storage.load_trades(trades_file.name)

    # Build index
    indexer = WalletIndexer(whale_percentile=whale_percentile)
    wallets = indexer.build_index(trades)

    if wallets.empty:
        console.print("[yellow]No wallets indexed![/yellow]")
        return

    # Save
    path = storage.save_wallets(wallets)
    console.print(f"[green]✓ Saved {len(wallets)} wallets to {path}[/green]")

    # Show summary
    table = Table(title="Wallet Index Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Total Wallets", f"{len(wallets):,}")
    table.add_row("Whale Wallets", str(wallets["is_whale"].sum()))
    table.add_row("Total Volume", f"${wallets['total_volume'].sum():,.0f}")
    table.add_row("Avg Trades per Wallet", f"{wallets['total_trades'].mean():.1f}")
    table.add_row("Avg Markets per Wallet", f"{wallets['unique_markets'].mean():.1f}")

    console.print(table)


@app.command()
def validate(
    data_dir: Path = typer.Option(
        Path("data/processed"),
        "--dir", "-d",
        help="Data directory to validate"
    ),
):
    """Validate stored data files."""
    console.print("[bold blue]Validating data files...[/bold blue]")

    storage = DataStorage(data_dir)
    validator = DataValidator()

    all_valid = True
    all_messages = []

    # Validate markets
    if storage.file_exists("selected_markets.parquet"):
        markets = storage.load_markets("selected_markets.parquet")
        valid, messages = validator.validate_markets(markets)
        all_valid = all_valid and valid
        all_messages.extend(messages)
    elif storage.file_exists("all_markets.parquet"):
        markets = storage.load_markets("all_markets.parquet")
        valid, messages = validator.validate_markets(markets)
        all_valid = all_valid and valid
        all_messages.extend(messages)
    else:
        console.print("[yellow]No markets file found[/yellow]")

    # Validate trades
    if storage.file_exists("trades.parquet"):
        trades = storage.load_trades()
        valid, messages = validator.validate_trades(trades)
        all_valid = all_valid and valid
        all_messages.extend(messages)
    else:
        console.print("[yellow]No trades file found[/yellow]")

    # Validate wallets
    if storage.file_exists("wallets.parquet"):
        wallets = storage.load_wallets()
        valid, messages = validator.validate_wallets(wallets)
        all_valid = all_valid and valid
        all_messages.extend(messages)
    else:
        console.print("[yellow]No wallets file found[/yellow]")

    # Print results
    console.print()
    for msg in all_messages:
        if msg.startswith("Error"):
            console.print(f"[red]{msg}[/red]")
        elif msg.startswith("Warning"):
            console.print(f"[yellow]{msg}[/yellow]")
        elif msg.startswith("✓"):
            console.print(f"[green]{msg}[/green]")
        else:
            console.print(msg)

    if all_valid:
        console.print("\n[bold green]✓ All validations passed![/bold green]")
    else:
        console.print("\n[bold red]✗ Validation failed![/bold red]")
        raise typer.Exit(1)


@app.command()
def run_pipeline(
    n_markets: int = typer.Option(
        100,
        "--count", "-n",
        help="Number of markets to select"
    ),
    lookback_days: int = typer.Option(
        7,
        "--days", "-d",
        help="Number of days to look back"
    ),
    output_dir: Path = typer.Option(
        Path("data/processed"),
        "--output", "-o",
        help="Output directory"
    ),
    skip_validation: bool = typer.Option(
        False,
        "--skip-validation",
        help="Skip validation step"
    ),
    refresh_markets: bool = typer.Option(
        False,
        "--refresh-markets",
        help="Force refresh market data from Polymarket API"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Show detailed logging"
    ),
):
    """Run complete data ingestion pipeline using Goldsky for trade data."""
    # Suppress logging unless verbose
    if not verbose:
        logging.getLogger("src.data.goldsky").setLevel(logging.WARNING)
        logging.getLogger("src.data.fetcher").setLevel(logging.WARNING)
        logging.getLogger("src.data.client").setLevel(logging.WARNING)

    console.print("[bold blue]═══ TRADY DATA INGESTION PIPELINE ═══[/bold blue]")
    console.print("[dim]Using Goldsky for full historical trade data[/dim]")
    console.print()

    storage = DataStorage(output_dir)

    # Create main progress display
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:

        # Overall pipeline progress (4 steps)
        pipeline_task = progress.add_task("[bold cyan]Pipeline Progress", total=4)

        # Step 1: Load or fetch markets
        progress.update(pipeline_task, description="[bold cyan]Step 1/4: Loading markets...")
        markets_task = progress.add_task("[green]Loading markets...", total=None)

        # Try to load existing markets first (avoid Polymarket API timeout)
        markets = pd.DataFrame() if refresh_markets else storage.load_markets("all_markets.parquet")
        if markets.empty:
            if refresh_markets:
                console.print("  [yellow]Refreshing markets from API...[/yellow]")
            else:
                console.print("  [yellow]No cached markets, fetching from API...[/yellow]")
            async def _fetch_markets():
                async with PolymarketClient() as client:
                    selector = MarketSelector(client)
                    return await selector.fetch_all_markets()
            markets = run_async(_fetch_markets())
            storage.save_markets(markets, "all_markets.parquet")
        else:
            console.print("  [dim]Using cached market data[/dim]")

        progress.update(markets_task, completed=1, total=1)
        progress.update(pipeline_task, advance=1)
        console.print(f"  [green]✓ Loaded {len(markets)} markets[/green]")

        # Step 2: Load or select markets
        progress.update(pipeline_task, description="[bold cyan]Step 2/4: Selecting markets...")
        select_task = progress.add_task("[green]Selecting markets...", total=None)

        # Try to load existing selected markets
        selected = storage.load_markets("selected_markets.parquet")
        if selected.empty or len(selected) != n_markets:
            # Need to select markets (no API call needed - just filtering)
            async def _select():
                async with PolymarketClient() as client:
                    selector = MarketSelector(client)
                    return selector.select_markets(markets, n_total=n_markets)
            selected = run_async(_select())
            storage.save_markets(selected, "selected_markets.parquet")
        else:
            console.print("  [dim]Using cached market selection[/dim]")

        progress.update(select_task, completed=1, total=1)
        progress.update(pipeline_task, advance=1)
        console.print(f"  [green]✓ Selected {len(selected)} markets[/green]")

    # Step 3: Fetch trades from Goldsky (outside Progress context for better display)
    console.print()
    console.print("[bold cyan]Step 3/4: Fetching trades from Goldsky...[/bold cyan]")

    from rich.live import Live
    from rich.text import Text

    start_time = datetime.now()
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    state = {"count": 0}

    def create_goldsky_display():
        elapsed = datetime.now() - start_time
        elapsed_str = str(elapsed).split('.')[0]
        rate = state["count"] / max(elapsed.total_seconds(), 1)

        text = Text()
        text.append(f"  Trades: ", style="dim")
        text.append(f"{state['count']:,}", style="bold green")
        text.append(f"  |  Rate: ", style="dim")
        text.append(f"{rate:,.0f}/sec", style="yellow")
        text.append(f"  |  Elapsed: ", style="dim")
        text.append(f"{elapsed_str}", style="white")
        text.append(f"  |  Since: ", style="dim")
        text.append(f"{cutoff.strftime('%Y-%m-%d')}", style="cyan")

        return text

    def goldsky_progress_callback(count: int):
        state["count"] = count

    async def _fetch_trades_goldsky():
        fetcher = GoldskyTradeFetcher(lookback_days=lookback_days)
        return await fetcher.fetch_all_trades(progress_callback=goldsky_progress_callback)

    with Live(create_goldsky_display(), refresh_per_second=4, console=console, transient=True) as live:
        async def fetch_with_updates():
            task = asyncio.create_task(_fetch_trades_goldsky())
            while not task.done():
                live.update(create_goldsky_display())
                await asyncio.sleep(0.25)
            return await task

        trades = asyncio.run(fetch_with_updates())

    console.print(f"  [green]✓ Fetched {len(trades):,} trades[/green]")
    storage.save_trades(trades)

    # Continue with remaining steps using Progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=False,
    ) as progress:

        # Step 4: Build wallet index
        console.print()
        console.print("[bold cyan]Step 4/4: Building wallet index...[/bold cyan]")
        wallet_task = progress.add_task("[green]Building wallet index...", total=None)

        indexer = WalletIndexer()
        wallets = indexer.build_index(trades)
        storage.save_wallets(wallets)
        progress.update(wallet_task, completed=1, total=1)

    console.print(f"  [green]✓ Indexed {len(wallets):,} wallets[/green]")

    # Validation (outside progress context)
    if not skip_validation:
        console.print()
        console.print("[bold]Validating data...[/bold]")
        validator = DataValidator()
        valid, _ = validator.validate_all(selected, trades, wallets)
        if valid:
            console.print("  [green]✓ All validations passed[/green]")
        else:
            console.print("  [yellow]⚠ Validation warnings (see logs)[/yellow]")

    # Summary
    console.print()
    console.print("[bold blue]═══ PIPELINE COMPLETE ═══[/bold blue]")

    table = Table(title="Data Summary")
    table.add_column("Data", style="cyan")
    table.add_column("Records", style="magenta")
    table.add_column("File", style="green")

    table.add_row("Markets", str(len(selected)), "selected_markets.parquet")
    table.add_row("Trades", f"{len(trades):,}", "trades.parquet")
    table.add_row("Wallets", f"{len(wallets):,}", "wallets.parquet")

    console.print(table)
    console.print(f"\nOutput directory: {output_dir.absolute()}")


@app.command()
def info(
    data_dir: Path = typer.Option(
        Path("data/processed"),
        "--dir", "-d",
        help="Data directory"
    ),
):
    """Show information about stored data files."""
    storage = DataStorage(data_dir)

    files = storage.list_files("*.parquet")

    if not files:
        console.print(f"[yellow]No Parquet files found in {data_dir}[/yellow]")
        return

    table = Table(title=f"Data Files in {data_dir}")
    table.add_column("File", style="cyan")
    table.add_column("Size", style="magenta")
    table.add_column("Records", style="green")

    for path in sorted(files):
        info = storage.get_file_info(path.name)
        try:
            import pandas as pd
            df = pd.read_parquet(path)
            records = len(df)
        except Exception:
            records = "?"

        table.add_row(
            path.name,
            f"{info['size_mb']:.2f} MB",
            str(records)
        )

    console.print(table)


def main():
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
