"""
CLI interface for signal detection pipeline.

Commands:
- detect-signals: Run signal detection on all trades
- analyze-signal: Analyze signals for a specific trade
- show-suspicious: Show most suspicious wallets/trades
- run-pipeline: Run complete signal detection pipeline
"""

import json
import logging
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from src.analysis import WalletCluster, WalletProfile

from .detector import InsiderSignalDetector
from .types import Trade

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# CLI app
app = typer.Typer(
    name="trady-signals",
    help="Trady signal detection pipeline for Polymarket insider detection",
    add_completion=False,
)

console = Console()


def load_data(data_dir: Path) -> tuple:
    """Load trades, markets, and wallet profiles."""
    trades_path = data_dir / "trades.parquet"
    markets_path = data_dir / "selected_markets.parquet"
    profiles_path = data_dir / "wallet_profiles.parquet"
    clusters_path = data_dir / "wallet_clusters.parquet"

    trades = pd.read_parquet(trades_path) if trades_path.exists() else pd.DataFrame()
    markets = pd.read_parquet(markets_path) if markets_path.exists() else None
    profiles_df = pd.read_parquet(profiles_path) if profiles_path.exists() else None
    clusters_df = pd.read_parquet(clusters_path) if clusters_path.exists() else None

    return trades, markets, profiles_df, clusters_df


def df_to_wallet_profiles(df: pd.DataFrame) -> dict:
    """Convert DataFrame to dict of WalletProfile objects."""
    profiles = {}
    if df is None or df.empty:
        return profiles

    for _, row in df.iterrows():
        profile = WalletProfile(
            address=row.get("address", row.get("wallet", "")),
            cluster_id=row.get("cluster_id"),
            first_seen=row.get("first_seen"),
            last_seen=row.get("last_seen"),
            total_trades=int(row.get("total_trades", 0)),
            total_volume=float(row.get("total_volume", 0)),
            unique_markets=int(row.get("unique_markets", 0)),
            is_whale=bool(row.get("is_whale", False)),
            freshness_score=float(row.get("freshness_score", 0)),
            is_zero_history=bool(row.get("is_zero_history", False)),
            is_new_to_polymarket=bool(row.get("is_new_to_polymarket", False)),
            is_recently_funded=bool(row.get("is_recently_funded", False)),
            days_on_polymarket=int(row.get("days_on_polymarket", 0)),
            primary_funding_source=str(row.get("primary_funding_source", "unknown")),
            funding_risk_score=float(row.get("funding_risk_score", 0)),
            has_privacy_funding=bool(row.get("has_privacy_funding", False)),
            retail_likelihood=float(row.get("retail_likelihood", 0.5)),
            sophistication_score=float(row.get("sophistication_score", 0.5)),
            avg_trade_size=float(row.get("avg_trade_size", 0)),
            max_trade_size=float(row.get("max_trade_size", 0)),
            preliminary_insider_score=float(row.get("preliminary_insider_score", 0)),
            negative_signal_score=float(row.get("negative_signal_score", 0)),
        )
        profiles[profile.address] = profile

    return profiles


def df_to_clusters(df: pd.DataFrame) -> list:
    """Convert DataFrame to list of WalletCluster objects."""
    clusters = []
    if df is None or df.empty:
        return clusters

    for _, row in df.iterrows():
        wallets_str = row.get("wallets", "")
        wallets = wallets_str.split(",") if wallets_str else []

        cluster = WalletCluster(
            cluster_id=str(row.get("cluster_id", "")),
            wallets=wallets,
            primary_wallet=str(row.get("primary_wallet", wallets[0] if wallets else "")),
            clustering_method=str(row.get("clustering_method", row.get("method", "unknown"))),
            confidence=float(row.get("confidence", 0)),
            total_volume=float(row.get("total_volume", 0)),
            combined_trades=int(row.get("combined_trades", row.get("wallet_count", 0))),
        )
        clusters.append(cluster)

    return clusters


@app.command()
def detect_signals(
    data_dir: Path = typer.Option(
        Path("data/processed"),
        "--data", "-d",
        help="Data directory with trades and profiles"
    ),
    output_file: Path = typer.Option(
        Path("data/processed/signals.parquet"),
        "--output", "-o",
        help="Output file for detected signals"
    ),
    min_score: float = typer.Option(
        0.3,
        "--min-score", "-s",
        help="Minimum score threshold for signals"
    ),
    top_n: int = typer.Option(
        50,
        "--top", "-n",
        help="Show top N signals in summary"
    ),
):
    """Run signal detection on all trades."""
    console.print("[bold blue]Running signal detection...[/bold blue]")

    trades, markets, profiles_df, clusters_df = load_data(data_dir)

    if trades.empty:
        console.print("[red]No trades data found![/red]")
        console.print("Run 'trady-data run-pipeline' and 'trady-analysis run-analysis' first.")
        raise typer.Exit(1)

    if profiles_df is None or profiles_df.empty:
        console.print("[red]No wallet profiles found![/red]")
        console.print("Run 'trady-analysis run-analysis' first.")
        raise typer.Exit(1)

    console.print(f"Loaded {len(trades)} trades")
    console.print(f"Loaded {len(profiles_df)} wallet profiles")

    # Convert to objects
    wallet_profiles = df_to_wallet_profiles(profiles_df)
    clusters = df_to_clusters(clusters_df)
    console.print(f"Loaded {len(clusters)} clusters")

    from rich.progress import Progress, BarColumn, TaskProgressColumn, TimeElapsedColumn

    with Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Analyzing wallets and trades...", total=100)

        detector = InsiderSignalDetector(
            trades_df=trades,
            markets_df=markets,
            wallet_profiles=wallet_profiles,
            clusters=clusters,
        )

        def progress_callback(current, total):
            if total > 0:
                pct = min(int((current / total) * 100), 100)
                progress.update(task, completed=pct)

        signals = detector.detect_all_signals(
            min_score=min_score,
            progress_callback=progress_callback,
        )

    # Save to parquet
    if signals:
        df = detector.to_dataframe(signals)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_file, index=False)
        console.print(f"[green]✓ Saved {len(df)} signals to {output_file}[/green]")
    else:
        console.print("[yellow]No signals detected above threshold[/yellow]")

    # Show summary
    console.print()
    table = Table(title="Signal Detection Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Total Trades Analyzed", str(len(trades) * 2))  # Both sides
    table.add_row("Signals Detected", str(len(signals)))
    table.add_row("Min Score Threshold", str(min_score))

    if signals:
        avg_score = sum(s.final_score for s in signals) / len(signals)
        table.add_row("Average Score", f"{avg_score:.3f}")
        table.add_row("Max Score", f"{signals[0].final_score:.3f}")

        # Count by category
        category_counts = {}
        for s in signals:
            for sig in s.signals:
                cat = sig.category.value
                category_counts[cat] = category_counts.get(cat, 0) + 1

        for cat, count in sorted(category_counts.items()):
            table.add_row(f"  {cat.title()} Signals", str(count))

    console.print(table)

    # Show top signals
    if signals:
        console.print()
        signal_table = Table(title=f"Top {min(top_n, len(signals))} Suspicious Trades")
        signal_table.add_column("Wallet", style="cyan")
        signal_table.add_column("Score", style="red")
        signal_table.add_column("Fresh", style="yellow")
        signal_table.add_column("Timing", style="green")
        signal_table.add_column("Sizing", style="blue")
        signal_table.add_column("Funding", style="magenta")
        signal_table.add_column("Size", style="white")

        for s in signals[:top_n]:
            signal_table.add_row(
                s.trade.wallet[:16] + "...",
                f"{s.final_score:.3f}",
                f"{s.aggregated.category_scores.get('freshness', 0):.2f}",
                f"{s.aggregated.category_scores.get('timing', 0):.2f}",
                f"{s.aggregated.category_scores.get('sizing', 0):.2f}",
                f"{s.aggregated.category_scores.get('funding', 0):.2f}",
                f"${s.trade.notional:,.0f}",
            )

        console.print(signal_table)


@app.command()
def analyze_signal(
    trade_id: str = typer.Argument(..., help="Trade ID to analyze"),
    data_dir: Path = typer.Option(
        Path("data/processed"),
        "--data", "-d",
        help="Data directory"
    ),
):
    """Analyze signals for a specific trade (debugging)."""
    console.print(f"[bold blue]Analyzing trade: {trade_id}[/bold blue]")

    trades, markets, profiles_df, clusters_df = load_data(data_dir)

    if trades.empty:
        console.print("[red]No trades data found![/red]")
        raise typer.Exit(1)

    wallet_profiles = df_to_wallet_profiles(profiles_df)
    clusters = df_to_clusters(clusters_df)

    detector = InsiderSignalDetector(
        trades_df=trades,
        markets_df=markets,
        wallet_profiles=wallet_profiles,
        clusters=clusters,
    )

    result = detector.analyze_trade(trade_id)

    if not result:
        console.print(f"[red]Trade not found: {trade_id}[/red]")
        raise typer.Exit(1)

    # Print trade info
    console.print()
    console.print("[bold]Trade Details:[/bold]")
    for key, value in result["trade"].items():
        console.print(f"  {key}: {value}")

    # Print wallet profile
    console.print()
    console.print("[bold]Wallet Profile:[/bold]")
    for key, value in result["wallet_profile"].items():
        console.print(f"  {key}: {value}")

    # Print score breakdown
    console.print()
    console.print("[bold]Score Breakdown:[/bold]")
    breakdown = result["score_breakdown"]
    console.print(f"  Final Score: [red]{breakdown['final_score']}[/red]")
    console.print(f"  Raw Score: {breakdown['raw_score']}")
    console.print(f"  Negative Discount: {breakdown['modifiers']['negative_discount']}")
    console.print(f"  Market Boost: {breakdown['modifiers']['market_boost']}")

    # Category breakdown
    console.print()
    console.print("[bold]Category Breakdown:[/bold]")
    for cat, info in breakdown.get("category_breakdown", {}).items():
        console.print(f"  {cat}: score={info['score']:.3f}, weight={info['weight']}, contribution={info['contribution']:.3f}")

    # Signals
    console.print()
    console.print("[bold]Detected Signals:[/bold]")
    for sig in result["signals"]:
        console.print(f"  [{sig['category']}] {sig['type']}: confidence={sig['confidence']:.3f}")

    # Filter reasons
    if result["filter_reasons"]:
        console.print()
        console.print("[bold]Filter Reasons (reduces score):[/bold]")
        for reason in result["filter_reasons"]:
            console.print(f"  - {reason}")


@app.command()
def show_suspicious(
    data_dir: Path = typer.Option(
        Path("data/processed"),
        "--data", "-d",
        help="Data directory"
    ),
    min_score: float = typer.Option(
        0.5,
        "--min-score", "-s",
        help="Minimum score threshold"
    ),
    top_n: int = typer.Option(
        20,
        "--top", "-n",
        help="Number of wallets to show"
    ),
):
    """Show most suspicious wallets."""
    console.print("[bold blue]Finding suspicious wallets...[/bold blue]")

    trades, markets, profiles_df, clusters_df = load_data(data_dir)

    if trades.empty:
        console.print("[red]No trades data found![/red]")
        raise typer.Exit(1)

    wallet_profiles = df_to_wallet_profiles(profiles_df)
    clusters = df_to_clusters(clusters_df)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Analyzing trades...", total=None)

        detector = InsiderSignalDetector(
            trades_df=trades,
            markets_df=markets,
            wallet_profiles=wallet_profiles,
            clusters=clusters,
        )

        suspicious = detector.get_suspicious_wallets(min_score=min_score)

    if not suspicious:
        console.print("[yellow]No suspicious wallets found above threshold[/yellow]")
        return

    # Show table
    table = Table(title=f"Top {min(top_n, len(suspicious))} Suspicious Wallets")
    table.add_column("Wallet", style="cyan")
    table.add_column("Avg Score", style="red")
    table.add_column("Max Score", style="magenta")
    table.add_column("Trades", style="green")
    table.add_column("Volume", style="yellow")
    table.add_column("Markets", style="blue")

    for wallet, stats in list(suspicious.items())[:top_n]:
        table.add_row(
            wallet[:16] + "...",
            f"{stats['avg_score']:.3f}",
            f"{stats['max_score']:.3f}",
            str(stats["trade_count"]),
            f"${stats['total_volume']:,.0f}",
            str(len(stats["markets"])),
        )

    console.print(table)


@app.command()
def run_pipeline(
    data_dir: Path = typer.Option(
        Path("data/processed"),
        "--data", "-d",
        help="Data directory"
    ),
    output_dir: Path = typer.Option(
        Path("data/processed"),
        "--output", "-o",
        help="Output directory"
    ),
    min_score: float = typer.Option(
        0.3,
        "--min-score", "-s",
        help="Minimum score threshold"
    ),
):
    """Run complete signal detection pipeline."""
    console.print("[bold blue]═══ SIGNAL DETECTION PIPELINE ═══[/bold blue]")
    console.print()

    trades, markets, profiles_df, clusters_df = load_data(data_dir)

    if trades.empty:
        console.print("[red]No trades data found![/red]")
        console.print("Run 'trady-data run-pipeline' first.")
        raise typer.Exit(1)

    if profiles_df is None or profiles_df.empty:
        console.print("[red]No wallet profiles found![/red]")
        console.print("Run 'trady-analysis run-analysis' first.")
        raise typer.Exit(1)

    console.print(f"Loaded {len(trades)} trades")
    console.print(f"Loaded {len(profiles_df)} wallet profiles")

    # Convert to objects
    wallet_profiles = df_to_wallet_profiles(profiles_df)
    clusters = df_to_clusters(clusters_df)

    console.print()
    console.print("[bold]Step 1/2: Detecting signals...[/bold]")

    detector = InsiderSignalDetector(
        trades_df=trades,
        markets_df=markets,
        wallet_profiles=wallet_profiles,
        clusters=clusters,
    )

    # Use progress bar for detection
    from rich.progress import Progress, BarColumn, TaskProgressColumn, TimeElapsedColumn

    with Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Analyzing wallets and trades...", total=100)

        def progress_callback(current, total):
            if total > 0:
                pct = min(int((current / total) * 100), 100)
                progress.update(task, completed=pct)

        signals = detector.detect_all_signals(
            min_score=min_score,
            progress_callback=progress_callback,
        )

    console.print(f"  [green]✓ Detected {len(signals)} signals[/green]")

    console.print()
    console.print("[bold]Step 2/2: Saving results...[/bold]")

    if signals:
        df = detector.to_dataframe(signals)
        output_path = output_dir / "signals.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        console.print(f"  [green]✓ Saved to {output_path}[/green]")

        # Compute suspicious wallets from already-detected signals (no re-running)
        wallet_stats: dict = {}
        for s in signals:
            wallet = s.trade.wallet
            if wallet not in wallet_stats:
                wallet_stats[wallet] = {
                    "trade_count": 0,
                    "total_score": 0.0,
                    "max_score": 0.0,
                    "total_volume": 0.0,
                    "markets": set(),
                }
            wallet_stats[wallet]["trade_count"] += 1
            wallet_stats[wallet]["total_score"] += s.final_score
            wallet_stats[wallet]["max_score"] = max(wallet_stats[wallet]["max_score"], s.final_score)
            wallet_stats[wallet]["total_volume"] += s.trade.notional
            wallet_stats[wallet]["markets"].add(s.trade.market_id)

        # Convert to final format
        suspicious = {}
        for wallet, stats in wallet_stats.items():
            avg_score = stats["total_score"] / stats["trade_count"]
            if avg_score >= min_score:
                suspicious[wallet] = {
                    "trade_count": stats["trade_count"],
                    "avg_score": round(avg_score, 3),
                    "max_score": round(stats["max_score"], 3),
                    "total_volume": round(stats["total_volume"], 2),
                    "markets": list(stats["markets"]),
                }

        if suspicious:
            # Sort by avg_score
            suspicious = dict(sorted(suspicious.items(), key=lambda x: x[1]["avg_score"], reverse=True))
            suspicious_df = pd.DataFrame([
                {"wallet": w, **{k: v for k, v in s.items() if k != "markets"}, "market_count": len(s["markets"])}
                for w, s in suspicious.items()
            ])
            suspicious_path = output_dir / "suspicious_wallets.parquet"
            suspicious_df.to_parquet(suspicious_path, index=False)
            console.print(f"  [green]✓ Saved {len(suspicious)} suspicious wallets to {suspicious_path}[/green]")
    else:
        suspicious = {}

    console.print()
    console.print("[bold blue]═══ PIPELINE COMPLETE ═══[/bold blue]")

    # Summary
    table = Table(title="Signal Detection Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Trades Analyzed", f"{len(trades):,}")
    table.add_row("Signals Detected", f"{len(signals):,}")
    table.add_row("Suspicious Wallets", f"{len(suspicious):,}")
    table.add_row("Min Score Threshold", str(min_score))

    if signals:
        avg_score = sum(s.final_score for s in signals) / len(signals)
        table.add_row("Average Score", f"{avg_score:.3f}")
        table.add_row("Max Score", f"{signals[0].final_score:.3f}")

    console.print(table)


def main():
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
