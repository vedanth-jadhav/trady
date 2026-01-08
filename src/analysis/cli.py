"""
CLI interface for wallet analysis pipeline.

Commands:
- analyze-freshness: Analyze wallet freshness
- analyze-behavior: Build behavior profiles
- cluster-wallets: Run wallet clustering
- build-profiles: Build complete wallet profiles
- run-analysis: Run complete wallet analysis pipeline
"""

import asyncio
from pathlib import Path
from typing import Optional
import logging

import typer
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from .freshness import FreshnessAnalyzer
from .behavior import BehaviorProfiler
from .clusterer import WalletClusterer
from .funding import FundingTracker
from .profiler import WalletProfileBuilder, NegativeSignalDetector

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# CLI app
app = typer.Typer(
    name="trady-analysis",
    help="Trady wallet analysis pipeline for Polymarket",
    add_completion=False,
)

console = Console()


def load_data(data_dir: Path) -> tuple:
    """Load trades, markets, and wallets data."""
    trades_path = data_dir / "trades.parquet"
    markets_path = data_dir / "selected_markets.parquet"
    wallets_path = data_dir / "wallets.parquet"

    trades = pd.read_parquet(trades_path) if trades_path.exists() else pd.DataFrame()
    markets = pd.read_parquet(markets_path) if markets_path.exists() else None
    wallets = pd.read_parquet(wallets_path) if wallets_path.exists() else None

    return trades, markets, wallets


@app.command()
def analyze_freshness(
    data_dir: Path = typer.Option(
        Path("data/processed"),
        "--data", "-d",
        help="Data directory with trades.parquet"
    ),
    output_file: Path = typer.Option(
        Path("data/processed/freshness_profiles.parquet"),
        "--output", "-o",
        help="Output file for freshness profiles"
    ),
    top_n: int = typer.Option(
        50,
        "--top", "-n",
        help="Show top N freshest wallets"
    ),
    min_volume: float = typer.Option(
        100,
        "--min-volume",
        help="Minimum volume filter"
    ),
):
    """Analyze wallet freshness to identify new/suspicious wallets."""
    console.print("[bold blue]Analyzing wallet freshness...[/bold blue]")

    trades, _, _ = load_data(data_dir)

    if trades.empty:
        console.print("[red]No trades data found![/red]")
        raise typer.Exit(1)

    console.print(f"Loaded {len(trades)} trades")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Computing freshness profiles...", total=None)

        analyzer = FreshnessAnalyzer(trades)
        freshest = analyzer.get_freshest_wallets(n=top_n, min_volume=min_volume)

    # Save all profiles
    all_profiles = analyzer.compute_all_freshness_profiles()
    df = pd.DataFrame([{
        "wallet": p.wallet,
        "freshness_score": p.freshness_score,
        "is_zero_history": p.is_zero_history,
        "is_new_to_polymarket": p.is_new_to_polymarket,
        "total_trades": p.total_polymarket_trades,
        "days_active": p.days_active,
    } for p in all_profiles.values()])

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file, index=False)
    console.print(f"[green]✓ Saved {len(df)} freshness profiles to {output_file}[/green]")

    # Show top freshest
    table = Table(title=f"Top {len(freshest)} Freshest Wallets")
    table.add_column("Wallet", style="cyan")
    table.add_column("Freshness", style="magenta")
    table.add_column("Trades", style="green")
    table.add_column("Days Active", style="yellow")
    table.add_column("New?", style="red")

    for profile in freshest[:20]:
        table.add_row(
            profile.wallet[:16] + "...",
            f"{profile.freshness_score:.3f}",
            str(profile.total_polymarket_trades),
            str(profile.days_active),
            "✓" if profile.is_new_to_polymarket else ""
        )

    console.print(table)


@app.command()
def analyze_behavior(
    data_dir: Path = typer.Option(
        Path("data/processed"),
        "--data", "-d",
        help="Data directory"
    ),
    output_file: Path = typer.Option(
        Path("data/processed/behavior_profiles.parquet"),
        "--output", "-o",
        help="Output file for behavior profiles"
    ),
    min_trades: int = typer.Option(
        3,
        "--min-trades",
        help="Minimum trades for analysis"
    ),
):
    """Build behavior profiles for all wallets."""
    console.print("[bold blue]Analyzing wallet behavior...[/bold blue]")

    trades, markets, _ = load_data(data_dir)

    if trades.empty:
        console.print("[red]No trades data found![/red]")
        raise typer.Exit(1)

    console.print(f"Loaded {len(trades)} trades")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Building behavior profiles...", total=None)

        profiler = BehaviorProfiler(trades, markets)

        # Get wallets with minimum trades
        trade_counts = pd.concat([
            trades['maker_address'],
            trades['taker_address']
        ]).value_counts()
        valid_wallets = trade_counts[trade_counts >= min_trades].index.tolist()

        profiles = profiler.build_all_profiles(valid_wallets)

    # Convert to DataFrame
    records = []
    for wallet, p in profiles.items():
        records.append({
            "wallet": wallet,
            "avg_trades_per_day": p.avg_trades_per_day,
            "burst_episodes": p.burst_episodes,
            "off_hours_ratio": p.off_hours_ratio,
            "avg_trade_size": p.avg_trade_size,
            "max_trade_size": p.max_trade_size,
            "size_variance": p.size_variance,
            "unique_markets": p.unique_markets,
            "market_concentration": p.market_concentration,
            "retail_likelihood": p.retail_likelihood,
            "sophistication_score": p.sophistication_score,
        })

    df = pd.DataFrame(records)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file, index=False)
    console.print(f"[green]✓ Saved {len(df)} behavior profiles to {output_file}[/green]")

    # Show summary
    table = Table(title="Behavior Analysis Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Total Wallets Analyzed", str(len(profiles)))
    table.add_row("Avg Sophistication Score", f"{df['sophistication_score'].mean():.3f}")
    table.add_row("High Sophistication (>0.7)", str((df['sophistication_score'] > 0.7).sum()))
    table.add_row("Burst Traders (>0 episodes)", str((df['burst_episodes'] > 0).sum()))
    table.add_row("Off-Hours Traders (>50%)", str((df['off_hours_ratio'] > 0.5).sum()))

    console.print(table)


@app.command()
def cluster_wallets(
    data_dir: Path = typer.Option(
        Path("data/processed"),
        "--data", "-d",
        help="Data directory"
    ),
    output_file: Path = typer.Option(
        Path("data/processed/wallet_clusters.parquet"),
        "--output", "-o",
        help="Output file for clusters"
    ),
    methods: str = typer.Option(
        "temporal,behavior",
        "--methods", "-m",
        help="Clustering methods (comma-separated: temporal,behavior,correlation)"
    ),
):
    """Cluster related wallets to detect coordinated activity."""
    console.print("[bold blue]Clustering wallets...[/bold blue]")

    trades, markets, wallets = load_data(data_dir)

    if trades.empty:
        console.print("[red]No trades data found![/red]")
        raise typer.Exit(1)

    console.print(f"Loaded {len(trades)} trades")

    # Prepare trades with wallet column
    trades_with_wallet = trades.copy()
    trades_with_wallet['wallet'] = trades_with_wallet['maker_address']

    # Load behavior profiles if available
    behavior_path = data_dir / "behavior_profiles.parquet"
    behavior_profiles = {}
    if behavior_path.exists():
        behavior_df = pd.read_parquet(behavior_path)
        from .types import BehaviorProfile
        for _, row in behavior_df.iterrows():
            behavior_profiles[row['wallet']] = BehaviorProfile(
                wallet=row['wallet'],
                avg_trades_per_day=row.get('avg_trades_per_day', 0),
                burst_episodes=row.get('burst_episodes', 0),
                off_hours_ratio=row.get('off_hours_ratio', 0),
                avg_trade_size=row.get('avg_trade_size', 0),
                max_trade_size=row.get('max_trade_size', 0),
                size_variance=row.get('size_variance', 0),
                unique_markets=row.get('unique_markets', 0),
                market_concentration=row.get('market_concentration', 0),
                niche_market_ratio=row.get('niche_market_ratio', 0),
            )
        console.print(f"Loaded {len(behavior_profiles)} behavior profiles")

    # Get wallet volumes
    wallet_volumes = {}
    if wallets is not None and 'address' in wallets.columns:
        wallet_volumes = dict(zip(wallets['address'], wallets.get('total_volume', [0] * len(wallets))))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Running clustering algorithms...", total=None)

        clusterer = WalletClusterer(
            trades_df=trades_with_wallet,
            behavior_profiles=behavior_profiles,
            wallet_volumes=wallet_volumes
        )

        method_list = [m.strip() for m in methods.split(",")]
        result = clusterer.run_full_clustering(methods=method_list)

    # Save clusters
    records = []
    for cluster in result.clusters:
        records.append({
            "cluster_id": cluster.cluster_id,
            "wallets": ",".join(cluster.wallets),
            "primary_wallet": cluster.primary_wallet,
            "wallet_count": len(cluster.wallets),
            "clustering_method": cluster.clustering_method,
            "confidence": cluster.confidence,
            "total_volume": cluster.total_volume,
            "combined_trades": cluster.combined_trades,
        })

    df = pd.DataFrame(records)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file, index=False)
    console.print(f"[green]✓ Saved {len(df)} clusters to {output_file}[/green]")

    # Show summary
    table = Table(title="Clustering Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Total Clusters", str(len(result.clusters)))
    table.add_row("Clustered Wallets", str(len(result.wallet_to_cluster)))
    table.add_row("Unclustered Wallets", str(len(result.unclustered_wallets)))

    if result.clusters:
        table.add_row("Avg Cluster Size", f"{sum(len(c.wallets) for c in result.clusters) / len(result.clusters):.1f}")
        table.add_row("Max Cluster Size", str(max(len(c.wallets) for c in result.clusters)))

    console.print(table)

    # Show top clusters
    if result.clusters:
        console.print()
        suspicious = clusterer.get_suspicious_clusters(result, min_wallets=2)[:10]

        if suspicious:
            cluster_table = Table(title="Top Suspicious Clusters")
            cluster_table.add_column("Cluster ID", style="cyan")
            cluster_table.add_column("Wallets", style="green")
            cluster_table.add_column("Method", style="yellow")
            cluster_table.add_column("Confidence", style="magenta")
            cluster_table.add_column("Volume", style="blue")

            for c in suspicious:
                cluster_table.add_row(
                    c.cluster_id[:8] + "...",
                    str(len(c.wallets)),
                    c.clustering_method,
                    f"{c.confidence:.2f}",
                    f"${c.total_volume:,.0f}"
                )

            console.print(cluster_table)


@app.command()
def build_profiles(
    data_dir: Path = typer.Option(
        Path("data/processed"),
        "--data", "-d",
        help="Data directory"
    ),
    output_file: Path = typer.Option(
        Path("data/processed/wallet_profiles.parquet"),
        "--output", "-o",
        help="Output file for profiles"
    ),
    top_n: int = typer.Option(
        50,
        "--top", "-n",
        help="Show top N insider candidates"
    ),
    run_clustering: bool = typer.Option(
        True,
        "--cluster/--no-cluster",
        help="Run clustering analysis"
    ),
):
    """Build complete wallet profiles combining all analyses."""
    console.print("[bold blue]Building complete wallet profiles...[/bold blue]")

    trades, markets, wallets = load_data(data_dir)

    if trades.empty:
        console.print("[red]No trades data found![/red]")
        raise typer.Exit(1)

    console.print(f"Loaded {len(trades)} trades")
    if wallets is not None:
        console.print(f"Loaded {len(wallets)} wallet base stats")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Building wallet profiles...", total=None)

        builder = WalletProfileBuilder(trades, markets, wallets)
        builder.compute_all_analyses(run_clustering=run_clustering)
        profiles = builder.build_all_profiles()

    # Convert to DataFrame and save
    df = builder.to_dataframe(profiles)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file, index=False)
    console.print(f"[green]✓ Saved {len(df)} wallet profiles to {output_file}[/green]")

    # Show summary
    table = Table(title="Profile Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Total Profiles", str(len(profiles)))
    table.add_row("Avg Insider Score", f"{df['preliminary_insider_score'].mean():.3f}")
    table.add_row("High Insider Score (>0.5)", str((df['preliminary_insider_score'] > 0.5).sum()))
    table.add_row("New Wallets", str(df['is_new_to_polymarket'].sum()))
    table.add_row("Clustered Wallets", str((df['cluster_id'].notna()).sum()))

    console.print(table)

    # Show top candidates
    top_candidates = builder.get_top_insider_candidates(profiles, n=top_n)

    if top_candidates:
        console.print()
        candidate_table = Table(title=f"Top {len(top_candidates)} Insider Candidates")
        candidate_table.add_column("Wallet", style="cyan")
        candidate_table.add_column("Insider Score", style="red")
        candidate_table.add_column("Freshness", style="yellow")
        candidate_table.add_column("Sophistication", style="green")
        candidate_table.add_column("Negative", style="blue")
        candidate_table.add_column("Volume", style="magenta")

        for p in top_candidates[:20]:
            candidate_table.add_row(
                p.address[:16] + "...",
                f"{p.preliminary_insider_score:.3f}",
                f"{p.freshness_score:.3f}",
                f"{p.sophistication_score:.3f}",
                f"{p.negative_signal_score:.3f}",
                f"${p.total_volume:,.0f}"
            )

        console.print(candidate_table)


@app.command()
def run_analysis(
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
    top_n: int = typer.Option(
        50,
        "--top", "-n",
        help="Show top N insider candidates"
    ),
):
    """Run complete wallet analysis pipeline."""
    console.print("[bold blue]═══ WALLET ANALYSIS PIPELINE ═══[/bold blue]")
    console.print()

    trades, markets, wallets = load_data(data_dir)

    if trades.empty:
        console.print("[red]No trades data found![/red]")
        console.print("Run 'trady-data run-pipeline' first to fetch data.")
        raise typer.Exit(1)

    console.print(f"Loaded {len(trades)} trades")

    # Step 1: Freshness Analysis
    console.print("[bold]Step 1/4: Analyzing freshness...[/bold]")
    freshness_analyzer = FreshnessAnalyzer(trades)
    freshness_profiles = freshness_analyzer.compute_all_freshness_profiles()
    console.print(f"  [green]✓ Computed {len(freshness_profiles)} freshness profiles[/green]")

    # Step 2: Behavior Analysis
    console.print("[bold]Step 2/4: Analyzing behavior...[/bold]")
    behavior_profiler = BehaviorProfiler(trades, markets)
    behavior_profiles = behavior_profiler.build_all_profiles()
    console.print(f"  [green]✓ Computed {len(behavior_profiles)} behavior profiles[/green]")

    # Step 3: Clustering
    console.print("[bold]Step 3/4: Clustering wallets...[/bold]")
    trades_with_wallet = trades.copy()
    trades_with_wallet['wallet'] = trades_with_wallet['maker_address']

    wallet_volumes = {}
    if wallets is not None and 'address' in wallets.columns:
        wallet_volumes = dict(zip(wallets['address'], wallets.get('total_volume', [0] * len(wallets))))

    clusterer = WalletClusterer(
        trades_df=trades_with_wallet,
        behavior_profiles=behavior_profiles,
        wallet_volumes=wallet_volumes
    )
    clustering_result = clusterer.run_full_clustering(methods=["temporal", "behavior"])
    console.print(f"  [green]✓ Found {len(clustering_result.clusters)} clusters[/green]")

    # Step 4: Build Complete Profiles
    console.print("[bold]Step 4/4: Building profiles...[/bold]")
    builder = WalletProfileBuilder(trades, markets, wallets)
    builder._freshness_profiles = freshness_profiles
    builder._behavior_profiles = behavior_profiles
    builder._clustering_result = clustering_result

    profiles = builder.build_all_profiles()
    console.print(f"  [green]✓ Built {len(profiles)} complete profiles[/green]")

    # Save all outputs
    console.print()
    console.print("[bold]Saving results...[/bold]")

    # Freshness profiles
    freshness_df = pd.DataFrame([{
        "wallet": p.wallet,
        "freshness_score": p.freshness_score,
        "is_new_to_polymarket": p.is_new_to_polymarket,
        "total_trades": p.total_polymarket_trades,
        "days_active": p.days_active,
    } for p in freshness_profiles.values()])
    freshness_df.to_parquet(output_dir / "freshness_profiles.parquet", index=False)

    # Behavior profiles
    behavior_df = pd.DataFrame([{
        "wallet": p.wallet,
        "avg_trades_per_day": p.avg_trades_per_day,
        "off_hours_ratio": p.off_hours_ratio,
        "market_concentration": p.market_concentration,
        "retail_likelihood": p.retail_likelihood,
        "sophistication_score": p.sophistication_score,
    } for p in behavior_profiles.values()])
    behavior_df.to_parquet(output_dir / "behavior_profiles.parquet", index=False)

    # Clusters
    cluster_df = pd.DataFrame([{
        "cluster_id": c.cluster_id,
        "wallets": ",".join(c.wallets),
        "wallet_count": len(c.wallets),
        "method": c.clustering_method,
        "confidence": c.confidence,
    } for c in clustering_result.clusters])
    cluster_df.to_parquet(output_dir / "wallet_clusters.parquet", index=False)

    # Full profiles
    profiles_df = builder.to_dataframe(profiles)
    profiles_df.to_parquet(output_dir / "wallet_profiles.parquet", index=False)

    console.print(f"  [green]✓ Saved all results to {output_dir}[/green]")

    # Summary
    console.print()
    console.print("[bold blue]═══ ANALYSIS COMPLETE ═══[/bold blue]")

    table = Table(title="Analysis Summary")
    table.add_column("Data", style="cyan")
    table.add_column("Records", style="magenta")
    table.add_column("File", style="green")

    table.add_row("Freshness Profiles", str(len(freshness_profiles)), "freshness_profiles.parquet")
    table.add_row("Behavior Profiles", str(len(behavior_profiles)), "behavior_profiles.parquet")
    table.add_row("Clusters", str(len(clustering_result.clusters)), "wallet_clusters.parquet")
    table.add_row("Complete Profiles", str(len(profiles)), "wallet_profiles.parquet")

    console.print(table)

    # Show top candidates
    top_candidates = builder.get_top_insider_candidates(profiles, n=top_n)

    if top_candidates:
        console.print()
        candidate_table = Table(title=f"Top {min(20, len(top_candidates))} Insider Candidates")
        candidate_table.add_column("Wallet", style="cyan")
        candidate_table.add_column("Score", style="red")
        candidate_table.add_column("Fresh", style="yellow")
        candidate_table.add_column("Soph", style="green")
        candidate_table.add_column("Volume", style="magenta")

        for p in top_candidates[:20]:
            candidate_table.add_row(
                p.address[:16] + "...",
                f"{p.preliminary_insider_score:.3f}",
                f"{p.freshness_score:.3f}",
                f"{p.sophistication_score:.3f}",
                f"${p.total_volume:,.0f}"
            )

        console.print(candidate_table)


def main():
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
