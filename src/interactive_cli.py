"""
Interactive CLI for Trady - Polymarket Insider Detection Bot

Provides a user-friendly interface for data ingestion and analysis workflows.
"""

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.layout import Layout
from rich.text import Text

try:
    import questionary
    from questionary import Style
except ImportError:
    questionary = None

# Import existing CLI functions
from src.data.cli import (
    fetch_markets as data_fetch_markets,
    select_markets as data_select_markets,
    fetch_trades as data_fetch_trades,
    build_wallet_index as data_build_wallet_index,
    run_pipeline as data_run_pipeline,
    info as data_info,
    validate as data_validate,
)

from src.analysis.cli import (
    analyze_freshness,
    analyze_behavior,
    cluster_wallets,
    build_profiles,
    run_analysis,
)

app = typer.Typer(
    name="trady",
    help="Interactive CLI for Trady - Polymarket Insider Detection Bot",
    add_completion=False,
)

console = Console()

# Custom style for questionary prompts
custom_style = Style([
    ('qmark', 'fg:#673ab7 bold'),
    ('question', 'bold'),
    ('answer', 'fg:#f44336 bold'),
    ('pointer', 'fg:#673ab7 bold'),
    ('highlighted', 'fg:#673ab7 bold'),
    ('selected', 'fg:#cc5454'),
    ('separator', 'fg:#cc5454'),
    ('instruction', ''),
    ('text', ''),
])


def check_questionary():
    """Check if questionary is installed."""
    if questionary is None:
        console.print("[red]Error: questionary is not installed.[/red]")
        console.print("Install it with: [cyan]pip install questionary[/cyan]")
        raise typer.Exit(1)


def show_banner():
    """Display the Trady banner."""
    banner_text = """
╔╦╗╦═╗╔═╗╔╦╗╦ ╦
 ║ ╠╦╝╠═╣ ║║╚╦╝
 ╩ ╩╚═╩ ╩═╩╝ ╩
    """
    console.print(Panel(
        Text(banner_text, style="bold cyan") +
        Text("\nPolymarket Insider Detection Bot", style="bold white"),
        border_style="cyan",
        box=box.DOUBLE
    ))


def show_data_summary():
    """Display summary of current data state."""
    data_dir = Path("data/processed")

    table = Table(title="Data Summary", box=box.ROUNDED, show_header=True, header_style="bold magenta")
    table.add_column("Dataset", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Size", justify="right")

    datasets = [
        ("all_markets.parquet", "Markets"),
        ("selected_markets.parquet", "Selected Markets"),
        ("trades.parquet", "Trades"),
        ("wallets.parquet", "Wallets"),
        ("freshness_profiles.parquet", "Freshness Analysis"),
        ("behavior_profiles.parquet", "Behavior Analysis"),
        ("wallet_clusters.parquet", "Wallet Clusters"),
        ("wallet_profiles.parquet", "Complete Profiles"),
    ]

    for filename, name in datasets:
        file_path = data_dir / filename
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            status = "[green]✓[/green]"
            size_str = f"{size_mb:.2f} MB" if size_mb >= 1 else f"{file_path.stat().st_size / 1024:.2f} KB"
        else:
            status = "[red]✗[/red]"
            size_str = "-"

        table.add_row(name, status, size_str)

    console.print("\n")
    console.print(table)
    console.print("\n")


@app.command()
def interactive():
    """Launch interactive mode with guided workflows."""
    check_questionary()

    show_banner()

    while True:
        show_data_summary()

        main_choice = questionary.select(
            "What would you like to do?",
            choices=[
                "📊 Data Pipeline - Fetch and prepare data",
                "🔍 Analysis Pipeline - Analyze wallets and detect insiders",
                "📈 Quick View - Show data info",
                "✅ Validate Data - Check data integrity",
                "⚙️  Individual Commands - Run specific commands",
                "❌ Exit",
            ],
            style=custom_style,
        ).ask()

        if main_choice is None or "Exit" in main_choice:
            console.print("\n[cyan]Goodbye![/cyan]\n")
            break

        if "Data Pipeline" in main_choice:
            run_data_pipeline_interactive()
        elif "Analysis Pipeline" in main_choice:
            run_analysis_pipeline_interactive()
        elif "Quick View" in main_choice:
            data_info(data_dir=Path("data/processed"))
        elif "Validate Data" in main_choice:
            data_validate(data_dir=Path("data/processed"))
        elif "Individual Commands" in main_choice:
            run_individual_commands()


def run_data_pipeline_interactive():
    """Interactive workflow for data pipeline."""
    check_questionary()

    console.print("\n[bold cyan]Data Pipeline Configuration[/bold cyan]\n")

    # Ask if they want to run full pipeline or individual steps
    pipeline_choice = questionary.select(
        "Choose pipeline mode:",
        choices=[
            "🚀 Run Full Pipeline (recommended)",
            "🔧 Run Individual Steps",
            "⬅️  Back to Main Menu",
        ],
        style=custom_style,
    ).ask()

    if pipeline_choice is None or "Back" in pipeline_choice:
        return

    if "Full Pipeline" in pipeline_choice:
        # Get parameters for full pipeline
        count = questionary.text(
            "Number of markets to analyze:",
            default="100",
            validate=lambda x: x.isdigit() and int(x) > 0,
        ).ask()

        if count is None:
            return

        days = questionary.text(
            "Lookback period (days):",
            default="90",
            validate=lambda x: x.isdigit() and int(x) > 0,
        ).ask()

        if days is None:
            return

        min_volume = questionary.text(
            "Minimum volume filter (default: 0):",
            default="0",
            validate=lambda x: x.replace(".", "").isdigit(),
        ).ask()

        if min_volume is None:
            return

        # Confirm and run
        confirm = questionary.confirm(
            f"Run full pipeline with {count} markets, {days} days lookback?",
            default=True,
        ).ask()

        if confirm:
            console.print("\n[bold green]Starting full data pipeline...[/bold green]\n")
            data_run_pipeline(
                n_markets=int(count),
                lookback_days=int(days),
                output_dir=Path("data/processed"),
            )

    else:  # Individual steps
        while True:
            step_choice = questionary.select(
                "Select step to run:",
                choices=[
                    "1️⃣  Fetch Markets",
                    "2️⃣  Select Markets",
                    "3️⃣  Fetch Trades",
                    "4️⃣  Build Wallet Index",
                    "⬅️  Back",
                ],
                style=custom_style,
            ).ask()

            if step_choice is None or "Back" in step_choice:
                break

            if "Fetch Markets" in step_choice:
                data_fetch_markets(
                    output_dir=Path("data/processed")
                )

            elif "Select Markets" in step_choice:
                count = questionary.text(
                    "Number of markets:",
                    default="100",
                    validate=lambda x: x.isdigit() and int(x) > 0,
                ).ask()
                if count:
                    data_select_markets(
                        n_markets=int(count),
                        output_dir=Path("data/processed")
                    )

            elif "Fetch Trades" in step_choice:
                days = questionary.text(
                    "Lookback period (days):",
                    default="90",
                    validate=lambda x: x.isdigit() and int(x) > 0,
                ).ask()
                if days:
                    data_fetch_trades(
                        lookback_days=int(days),
                        market_file=Path("data/processed/selected_markets.parquet"),
                        output_dir=Path("data/processed")
                    )

            elif "Build Wallet Index" in step_choice:
                data_build_wallet_index(
                    trades_file=Path("data/processed/trades.parquet"),
                    output_dir=Path("data/processed")
                )


def run_analysis_pipeline_interactive():
    """Interactive workflow for analysis pipeline."""
    check_questionary()

    console.print("\n[bold cyan]Analysis Pipeline Configuration[/bold cyan]\n")

    # Ask if they want to run full pipeline or individual steps
    pipeline_choice = questionary.select(
        "Choose analysis mode:",
        choices=[
            "🚀 Run Full Analysis (recommended)",
            "🔧 Run Individual Analysis Steps",
            "⬅️  Back to Main Menu",
        ],
        style=custom_style,
    ).ask()

    if pipeline_choice is None or "Back" in pipeline_choice:
        return

    if "Full Analysis" in pipeline_choice:
        # Get parameters for full analysis
        top = questionary.text(
            "Number of top candidates to show:",
            default="50",
            validate=lambda x: x.isdigit() and int(x) > 0,
        ).ask()

        if top is None:
            return

        # Confirm and run
        confirm = questionary.confirm(
            f"Run full analysis showing top {top} candidates?",
            default=True,
        ).ask()

        if confirm:
            console.print("\n[bold green]Starting full analysis pipeline...[/bold green]\n")
            run_analysis(
                data_dir=Path("data/processed"),
                output_dir=Path("data/processed"),
                top_n=int(top)
            )

    else:  # Individual steps
        while True:
            step_choice = questionary.select(
                "Select analysis step:",
                choices=[
                    "1️⃣  Analyze Freshness (detect new wallets)",
                    "2️⃣  Analyze Behavior (trading patterns)",
                    "3️⃣  Cluster Wallets (find coordinated activity)",
                    "4️⃣  Build Complete Profiles",
                    "⬅️  Back",
                ],
                style=custom_style,
            ).ask()

            if step_choice is None or "Back" in step_choice:
                break

            if "Freshness" in step_choice:
                top = questionary.text(
                    "Number of top candidates:",
                    default="50",
                    validate=lambda x: x.isdigit() and int(x) > 0,
                ).ask()
                if top:
                    analyze_freshness(
                        data_dir=Path("data/processed"),
                        output_file=Path("data/processed/freshness_profiles.parquet"),
                        top_n=int(top)
                    )

            elif "Behavior" in step_choice:
                analyze_behavior(
                    data_dir=Path("data/processed"),
                    output_file=Path("data/processed/behavior_profiles.parquet")
                )

            elif "Cluster" in step_choice:
                methods = questionary.checkbox(
                    "Select clustering methods:",
                    choices=[
                        "temporal (timing patterns)",
                        "behavior (trading style)",
                    ],
                ).ask()
                if methods:
                    methods_str = ",".join([m.split()[0] for m in methods])
                    cluster_wallets(
                        data_dir=Path("data/processed"),
                        output_file=Path("data/processed/wallet_clusters.parquet"),
                        methods=methods_str
                    )

            elif "Complete Profiles" in step_choice:
                top = questionary.text(
                    "Number of top candidates:",
                    default="50",
                    validate=lambda x: x.isdigit() and int(x) > 0,
                ).ask()
                if top:
                    build_profiles(
                        data_dir=Path("data/processed"),
                        output_file=Path("data/processed/wallet_profiles.parquet"),
                        top_n=int(top)
                    )


def run_individual_commands():
    """Menu for running individual commands."""
    check_questionary()

    while True:
        category = questionary.select(
            "Select command category:",
            choices=[
                "📊 Data Commands",
                "🔍 Analysis Commands",
                "⬅️  Back to Main Menu",
            ],
            style=custom_style,
        ).ask()

        if category is None or "Back" in category:
            break

        if "Data Commands" in category:
            cmd = questionary.select(
                "Select data command:",
                choices=[
                    "fetch-markets",
                    "select-markets",
                    "fetch-trades",
                    "build-wallet-index",
                    "run-pipeline",
                    "info",
                    "validate",
                    "⬅️  Back",
                ],
                style=custom_style,
            ).ask()

            if cmd and cmd != "⬅️  Back":
                execute_data_command(cmd)

        elif "Analysis Commands" in category:
            cmd = questionary.select(
                "Select analysis command:",
                choices=[
                    "analyze-freshness",
                    "analyze-behavior",
                    "cluster-wallets",
                    "build-profiles",
                    "run-analysis",
                    "⬅️  Back",
                ],
                style=custom_style,
            ).ask()

            if cmd and cmd != "⬅️  Back":
                execute_analysis_command(cmd)


def execute_data_command(cmd: str):
    """Execute a data command with parameter prompts."""
    if cmd == "fetch-markets":
        data_fetch_markets(
            output_dir=Path("data/processed")
        )

    elif cmd == "select-markets":
        count = int(questionary.text("Number of markets:", default="100").ask() or "100")
        data_select_markets(
            n_markets=count,
            output_dir=Path("data/processed")
        )

    elif cmd == "fetch-trades":
        days = int(questionary.text("Lookback days:", default="90").ask() or "90")
        data_fetch_trades(
            lookback_days=days,
            market_file=Path("data/processed/selected_markets.parquet"),
            output_dir=Path("data/processed")
        )

    elif cmd == "build-wallet-index":
        data_build_wallet_index(
            trades_file=Path("data/processed/trades.parquet"),
            output_dir=Path("data/processed")
        )

    elif cmd == "run-pipeline":
        count = int(questionary.text("Number of markets:", default="100").ask() or "100")
        days = int(questionary.text("Lookback days:", default="90").ask() or "90")
        data_run_pipeline(
            n_markets=count,
            lookback_days=days,
            output_dir=Path("data/processed")
        )

    elif cmd == "info":
        data_info(data_dir=Path("data/processed"))

    elif cmd == "validate":
        data_validate(data_dir=Path("data/processed"))


def execute_analysis_command(cmd: str):
    """Execute an analysis command with parameter prompts."""
    if cmd == "analyze-freshness":
        top = int(questionary.text("Top candidates:", default="50").ask() or "50")
        analyze_freshness(
            data_dir=Path("data/processed"),
            output_file=Path("data/processed/freshness_profiles.parquet"),
            top_n=top
        )

    elif cmd == "analyze-behavior":
        analyze_behavior(
            data_dir=Path("data/processed"),
            output_file=Path("data/processed/behavior_profiles.parquet")
        )

    elif cmd == "cluster-wallets":
        methods = questionary.text(
            "Methods (comma-separated):",
            default="temporal,behavior"
        ).ask() or "temporal,behavior"
        cluster_wallets(
            data_dir=Path("data/processed"),
            output_file=Path("data/processed/wallet_clusters.parquet"),
            methods=methods
        )

    elif cmd == "build-profiles":
        top = int(questionary.text("Top candidates:", default="50").ask() or "50")
        build_profiles(
            data_dir=Path("data/processed"),
            output_file=Path("data/processed/wallet_profiles.parquet"),
            top_n=top
        )

    elif cmd == "run-analysis":
        top = int(questionary.text("Top candidates:", default="50").ask() or "50")
        run_analysis(
            data_dir=Path("data/processed"),
            output_dir=Path("data/processed"),
            top_n=top
        )


@app.command()
def quick():
    """Quick mode: Run full pipeline with default settings."""
    console.print("\n[bold cyan]Quick Mode: Running full pipeline with defaults[/bold cyan]")
    console.print("Markets: 100 | Days: 90\n")

    confirm = typer.confirm("Continue?", default=True)
    if confirm:
        data_run_pipeline(
            n_markets=100,
            lookback_days=90,
            output_dir=Path("data/processed")
        )
        console.print("\n[bold green]Data pipeline complete! Starting analysis...[/bold green]\n")
        run_analysis(
            data_dir=Path("data/processed"),
            output_dir=Path("data/processed"),
            top_n=50
        )
        console.print("\n[bold green]Analysis complete![/bold green]\n")


@app.command()
def wizard():
    """
    Alias for interactive mode - launches the full interactive wizard.
    """
    interactive()


def main():
    """Main entry point for interactive CLI."""
    app()


if __name__ == "__main__":
    main()
