
import typer
import pandas as pd
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich import box
from typing import Optional, List, Dict

from .model import InsiderScoringModel
from .labels import GroundTruthLabeler
from .features import FeatureEngineer
from .service import InsiderScoringService
from src.signals.types import Trade, TradeSignal

app = typer.Typer(
    name="trady-ml",
    help="Trady ML pipeline commands",
    add_completion=False,
)

console = Console()

def load_data(data_dir: Path):
    """Load necessary data for ML with memory optimization."""
    trades_path = data_dir / "trades.parquet"
    trades_data_path = data_dir / "trades_data"
    
    markets_path = data_dir / "selected_markets.parquet"
    profiles_path = data_dir / "wallet_profiles.parquet"
    
    # 1. Load context first (markets/profiles) to enable filtering
    markets = pd.read_parquet(markets_path) if markets_path.exists() else pd.DataFrame()
    profiles_df = pd.read_parquet(profiles_path) if profiles_path.exists() else pd.DataFrame()
    
    # Identify relevant market IDs for filtering
    valid_market_ids = set()
    if not markets.empty:
        if "market_id" in markets.columns:
            valid_market_ids = set(markets.market_id.unique())
        else:
             valid_market_ids = set(markets.index.tolist())
             
        # Normalize: Add decimal string versions of hex IDs (for Goldsky compatibility)
        decimal_ids = set()
        for mid in valid_market_ids:
            if isinstance(mid, str) and mid.startswith("0x"):
                try:
                    decimal_ids.add(str(int(mid, 16)))
                except ValueError:
                    pass
        valid_market_ids.update(decimal_ids)
    
    trades = pd.DataFrame()
    
    # 2. Smart load trades
    if trades_data_path.exists():
        console.print(f"[dim]Streaming trades from {trades_data_path}...[/dim]")
        
        # Stream and filter to save RAM
        chunks = []
        files = sorted(list(trades_data_path.glob("*.parquet")))
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[green]Filtering trades...", total=len(files))
            
            for f in files:
                try:
                    df_chunk = pd.read_parquet(f)
                    
                    # Try filtering first
                    if valid_market_ids:
                        col = "market_id" if "market_id" in df_chunk.columns else "condition_id"
                        if col in df_chunk.columns:
                            filtered_chunk = df_chunk[df_chunk[col].isin(valid_market_ids)]
                            if not filtered_chunk.empty:
                                chunks.append(filtered_chunk)
                                continue # Success, move to next file
                                
                    # FALLBACK: If we are here, either no filter or filter found nothing in this file.
                    # If we have collected NOTHING so far after checking many files, maybe our filter is too strict (no overlap).
                    # But we can't just load everything (OOM).
                    # Let's trust the filter for now, BUT if we finish with 0 chunks, we should maybe warn the user 
                    # or load a small sample to show *something*.
                    
                    pass 

                except Exception as e:
                    console.print(f"[yellow]Skipping bad file {f.name}: {e}[/yellow]")
                    
                progress.update(task, advance=1)
        
        if chunks:
            trades = pd.concat(chunks, ignore_index=True)
            console.print(f"[green]Loaded {len(trades)} relevant trades via strict filter.[/green]")
        else:
            # OPTION B: Smart Full Scan (Downsampling)
            # The user wants to leverage the full 26M dataset.
            # Loading 26M rows into RAM will crash.
            # Solution: Scan ALL files, but keep only:
            # 1. Useful Signal (High/Low price trades) -> Keep 100%
            # 2. Background Noise -> Keep 10% (Sufficient for distribution)
            
            console.print("[yellow]Selected markets not found. Switching to Smart Full Scan of 26M trades...[/yellow]")
            console.print("[dim]Strategy: Keep 100% of extreme trades (signal) + 10% random sample (noise).[/dim]")
            
            chunks = []
            import numpy as np
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console
            ) as progress:
                # Iterate ALL files
                task = progress.add_task("[cyan]Scanning full dataset...[/cyan]", total=len(files))
                
                for f in files:
                    try:
                        # Optimization: Load only essential columns
                        df = pd.read_parquet(f)
                        
                        # 1. Identify "Interesting" trades (Potential synthetic resolution)
                        # Price > 0.95 (Win) or < 0.05 (Loss)
                        signal_mask = (df.price > 0.95) | (df.price < 0.05)
                        
                        # 2. Random sample of the rest (Noise)
                        # We use a 10% sample which gives us ~2.5M rows total from 26M, fitting in RAM
                        noise_mask = np.random.random(len(df)) < 0.10
                        
                        # Combine filters
                        keep_mask = signal_mask | noise_mask
                        selected = df[keep_mask]
                        
                        if not selected.empty:
                            chunks.append(selected)
                            
                    except Exception as e:
                         # console.print(f"[dim]Skipping {f.name}: {e}[/dim]")
                         pass
                    progress.update(task, advance=1)

            if chunks:
                trades = pd.concat(chunks, ignore_index=True)
                console.print(f"[green]Smart Scan complete: Loaded {len(trades)} representative trades.[/green]")
            
    elif trades_path.exists():
        trades = pd.read_parquet(trades_path)
        # Apply filter immediately
        if valid_market_ids and not trades.empty:
             col = "market_id" if "market_id" in trades.columns else "condition_id"
             if col in trades.columns:
                 trades = trades[trades[col].isin(valid_market_ids)]
    
    return trades, markets, profiles_df
    
from datetime import datetime
from src.analysis.types import WalletProfile

def _convert_trades(df: pd.DataFrame) -> List[Trade]:
    """Convert trades DataFrame to Trade objects."""
    trades = []
    # Identify wallet column (maker_address or wallet)
    wallet_col = "maker_address" if "maker_address" in df.columns else "wallet"
    
    for _, row in df.iterrows():
        try:
            trades.append(Trade.from_row(row, wallet_column=wallet_col))
        except Exception:
            continue
    return trades

def _convert_profiles(df: pd.DataFrame) -> Dict[str, WalletProfile]:
    """Convert profiles DataFrame to WalletProfile objects."""
    profiles = {}
    if df.empty:
        return profiles
        
    for _, row in df.iterrows():
        try:
            # Map columns to WalletProfile fields
            # Handle potential missing columns with defaults
            profile = WalletProfile(
                address=row.get("wallet", row.get("address", "")),
                # Basic
                total_trades=row.get("total_trades", 0),
                total_volume=row.get("total_volume", 0.0),
                unique_markets=row.get("unique_markets", 0),
                is_whale=row.get("is_whale", False),
                
                # Freshness
                freshness_score=row.get("freshness_score", 0.0),
                is_zero_history=row.get("is_zero_history", False),
                is_new_to_polymarket=row.get("is_new_to_polymarket", False),
                days_on_polymarket=row.get("days_on_polymarket", 0),
                
                # Funding
                primary_funding_source=row.get("primary_funding_source", "unknown"),
                funding_risk_score=row.get("funding_risk_score", 0.0),
                has_privacy_funding=row.get("has_privacy_funding", False),
                
                # Behavior
                retail_likelihood=row.get("retail_likelihood", 0.5),
                sophistication_score=row.get("sophistication_score", 0.5),
                win_rate=row.get("win_rate", None),
                off_hours_ratio=row.get("off_hours_ratio", 0.0),
                burst_episodes=row.get("burst_episodes", 0),
                avg_trade_size=row.get("avg_trade_size", 0.0),
                max_trade_size=row.get("max_trade_size", 0.0),
                
                # Performance
                sharpe_ratio=row.get("sharpe_ratio", None),
                early_exit_ratio=row.get("early_exit_ratio", 0.0),
            )
            profiles[profile.address] = profile
        except Exception:
            continue
    return profiles

def _load_signals(data_dir: Path) -> Dict[str, TradeSignal]:
    """
    Load pre-computed signals if available. 
    In the future this could compute them on the fly.
    """
    # For now return empty, FeatureEngineer will calculate raw features mostly
    # or we can implement loading logic here later
    return {}

@app.command()
def train(
    data_dir: Path = typer.Option(Path("data/processed"), "--data", "-d", help="Data directory"),
    output_model: Path = typer.Option(Path("data/models/insider_model.joblib"), "--output", "-o", help="Output model path"),
):
    """Train the XGBoost insider scoring model."""
    console.print("[bold blue]Training Insider Scoring Model...[/bold blue]")
    
    # 1. Load Data
    with console.status("Loading data..."):
        trades_df, markets_df, profiles_df = load_data(data_dir)
        if trades_df.empty:
            console.print("[red]No trades data found.[/red]")
            raise typer.Exit(1)
            
        trades = _convert_trades(trades_df)
        profiles = _convert_profiles(profiles_df)
        signals = _load_signals(data_dir)
        console.print(f"Loaded {len(trades)} trades and {len(profiles)} profiles.")

        # --- SYNTHETIC RESOLUTION LOGIC ---
        # If we have no resolved markets overlapping with our data (common when fetching recent data vs old keys),
        # we must synthetize resolution based on price to enable training.
        resolved_count = markets_df[markets_df['is_resolved'] == True].shape[0] if 'is_resolved' in markets_df.columns else 0
        console.print(f"[dim]Metadata has {resolved_count} resolved markets.[/dim]")
        
        # Check if we need to synthesize
        # We synthesize if we detect implicitly resolved markets in the trades (price > 0.98 or < 0.02)
        # matches logic: Active markets that effectively ended.
        console.print("[yellow]Scanning for implicitly resolved markets (price > 0.98)...[/yellow]")
        
        # Identify high-confidence winners in the loaded trades
        # Group by market_id to find max price
        if not trades_df.empty:
            market_col = "market_id" if "market_id" in trades_df.columns else "condition_id"
            if market_col in trades_df.columns:
                # Find markets that hit 0.99 (Winner = Yes)
                max_prices = trades_df.groupby(market_col)['price'].max()
                winners = max_prices[max_prices > 0.98].index.tolist()
                
                if winners:
                    console.print(f"[green]Found {len(winners)} implicitly resolved markets (Winners)![/green]")
                    
                    # Ensure columns exist
                    if 'is_resolved' not in markets_df.columns:
                        markets_df['is_resolved'] = False
                    if 'resolution' not in markets_df.columns:
                        markets_df['resolution'] = None
                        
                    # We need to map decimal IDs (from trades) back to whatever format markets_df uses (likely Hex)
                    # Or just ensure we can match them.
                    # Simplest way: Add new rows to markets_df for these winners if they aren't there, 
                    # or update existing ones.
                    
                    # Create a lookup for markets_df index/id
                    # We'll just force update any that match
                    
                    # 1. Normalize markets_df IDs for matching
                    markets_df['temp_id_str'] = markets_df['market_id'].astype(str) if 'market_id' in markets_df.columns else markets_df.index.astype(str)
                    
                    # 2. Convert winners (decimal strings) to set
                    winner_set = set(winners)
                    
                    # 3. Also handle hex conversion matching
                    def normalize_to_decimal(val):
                        if isinstance(val, str) and val.startswith("0x"):
                            try:
                                return str(int(val, 16))
                            except: return val
                        return str(val)
                        
                    markets_df['decimal_id'] = markets_df['temp_id_str'].apply(normalize_to_decimal)
                    
                    # 4. Mark matches as resolved
                    mask = markets_df['decimal_id'].isin(winner_set)
                    match_count = mask.sum()
                    
                    if match_count > 0:
                        markets_df.loc[mask, 'is_resolved'] = True
                        markets_df.loc[mask, 'resolution'] = "Yes" # Assume Yes for > 0.98
                        console.print(f"[green]Updated {match_count} existing markets to Resolved status.[/green]")
                    
                    # 5. Determine if we need to ADD missing markets (if trades reference markets not in metadata)
                    # For minimal training, we might not need to add the metadata rows if the labeler handles missing markets gracefully (it returns 0.0).
                    # But then we get 0 positives. 
                    # So we MUST have the market in `markets_df` for it to be labeled.
                    
                    # Identify winners NOT in markets_df
                    existing_decimals = set(markets_df['decimal_id'])
                    missing_winners = [w for w in winners if w not in existing_decimals]
                    
                    if missing_winners:
                        console.print(f"[dim]Synthesizing metadata for {len(missing_winners)} missing winner markets...[/dim]")
                        new_rows = []
                        for mid in missing_winners:
                            new_rows.append({
                                'market_id': mid,
                                'is_resolved': True,
                                'resolution': 'Yes', # Implied
                                'decimal_id': mid,
                                'volume_tier': 'medium' # Default
                            })
                        new_markets = pd.DataFrame(new_rows)
                        markets_df = pd.concat([markets_df, new_markets], ignore_index=True)
                        
                    console.print(f"[bold green]Total Resolved Markets for Training: {markets_df[markets_df['is_resolved']==True].shape[0]}[/bold green]")
        # -------------------------------

    # 2. Initialize Components
    feature_engineer = FeatureEngineer(
        trades_df=trades_df,
        markets_df=markets_df,
        wallet_profiles=profiles,
        signals=signals
    )
    
    labeler = GroundTruthLabeler(
        trades_df=trades_df,
        markets_df=markets_df,
        wallet_profiles=profiles
    )
    
    # 3. Prepare Dataset (OPTIMIZED - O(N) instead of O(N²))
    from tqdm import tqdm
    
    console.print("[dim]Extracting features (vectorized batch)...[/dim]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        # Step 1: Generate labels (vectorized)
        task1 = progress.add_task("[green]Generating labels...", total=100)
        y_series = labeler.generate_labels_batch(trades_df)
        progress.update(task1, completed=100)
        
        # Filter out zero/invalid labels - keep only trades with meaningful labels
        # For training, we want trades from resolved markets with positive OR negative signals
        valid_mask = y_series.notna()
        
        # Step 2: Extract features (vectorized)
        task2 = progress.add_task("[blue]Extracting features...", total=100)
        X = feature_engineer.extract_features_batch(trades_df)
        progress.update(task2, completed=100)
        
        # Apply the valid mask to both X and y
        X = X.loc[valid_mask]
        y_series = y_series.loc[valid_mask]
        
    valid_count = len(X)
    
    if valid_count < 10:
        console.print("[red]Not enough labeled data points for training (found < 10).[/red]")
        console.print("Ensure you have resolved markets in your dataset.")
        raise typer.Exit(1)
        
    # Check label distribution
    if len(y_series.unique()) < 2:
        console.print(f"[red]Error: Only one class found in labels: {y_series.unique()}[/red]")
        console.print("Cannot train model with only one class. Ensure you have both positive and negative examples.")
        # Verify if we have any resolved markets
        resolved_count = markets_df["is_resolved"].sum() if "is_resolved" in markets_df.columns else "Unknown"
        console.print(f"Resolved markets: {resolved_count}")
        raise typer.Exit(1)

    # Convert to binary for XGBClassifier if they are continuous
    # This is important as XGBClassifier expects classes
    y_binary = (y_series > 0.5).astype(int)
    if len(y_binary.unique()) < 2:
         console.print("[red]Error: After thresholding at 0.5, only one class remains.[/red]")
         raise typer.Exit(1)
         
    console.print(f"Prepared {len(X)} samples for training.")
    console.print(f"Class balance: {y_binary.value_counts().to_dict()}")
    y_series = y_binary # Use binary labels for training
    
    # Downsample if too large (prevent XGBoost OOM on 8GB machines)
    MAX_TRAIN_SAMPLES = 500_000
    if len(X) > MAX_TRAIN_SAMPLES:
        console.print(f"[yellow]Dataset too large ({len(X):,} rows). Downsampling to {MAX_TRAIN_SAMPLES:,} for training...[/yellow]")
        
        # Stratified sampling to preserve class balance
        from sklearn.model_selection import train_test_split
        
        # Keep all positive samples (minority class) and sample negatives
        pos_mask = y_series == 1
        neg_mask = y_series == 0
        
        n_pos = pos_mask.sum()
        n_neg_needed = MAX_TRAIN_SAMPLES - n_pos
        
        if n_neg_needed > 0 and neg_mask.sum() > n_neg_needed:
            # Random sample of negatives
            neg_indices = y_series[neg_mask].sample(n=n_neg_needed, random_state=42).index
            keep_indices = y_series[pos_mask].index.tolist() + neg_indices.tolist()
            X = X.loc[keep_indices]
            y_series = y_series.loc[keep_indices]
        
        console.print(f"[green]Downsampled to {len(X):,} samples. New class balance: {y_series.value_counts().to_dict()}[/green]")
    
    # Convert to float32 to reduce memory usage
    X = X.astype('float32')

    # 4. Train Model
    with console.status("Training model..."):
        model = InsiderScoringModel()
        model.train(X, y_series)
        
        # Create output dir if needed
        output_model.parent.mkdir(parents=True, exist_ok=True)
        model.save(output_model)
    
    console.print(f"[green]✓ Model trained and saved to {output_model}[/green]")

@app.command()
def score(
    data_dir: Path = typer.Option(Path("data/processed"), "--data", "-d", help="Data directory"),
    model_path: Path = typer.Option(Path("data/models/insider_model.joblib"), "--model", "-m", help="Model path"),
    output_file: Path = typer.Option(Path("data/processed/scored_trades.parquet"), "--output", "-o", help="Output file"),
    limit: int = typer.Option(10000, "--limit", "-l", help="Max trades to score (default 10K, use 0 for all)"),
    high_signal: bool = typer.Option(False, "--high-signal", "-s", help="Only score high-signal trades (resolved markets, extreme prices)"),
):
    """Score trades using the trained model."""
    console.print("[bold blue]Scoring trades...[/bold blue]")
    
    # 1. Load Data
    with console.status("Loading data..."):
        trades_df, markets_df, profiles_df = load_data(data_dir)
        if trades_df.empty:
            console.print("[red]No trades data found.[/red]")
            raise typer.Exit(1)
        
        # Apply high-signal filter if requested
        if high_signal:
            console.print("[cyan]Filtering for high-signal trades only...[/cyan]")
            
            # Get resolved market IDs
            resolved_ids = set()
            if 'is_resolved' in markets_df.columns:
                resolved_ids = set(markets_df[markets_df['is_resolved'] == True]['market_id'].astype(str).tolist())
            
            # Also find implicitly resolved (price > 0.95 or < 0.05)
            market_col = "market_id" if "market_id" in trades_df.columns else "condition_id"
            extreme_price_mask = (trades_df['price'] > 0.95) | (trades_df['price'] < 0.05)
            
            # Filter: resolved market OR extreme price
            in_resolved = trades_df[market_col].astype(str).isin(resolved_ids)
            trades_df = trades_df[in_resolved | extreme_price_mask]
            
            console.print(f"[green]Found {len(trades_df):,} high-signal trades[/green]")
        
        # Apply limit to prevent resource exhaustion
        if limit > 0 and len(trades_df) > limit:
            console.print(f"[yellow]Limiting to {limit:,} trades (out of {len(trades_df):,}). Use --limit 0 for all.[/yellow]")
            trades_df = trades_df.head(limit)
        trades = _convert_trades(trades_df)
        profiles = _convert_profiles(profiles_df)
        signals = _load_signals(data_dir)
    
    # 2. Load Model
    try:
        model = InsiderScoringModel.load(model_path)
        console.print(f"Loaded model from {model_path}")
    except (FileNotFoundError, Exception):
        console.print("[yellow]Model not found or could not load. Using rule-based overrides only.[/yellow]")
        model = InsiderScoringModel()

    # 3. Initialize Service
    feature_engineer = FeatureEngineer(
        trades_df=trades_df,
        markets_df=markets_df,
        wallet_profiles=profiles,
        signals=signals
    )
    
    service = InsiderScoringService(model, feature_engineer)

    # 4. Score Trades
    with console.status("Scoring trades..."):
        results = service.score_batch(trades)
        
    if not results:
        console.print("[yellow]No trades scored.[/yellow]")
        return

    # 5. Save Results
    output_records = []
    for r in results:
        rec = {
            "trade_id": r.trade.trade_id,
            "wallet": r.trade.wallet,
            "market_id": r.trade.market_id,
            "ml_score": r.ml_score,
            "final_score": r.final_score,
            "confidence_tier": r.confidence_tier,
            **r.features
        }
        output_records.append(rec)
        
    df_out = pd.DataFrame(output_records)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(output_file, index=False)
    
    console.print(f"[green]✓ Scored {len(df_out)} trades. Saved to {output_file}[/green]")
    
    # Show summary of high scores
    high_scores = df_out[df_out["final_score"] > 0.7]
    if not high_scores.empty:
        console.print(f"\n[bold]Found {len(high_scores)} high-probability insider trades:[/bold]")
        table = Table(box=box.SIMPLE)
        table.add_column("Wallet", style="cyan")
        table.add_column("Score", style="magenta")
        table.add_column("Conf", style="green")
        
        for _, row in high_scores.sort_values("final_score", ascending=False).head(10).iterrows():
            table.add_row(
                row["wallet"][:10] + "...",
                f"{row['final_score']:.3f}",
                row["confidence_tier"]
            )
        console.print(table)


@app.command()
def suspects(
    scored_file: Path = typer.Option(Path("data/processed/scored_trades.parquet"), "--input", "-i", help="Scored trades file"),
    top_n: int = typer.Option(20, "--top", "-n", help="Number of top suspects to show"),
    min_trades: int = typer.Option(2, "--min-trades", "-m", help="Minimum trades per wallet"),
):
    """List most suspicious wallets from scored trades."""
    import pandas as pd
    
    if not scored_file.exists():
        console.print(f"[red]Scored trades file not found: {scored_file}[/red]")
        console.print("[yellow]Run 'trady-ml score --high-signal' first.[/yellow]")
        raise typer.Exit(1)
    
    console.print("[bold blue]Analyzing suspicious wallets...[/bold blue]")
    
    df = pd.read_parquet(scored_file)
    
    # Add percentile ranking
    df['score_percentile'] = df['final_score'].rank(pct=True) * 100
    
    # Aggregate by wallet
    wallet_stats = df.groupby('wallet').agg({
        'final_score': ['mean', 'max', 'sum'],
        'trade_id': 'count',
        'trade_notional': 'sum',
        'score_percentile': 'mean'
    }).round(6)
    
    wallet_stats.columns = ['avg_score', 'max_score', 'total_score', 'trade_count', 'total_volume', 'avg_percentile']
    wallet_stats = wallet_stats.reset_index()
    
    # Filter by min trades
    wallet_stats = wallet_stats[wallet_stats['trade_count'] >= min_trades]
    
    # Rank by total_score (sum of all trade scores for that wallet)
    wallet_stats = wallet_stats.sort_values('total_score', ascending=False).head(top_n)
    
    # Display results
    console.print(f"\n[bold]Top {len(wallet_stats)} Suspicious Wallets[/bold] (min {min_trades} trades):\n")
    
    table = Table(box=box.ROUNDED)
    table.add_column("Rank", style="dim")
    table.add_column("Wallet", style="cyan")
    table.add_column("Trades", justify="right")
    table.add_column("Volume ($)", justify="right", style="green")
    table.add_column("Avg Pctl", justify="right", style="magenta")
    table.add_column("Total Score", justify="right", style="yellow")
    
    for i, (_, row) in enumerate(wallet_stats.iterrows(), 1):
        pctl_color = "red" if row['avg_percentile'] > 90 else ("yellow" if row['avg_percentile'] > 75 else "white")
        table.add_row(
            str(i),
            row['wallet'],  # Full wallet address
            str(int(row['trade_count'])),
            f"${row['total_volume']:,.2f}",
            f"[{pctl_color}]{row['avg_percentile']:.1f}%[/{pctl_color}]",
            f"{row['total_score']:.6f}"
        )
    
    console.print(table)
    
    # Summary stats
    console.print(f"\n[dim]Based on {len(df):,} scored trades[/dim]")


if __name__ == "__main__":
    app()
