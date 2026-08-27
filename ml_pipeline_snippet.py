
def run_ml_pipeline_interactive():
    """Interactive workflow for ML pipeline."""
    check_questionary()

    console.print("\n[bold cyan]ML Pipeline Configuration[/bold cyan]\n")

    while True:
        action = questionary.select(
            "Select ML action:",
            choices=[
                "🏋️  Train Model (XGBoost)",
                "🎯 Score Trades",
                "⬅️  Back to Main Menu",
            ],
            style=custom_style,
        ).ask()

        if action is None or "Back" in action:
            break

        if "Train Model" in action:
            confirm = questionary.confirm(
                "Train new model? This will overwrite existing models.",
                default=True
            ).ask()
            
            if confirm:
                try:
                    ml_train(
                        data_dir=Path("data/processed"),
                        output_model=Path("data/models/insider_model.joblib")
                    )
                except Exception as e:
                    console.print(f"[red]Training failed: {e}[/red]")

        elif "Score Trades" in action:
            confirm = questionary.confirm(
                "Score trades using existing model?",
                default=True
            ).ask()
            
            if confirm:
                try:
                    ml_score(
                        data_dir=Path("data/processed"),
                        model_path=Path("data/models/insider_model.joblib"),
                        output_file=Path("data/processed/scored_trades.parquet")
                    )
                except Exception as e:
                    console.print(f"[red]Scoring failed: {e}[/red]")
