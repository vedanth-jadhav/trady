# Interactive CLI for Trady

An intuitive, menu-driven interface for the Polymarket Insider Detection Bot.

## Quick Start

After installing the package, you can launch the interactive CLI in three ways:

### 1. Interactive Mode (Recommended)
```bash
trady interactive
# or
trady wizard
```

This launches a full-featured interactive menu where you can:
- View current data status at a glance
- Run complete pipelines with guided parameter selection
- Execute individual commands with prompts for each parameter
- Navigate through workflows step-by-step

### 2. Quick Mode
```bash
trady quick
```

Runs the complete pipeline with sensible defaults:
- Fetches 100 top markets
- 90-day lookback period
- Analyzes top 50 candidates
- No interaction required - just confirm and go!

### 3. Traditional CLI (Still Available)
```bash
# Data pipeline
trady-data run-pipeline --count 100 --days 90

# Analysis pipeline
trady-analysis run-analysis --top 50
```

## Features

### 📊 Visual Dashboard
The interactive mode shows you the current state of your data:
- ✓ Which datasets exist
- ✗ Which datasets are missing
- 📏 Size of each dataset

### 🎯 Guided Workflows

#### Data Pipeline Workflow
1. Choose between full pipeline or individual steps
2. Configure parameters with helpful defaults
3. See progress with rich terminal output
4. Review summary when complete

#### Analysis Pipeline Workflow
1. Choose between complete analysis or individual steps
2. Select analysis methods (freshness, behavior, clustering)
3. Configure top candidate count
4. View results in formatted tables

### 🔧 Individual Command Mode
Access any command from either module:
- **Data Commands**: fetch-markets, select-markets, fetch-trades, build-wallet-index, validate, info
- **Analysis Commands**: analyze-freshness, analyze-behavior, cluster-wallets, build-profiles

Each command prompts for its parameters interactively.

## Example Session

```bash
$ trady interactive

╔╦╗╦═╗╔═╗╔╦╗╦ ╦
 ║ ╠╦╝╠═╣ ║║╚╦╝
 ╩ ╩╚═╩ ╩═╩╝ ╩
Polymarket Insider Detection Bot

┌─────── Data Summary ───────┐
│ Dataset              Status │
├──────────────────────────── │
│ Markets                  ✓ │
│ Selected Markets         ✓ │
│ Trades                   ✓ │
│ Wallets                  ✓ │
│ Freshness Analysis       ✓ │
│ Behavior Analysis        ✓ │
│ Wallet Clusters          ✓ │
│ Complete Profiles        ✓ │
└─────────────────────────────┘

? What would you like to do?
  📊 Data Pipeline - Fetch and prepare data
> 🔍 Analysis Pipeline - Analyze wallets and detect insiders
  📈 Quick View - Show data info
  ✅ Validate Data - Check data integrity
  ⚙️  Individual Commands - Run specific commands
  ❌ Exit
```

## Installation

The interactive CLI requires the `questionary` package:

```bash
# Install in development mode (recommended)
pip install -e .

# Or install from requirements
pip install -r requirements.txt
```

The package is automatically installed with the rest of the Trady dependencies.

## Tips

1. **First Time Users**: Start with `trady quick` to run everything with defaults
2. **Regular Use**: Use `trady interactive` for full control with guided workflows
3. **Automation**: Use the traditional `trady-data` and `trady-analysis` commands for scripts
4. **Data Check**: Run `trady interactive` and select "Quick View" to see what data you have

## Keyboard Shortcuts

In interactive mode:
- `↑/↓` - Navigate menu options
- `Enter` - Select option
- `Ctrl+C` - Exit (or select Exit option)
- `Space` - Select checkbox items (for multi-select)

## What's Next?

After running the pipelines:
- Check `data/processed/` for Parquet files
- View wallet profiles with highest insider scores
- Examine clusters of coordinated wallets
- Review freshness and behavior metrics

## Troubleshooting

**"questionary is not installed" error:**
```bash
pip install questionary
```

**Import errors:**
Make sure you've installed the package:
```bash
pip install -e .
```

**Data not found:**
Run the data pipeline first:
```bash
trady interactive  # Select "Data Pipeline"
# or
trady quick
```
