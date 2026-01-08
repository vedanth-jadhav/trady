"""
Signal detection module for Polymarket insider detection.

This module provides tools for detecting suspicious trading patterns
that may indicate insider activity. It analyzes trades across multiple
signal categories:

- Freshness: New wallets, recently funded, low activity
- Timing: Pre-news trades, off-hours, pre-spike, burst trading
- Sizing: Whale trades, unusual sizes, concentrated positions
- Funding: Privacy tools, tracked wallets, bridges, CEX
- Cluster: Coordinated activity across related wallets

Phase 3 of the Trady insider detection pipeline.
"""

from .types import (
    # Enums
    SignalCategory,
    FreshnessSignalType,
    TimingSignalType,
    SizingSignalType,
    FundingSignalType,
    ClusterSignalType,
    # Dataclasses
    Signal,
    AggregatedSignal,
    Trade,
    Market,
    TradeSignal,
)

from .freshness import FreshnessSignalDetector
from .timing import TimingSignalDetector
from .sizing import SizingSignalDetector
from .funding import FundingSignalDetector
from .cluster import ClusterSignalDetector
from .market_context import MarketContextAnalyzer
from .negative_filter import NegativeSignalFilter
from .aggregator import SignalAggregator
from .detector import InsiderSignalDetector

__all__ = [
    # Enums
    "SignalCategory",
    "FreshnessSignalType",
    "TimingSignalType",
    "SizingSignalType",
    "FundingSignalType",
    "ClusterSignalType",
    # Dataclasses
    "Signal",
    "AggregatedSignal",
    "Trade",
    "Market",
    "TradeSignal",
    # Detectors
    "FreshnessSignalDetector",
    "TimingSignalDetector",
    "SizingSignalDetector",
    "FundingSignalDetector",
    "ClusterSignalDetector",
    "MarketContextAnalyzer",
    "NegativeSignalFilter",
    "SignalAggregator",
    "InsiderSignalDetector",
]
