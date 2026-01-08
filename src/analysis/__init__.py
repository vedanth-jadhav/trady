"""
Wallet analysis module for Polymarket trading analysis.

This module provides tools for analyzing wallet behavior, funding sources,
freshness, and clustering to identify potential insider trading patterns.

Phase 2 of the Trady insider detection pipeline.
"""

from .types import (
    # Enums
    FundingSource,

    # Dataclasses
    BehaviorProfile,
    ClusteringResult,
    FreshnessProfile,
    FundingProfile,
    WalletCluster,
    WalletProfile,
)

# Import analyzers
from .freshness import FreshnessAnalyzer
from .behavior import BehaviorProfiler
from .clusterer import WalletClusterer, UnionFind
from .funding import FundingTracker
from .profiler import WalletProfileBuilder, NegativeSignalDetector

__all__ = [
    # Enums
    "FundingSource",

    # Dataclasses
    "BehaviorProfile",
    "ClusteringResult",
    "FreshnessProfile",
    "FundingProfile",
    "WalletCluster",
    "WalletProfile",

    # Analyzers
    "FreshnessAnalyzer",
    "BehaviorProfiler",
    "WalletClusterer",
    "UnionFind",
    "FundingTracker",
    "WalletProfileBuilder",
    "NegativeSignalDetector",
]
