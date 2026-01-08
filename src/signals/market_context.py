"""
Market context analyzer.

Determines if a market is in a "high-insider" category and analyzes
market characteristics that affect insider likelihood.
"""

from typing import Dict, Optional

import pandas as pd

from .types import Market


class MarketContextAnalyzer:
    """
    Analyzes market context for insider likelihood.

    High-insider categories:
    - Political events (elections, appointments)
    - Corporate events (earnings, M&A)
    - Regulatory decisions
    - Crypto (insider trading common)

    Low-insider categories:
    - Sports outcomes
    - Entertainment
    - Weather events
    - Random events
    """

    HIGH_INSIDER_CATEGORIES = [
        "Politics",
        "Business",
        "Crypto",  # Insider trading common
        "Science",  # Research announcements
        "Economy",
        "Tech",
        "Legal",
    ]

    LOW_INSIDER_CATEGORIES = [
        "Sports",
        "Entertainment",
        "Weather",
        "Pop Culture",
    ]

    # Keywords that suggest high insider likelihood
    HIGH_INSIDER_KEYWORDS = [
        "election",
        "resign",
        "appointed",
        "merger",
        "acquisition",
        "earnings",
        "announcement",
        "decision",
        "ruling",
        "verdict",
        "approval",
        "launch",
        "release",
    ]

    def __init__(self, markets_df: pd.DataFrame):
        """
        Initialize analyzer with market data.

        Args:
            markets_df: DataFrame with market data
        """
        self.markets = markets_df
        self._market_cache: Dict[str, Market] = {}
        self._build_market_cache()

    def _build_market_cache(self) -> None:
        """Build market lookup cache."""
        if self.markets is None or self.markets.empty:
            return

        for _, row in self.markets.iterrows():
            market = Market.from_row(row)
            self._market_cache[market.market_id] = market

    def get_market(self, market_id: str) -> Optional[Market]:
        """Get market by ID."""
        return self._market_cache.get(market_id)

    def get_insider_category_score(self, market: Market) -> float:
        """
        Score market's insider likelihood based on category.

        Args:
            market: Market to analyze

        Returns:
            Score 0-1, higher = more likely to have insider activity
        """
        category = market.category

        # Check category
        if category in self.HIGH_INSIDER_CATEGORIES:
            base_score = 0.8
        elif category in self.LOW_INSIDER_CATEGORIES:
            base_score = 0.2
        else:
            base_score = 0.5

        # Check keywords in question
        question_lower = market.question.lower()
        keyword_boost = 0.0

        for keyword in self.HIGH_INSIDER_KEYWORDS:
            if keyword in question_lower:
                keyword_boost += 0.05

        # Cap keyword boost at 0.15
        keyword_boost = min(keyword_boost, 0.15)

        return min(base_score + keyword_boost, 1.0)

    def get_market_characteristics(self, market: Market) -> Dict:
        """
        Analyze market characteristics.

        Args:
            market: Market to analyze

        Returns:
            Dict with market characteristics:
            - is_niche: Low participation market
            - is_near_resolution: Market close to resolution
            - liquidity_tier: "low", "medium", "high"
        """
        # Determine liquidity tier
        if market.liquidity < 1000:
            liquidity_tier = "low"
        elif market.liquidity < 10000:
            liquidity_tier = "medium"
        else:
            liquidity_tier = "high"

        # Niche market: low volume
        is_niche = market.volume < 5000

        return {
            "is_niche": is_niche,
            "is_resolved": market.is_resolved,
            "liquidity_tier": liquidity_tier,
            "volume": market.volume,
            "category": market.category,
            "insider_category_score": self.get_insider_category_score(market),
        }

    def compute_market_boost(self, market: Market) -> float:
        """
        Compute market-based score boost.

        Niche + high-insider category = higher boost.

        Args:
            market: Market to analyze

        Returns:
            Boost multiplier 0.5-1.5
        """
        insider_score = self.get_insider_category_score(market)
        chars = self.get_market_characteristics(market)

        # Base is insider category score
        boost = insider_score

        # Niche markets get a boost
        if chars["is_niche"]:
            boost += 0.1

        # Low liquidity markets get a boost (easier to manipulate)
        if chars["liquidity_tier"] == "low":
            boost += 0.1

        return min(max(boost, 0.5), 1.5)
