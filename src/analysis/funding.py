"""
Funding Tracker for wallet funding source analysis.

Tracks and categorizes wallet funding sources to identify suspicious patterns
like privacy tool usage, cross-chain bridging, or funding from tracked wallets.

Note: Full funding analysis requires on-chain data beyond Polymarket API.
This module provides hooks for integration with blockchain data providers.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field

from .types import FundingSource, FundingProfile

logger = logging.getLogger(__name__)


@dataclass
class FundingTransaction:
    """Represents an incoming funding transaction."""
    tx_hash: str
    from_address: str
    to_address: str
    amount: float
    timestamp: datetime
    source_type: FundingSource = FundingSource.UNKNOWN


class FundingTracker:
    """
    Tracks funding sources for wallets.

    Funding Source Priority (for insider detection):
    1. Privacy tools (Tornado Cash, etc.) - CRITICAL
    2. From known tracked wallets - HIGH
    3. Cross-chain bridge - MEDIUM-HIGH
    4. CEX withdrawals - MEDIUM
    5. Direct wallet - LOW

    Note: Full funding analysis requires on-chain data beyond
    Polymarket API. This module provides hooks for integration
    with blockchain data providers (Etherscan, Alchemy, etc.)
    """

    # Known privacy tool contract addresses (Polygon)
    PRIVACY_CONTRACTS: Dict[str, str] = {
        # Tornado Cash contracts are sanctioned - detection only
        "0x94a1b5cdb22c43faab4abeb5c74999895464ddaf": "tornado_cash_polygon",
    }

    # Known CEX hot wallet addresses
    CEX_WALLETS: Dict[str, str] = {
        "0x28c6c06298d514db089934071355e5743bf21d60": "binance_hot_1",
        "0x21a31ee1afc51d94c2efccaa2092ad1028285549": "binance_hot_2",
        "0xa9d1e08c7793af67e9d92fe308d5697fb81d3e43": "coinbase_hot_1",
        "0x71660c4005ba85c37ccec55d0c4493e66fe775d3": "coinbase_hot_2",
        "0x2910543af39aba0cd09dbb2d50200b3e800a63d2": "kraken_hot_1",
        "0x53d284357ec70ce289d6d64134dfac8e511c8a3d": "kraken_hot_2",
        "0x2faf487a4414fe77e2327f0bf4ae2a264a776ad2": "ftx_hot",  # historical
    }

    # Known bridge contract addresses
    BRIDGE_CONTRACTS: Dict[str, str] = {
        "0xa0c68c638235ee32657e8f720a23cec1bfc77c77": "polygon_bridge",
        "0x40ec5b33f54e0e8a33a975908c5ba1c14e5bbbdf": "polygon_erc20_bridge",
        "0x8484ef722627bf18ca5ae6bcf031c23e6e922b30": "arbitrum_bridge",
        "0x4c36d2919e407f0cc2ee3c993ccf8ac26d9ce64e": "hop_polygon_bridge",
    }

    def __init__(
        self,
        tracked_wallets: Optional[Set[str]] = None,
        blockchain_client=None
    ):
        """
        Args:
            tracked_wallets: Set of wallet addresses we're tracking (for cross-detection)
            blockchain_client: Optional client for on-chain queries (Etherscan, Alchemy, etc.)
        """
        self.tracked_wallets = {w.lower() for w in (tracked_wallets or set())}
        self.blockchain_client = blockchain_client
        self._funding_cache: Dict[str, List[FundingTransaction]] = {}

    def classify_funding_source(self, from_address: str) -> FundingSource:
        """
        Classify the source of funds based on the sender address.

        Args:
            from_address: The address funds came from

        Returns:
            FundingSource enum with associated risk level
        """
        from_addr_lower = from_address.lower()

        # Check privacy tools first (highest risk)
        if from_addr_lower in self.PRIVACY_CONTRACTS:
            logger.info(f"Privacy tool detected: {self.PRIVACY_CONTRACTS[from_addr_lower]}")
            return FundingSource.PRIVACY_TOOL

        # Check if from a tracked wallet
        if from_addr_lower in self.tracked_wallets:
            logger.info(f"Funding from tracked wallet: {from_addr_lower[:10]}...")
            return FundingSource.TRACKED_WALLET

        # Check bridge contracts
        if from_addr_lower in self.BRIDGE_CONTRACTS:
            logger.info(f"Bridge detected: {self.BRIDGE_CONTRACTS[from_addr_lower]}")
            return FundingSource.CROSS_CHAIN

        # Check CEX wallets
        if from_addr_lower in self.CEX_WALLETS:
            logger.info(f"CEX detected: {self.CEX_WALLETS[from_addr_lower]}")
            return FundingSource.CEX_WITHDRAWAL

        # Check if it looks like a contract (basic heuristic)
        # In practice, you'd check if address has code on-chain
        # For now, mark as direct transfer
        return FundingSource.DIRECT_WALLET

    def get_funding_transactions(
        self,
        wallet: str,
        before_timestamp: Optional[datetime] = None
    ) -> List[FundingTransaction]:
        """
        Get funding transactions for a wallet.

        Note: This requires on-chain data. Without a blockchain client,
        returns an empty list.

        Args:
            wallet: Wallet address to get funding for
            before_timestamp: Only get transactions before this time

        Returns:
            List of FundingTransaction objects
        """
        wallet_lower = wallet.lower()

        # Check cache
        if wallet_lower in self._funding_cache:
            txs = self._funding_cache[wallet_lower]
            if before_timestamp:
                txs = [tx for tx in txs if tx.timestamp < before_timestamp]
            return txs

        # Without blockchain client, we can't get real funding data
        if self.blockchain_client is None:
            logger.debug(f"No blockchain client - cannot fetch funding for {wallet[:10]}...")
            return []

        # Placeholder for blockchain integration
        # In practice, you'd call:
        # txs = await self.blockchain_client.get_transactions(wallet)
        # and filter for incoming value transfers
        logger.warning("Blockchain client integration not implemented")
        return []

    def compute_funding_profile(
        self,
        wallet: str,
        funding_transactions: Optional[List[FundingTransaction]] = None
    ) -> FundingProfile:
        """
        Build complete funding profile for a wallet.

        Args:
            wallet: Wallet address to analyze
            funding_transactions: Optional pre-fetched transactions

        Returns:
            FundingProfile with risk assessment
        """
        wallet_lower = wallet.lower()

        # Get transactions if not provided
        if funding_transactions is None:
            funding_transactions = self.get_funding_transactions(wallet)

        # If no transactions, return default profile
        if not funding_transactions:
            return FundingProfile(
                wallet=wallet_lower,
                primary_funding_source=FundingSource.UNKNOWN,
                funding_sources=[FundingSource.UNKNOWN],
                funding_risk_score=0.3,  # Medium-low for unknown
                total_funded_amount=0.0,
                from_tracked_wallets=[]
            )

        # Classify all funding sources
        sources = []
        total_amount = 0.0
        from_tracked = []

        for tx in funding_transactions:
            source = self.classify_funding_source(tx.from_address)
            sources.append(source)
            total_amount += tx.amount

            if source == FundingSource.TRACKED_WALLET:
                from_tracked.append(tx.from_address.lower())

        # Determine primary source (highest risk or most common)
        source_counts = {}
        for s in sources:
            source_counts[s] = source_counts.get(s, 0) + 1

        # Prioritize by risk score, then by count
        primary_source = max(
            source_counts.keys(),
            key=lambda s: (s.risk_score, source_counts[s])
        )

        # Calculate weighted risk score (guard against empty sources)
        if sources:
            weighted_risk = sum(s.risk_score for s in sources) / len(sources)
        else:
            weighted_risk = 0.3  # Default medium-low risk

        # Boost risk if privacy tools or tracked wallets are involved
        if FundingSource.PRIVACY_TOOL in sources:
            weighted_risk = min(1.0, weighted_risk + 0.3)
        if FundingSource.TRACKED_WALLET in sources:
            weighted_risk = min(1.0, weighted_risk + 0.2)

        return FundingProfile(
            wallet=wallet_lower,
            primary_funding_source=primary_source,
            funding_sources=list(set(sources)),
            funding_risk_score=round(weighted_risk, 4),
            total_funded_amount=total_amount,
            from_tracked_wallets=list(set(from_tracked))
        )

    def compute_all_funding_profiles(
        self,
        wallets: List[str]
    ) -> Dict[str, FundingProfile]:
        """
        Compute funding profiles for all wallets.

        Args:
            wallets: List of wallet addresses to analyze

        Returns:
            Dict mapping wallet address to FundingProfile
        """
        logger.info(f"Computing funding profiles for {len(wallets)} wallets")

        profiles = {}
        for i, wallet in enumerate(wallets):
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(wallets)} wallets")

            profiles[wallet.lower()] = self.compute_funding_profile(wallet)

        logger.info(f"Completed funding analysis for {len(profiles)} wallets")
        return profiles

    def add_tracked_wallet(self, wallet: str) -> None:
        """Add a wallet to the tracked wallets set."""
        self.tracked_wallets.add(wallet.lower())

    def add_tracked_wallets(self, wallets: List[str]) -> None:
        """Add multiple wallets to the tracked wallets set."""
        self.tracked_wallets.update(w.lower() for w in wallets)

    def is_suspicious_funding(self, profile: FundingProfile) -> bool:
        """
        Check if a funding profile indicates suspicious activity.

        Args:
            profile: FundingProfile to check

        Returns:
            True if funding pattern is suspicious
        """
        # High risk score is suspicious
        if profile.funding_risk_score >= 0.7:
            return True

        # Privacy tool usage is always suspicious
        if profile.primary_funding_source == FundingSource.PRIVACY_TOOL:
            return True

        # Multiple tracked wallet funding is suspicious
        if len(profile.from_tracked_wallets) >= 2:
            return True

        return False
