"""
Cluster signal detector.

Detects coordinated activity across related wallets.
"""

from datetime import timedelta
from typing import Dict, List, Optional

import pandas as pd

from src.analysis import WalletCluster, WalletProfile

from .types import (
    ClusterSignalType,
    Signal,
    SignalCategory,
    Trade,
)


class ClusterSignalDetector:
    """
    Detects signals based on wallet cluster behavior.

    Signals:
    - CLUSTER_COORDINATION: Multiple cluster members trading same market
    - CLUSTER_CONCENTRATION: Cluster building concentrated position
    - NEW_CLUSTER_MEMBER: Trade from newly identified cluster member
    """

    # Thresholds
    COORDINATION_MIN_MEMBERS = 2
    COORDINATION_WINDOW_HOURS = 48
    CONCENTRATION_THRESHOLD = 10000  # $10k+

    def __init__(
        self,
        wallet_profiles: Dict[str, WalletProfile],
        clusters: List[WalletCluster],
        trades_df: pd.DataFrame,
    ):
        """
        Initialize detector with wallet and cluster data.

        Args:
            wallet_profiles: Dict mapping wallet address to WalletProfile
            clusters: List of wallet clusters
            trades_df: DataFrame with all trades
        """
        self.profiles = wallet_profiles
        self.clusters = clusters
        self.trades = trades_df
        self.wallet_to_cluster: Dict[str, WalletCluster] = {}
        self._cluster_trades_cache: Dict[str, pd.DataFrame] = {}
        self._build_lookup()

    def _build_lookup(self) -> None:
        """Build wallet -> cluster lookup."""
        for cluster in self.clusters:
            for wallet in cluster.wallets:
                self.wallet_to_cluster[wallet] = cluster

    def detect(
        self,
        trade: Trade,
        wallet_profile: WalletProfile,
    ) -> List[Signal]:
        """
        Detect cluster signals for a trade.

        Args:
            trade: The trade to analyze
            wallet_profile: Profile of the wallet

        Returns:
            List of detected signals
        """
        signals = []

        cluster = self.wallet_to_cluster.get(wallet_profile.address)
        if not cluster:
            return signals

        # Cluster coordination: Other members also traded this market
        coordination_signal = self._detect_cluster_coordination(
            trade, cluster, wallet_profile.address
        )
        if coordination_signal:
            signals.append(coordination_signal)

        # Cluster concentration: Total cluster position is large
        concentration_signal = self._detect_cluster_concentration(trade, cluster)
        if concentration_signal:
            signals.append(concentration_signal)

        # New cluster member: Wallet was recently added to cluster
        # (This would require tracking when clusters were formed - simplified here)
        if wallet_profile.total_trades <= 3 and len(cluster.wallets) > 1:
            signals.append(Signal(
                type=ClusterSignalType.NEW_CLUSTER_MEMBER,
                category=SignalCategory.CLUSTER,
                confidence=0.5,
                details={
                    "cluster_id": cluster.cluster_id,
                    "wallet_trades": wallet_profile.total_trades,
                    "cluster_size": len(cluster.wallets),
                }
            ))

        return signals

    def _detect_cluster_coordination(
        self,
        trade: Trade,
        cluster: WalletCluster,
        current_wallet: str,
    ) -> Optional[Signal]:
        """Detect if other cluster members are trading the same market."""
        other_members = [w for w in cluster.wallets if w != current_wallet]

        if not other_members:
            return None

        # Get trades from other cluster members in this market
        same_market_trades = self._get_cluster_market_activity(
            other_members,
            trade.market_id,
            trade.timestamp,
            window_hours=self.COORDINATION_WINDOW_HOURS,
        )

        if len(same_market_trades) >= self.COORDINATION_MIN_MEMBERS:
            return Signal(
                type=ClusterSignalType.CLUSTER_COORDINATION,
                category=SignalCategory.CLUSTER,
                confidence=min(len(same_market_trades) * 0.3, 1.0),
                details={
                    "cluster_id": cluster.cluster_id,
                    "coordinated_trades": len(same_market_trades),
                    "other_members_count": len(other_members),
                    "window_hours": self.COORDINATION_WINDOW_HOURS,
                }
            )

        return None

    def _detect_cluster_concentration(
        self,
        trade: Trade,
        cluster: WalletCluster,
    ) -> Optional[Signal]:
        """Detect if cluster is building a concentrated position."""
        cluster_position = self._get_cluster_market_position(
            cluster.wallets,
            trade.market_id,
        )

        if cluster_position > self.CONCENTRATION_THRESHOLD:
            return Signal(
                type=ClusterSignalType.CLUSTER_CONCENTRATION,
                category=SignalCategory.CLUSTER,
                confidence=min(cluster_position / 50000, 1.0),
                details={
                    "cluster_id": cluster.cluster_id,
                    "total_position": round(cluster_position, 2),
                    "cluster_size": len(cluster.wallets),
                }
            )

        return None

    def _get_cluster_market_activity(
        self,
        wallets: List[str],
        market_id: str,
        reference_time: pd.Timestamp,
        window_hours: int,
    ) -> List[Dict]:
        """Get trades by cluster members in market within time window."""
        if self.trades.empty:
            return []

        window = timedelta(hours=window_hours)

        # Filter trades
        mask = (
            ((self.trades["maker_address"].isin(wallets)) |
             (self.trades["taker_address"].isin(wallets))) &
            (self.trades["market_id"] == market_id) &
            (self.trades["timestamp"] >= reference_time - window) &
            (self.trades["timestamp"] <= reference_time + window)
        )

        matching_trades = self.trades[mask]

        # Return as list of dicts
        return matching_trades.to_dict("records")

    def _get_cluster_market_position(
        self,
        wallets: List[str],
        market_id: str,
    ) -> float:
        """Get total cluster position in market."""
        if self.trades.empty:
            return 0.0

        # Filter trades
        mask = (
            ((self.trades["maker_address"].isin(wallets)) |
             (self.trades["taker_address"].isin(wallets))) &
            (self.trades["market_id"] == market_id)
        )

        matching_trades = self.trades[mask]

        if matching_trades.empty:
            return 0.0

        return float(matching_trades["notional"].sum())
