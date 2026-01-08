"""
Wallet Clustering Module

Identifies related wallets that may belong to the same entity using multiple methods:
1. Temporal clustering: Wallets trading at similar times on same markets
2. Behavioral clustering: Similar trading patterns (ML-based)
3. Position correlation: Correlated position changes

Critical for detecting coordinated insider trading.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass
import logging
import uuid

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from .types import WalletCluster, ClusteringResult, BehaviorProfile

logger = logging.getLogger(__name__)


class UnionFind:
    """Union-Find data structure for merging clusters."""

    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1


class WalletClusterer:
    """
    Clusters related wallets using multiple methods.

    Clustering Methods:
    1. Temporal Clustering: Wallets trading at similar times on same markets
    2. Behavioral Clustering: Similar trading patterns (uses scikit-learn)
    3. Position Correlation: Correlated position changes

    Uses combination of heuristic rules and ML clustering.
    """

    def __init__(
        self,
        trades_df: pd.DataFrame,
        behavior_profiles: Optional[Dict[str, BehaviorProfile]] = None,
        wallet_volumes: Optional[Dict[str, float]] = None
    ):
        self.trades = trades_df
        self.behavior_profiles = behavior_profiles or {}
        self.wallet_volumes = wallet_volumes or {}

    def cluster_by_temporal_proximity(
        self,
        time_window_minutes: int = 5,
        min_co_occurrences: int = 3
    ) -> List[WalletCluster]:
        """
        Cluster wallets that frequently trade the same market within a short time window.

        OPTIMIZED: Uses time-bucket grouping instead of O(n²) pairwise comparison.
        """
        if self.trades.empty:
            logger.warning("No trades data available for temporal clustering")
            return []

        if 'timestamp' not in self.trades.columns:
            logger.warning("No timestamp column in trades data")
            return []

        df = self.trades.copy()
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Create time buckets (trades in same bucket are "co-occurring")
        df['time_bucket'] = df['timestamp'].dt.floor(f'{time_window_minutes}min')

        # Group by market + time bucket to find wallets trading together
        logger.info("Computing temporal co-occurrences using bucket method...")

        co_occurrence_counts = defaultdict(int)

        # Group by market and time bucket
        grouped = df.groupby(['market_id', 'time_bucket'])['wallet'].apply(list)

        for wallets in grouped:
            if len(wallets) < 2:
                continue
            # Get unique wallets in this bucket
            unique_wallets = list(set(wallets))
            if len(unique_wallets) < 2:
                continue
            # Count pairs
            for i in range(len(unique_wallets)):
                for j in range(i + 1, len(unique_wallets)):
                    pair = tuple(sorted([unique_wallets[i], unique_wallets[j]]))
                    co_occurrence_counts[pair] += 1

        # Filter pairs with sufficient co-occurrences
        significant_pairs = {
            pair for pair, count in co_occurrence_counts.items()
            if count >= min_co_occurrences
        }

        logger.info(f"Found {len(significant_pairs)} significant wallet pairs")

        if not significant_pairs:
            logger.info("No significant temporal correlations found")
            return []

        # Use Union-Find to cluster connected wallets
        uf = UnionFind()
        for wallet1, wallet2 in significant_pairs:
            uf.union(wallet1, wallet2)

        # Group wallets by cluster
        clusters_dict = defaultdict(list)
        for wallet1, wallet2 in significant_pairs:
            root = uf.find(wallet1)
            if wallet1 not in clusters_dict[root]:
                clusters_dict[root].append(wallet1)
            if wallet2 not in clusters_dict[root]:
                clusters_dict[root].append(wallet2)

        # Build WalletCluster objects
        clusters = []
        for wallets in clusters_dict.values():
            if len(wallets) < 2:
                continue

            # Calculate total volume and trades
            cluster_trades = df[df['wallet'].isin(wallets)]
            total_volume = cluster_trades['size'].sum() if 'size' in cluster_trades.columns else 0
            combined_trades = len(cluster_trades)

            # Find primary wallet (highest volume)
            wallet_vols = {w: self.wallet_volumes.get(w, 0) for w in wallets}
            primary_wallet = max(wallet_vols, key=wallet_vols.get)

            # Calculate confidence based on co-occurrence strength
            avg_co_occurrence = sum(
                co_occurrence_counts[tuple(sorted([w1, w2]))]
                for w1 in wallets for w2 in wallets if w1 < w2
            ) / (len(wallets) * (len(wallets) - 1) / 2) if len(wallets) > 1 else 0

            confidence = min(0.99, avg_co_occurrence / 10)  # Normalize to 0-1

            cluster = WalletCluster(
                cluster_id=str(uuid.uuid4()),
                wallets=wallets,
                primary_wallet=primary_wallet,
                clustering_method="temporal",
                confidence=confidence,
                total_volume=float(total_volume),
                combined_trades=combined_trades
            )
            clusters.append(cluster)

        logger.info(f"Found {len(clusters)} temporal clusters from {len(significant_pairs)} wallet pairs")
        return clusters

    def cluster_by_behavior(
        self,
        similarity_threshold: float = 0.7,
        min_trades: int = 5
    ) -> List[WalletCluster]:
        """
        Cluster wallets with similar trading behavior using DBSCAN.

        Features for clustering:
        - avg_trade_size (normalized)
        - avg_trades_per_day
        - off_hours_ratio
        - market_concentration
        - size_variance

        Uses scikit-learn DBSCAN for density-based clustering.
        """
        if not self.behavior_profiles:
            logger.warning("No behavior profiles available for behavioral clustering")
            return []

        # Filter wallets with minimum trades
        if not self.trades.empty and 'wallet' in self.trades.columns:
            trade_counts = self.trades['wallet'].value_counts()
            valid_wallets = set(trade_counts[trade_counts >= min_trades].index)
            profiles = {
                w: p for w, p in self.behavior_profiles.items()
                if w in valid_wallets
            }
        else:
            profiles = self.behavior_profiles

        if len(profiles) < 2:
            logger.info("Not enough wallets with behavior profiles for clustering")
            return []

        # Extract features
        wallets = list(profiles.keys())
        features = []

        for wallet in wallets:
            profile = profiles[wallet]
            features.append([
                profile.avg_trade_size,
                profile.avg_trades_per_day,
                profile.off_hours_ratio,
                profile.market_concentration,
                profile.size_variance
            ])

        features_array = np.array(features)

        # Handle missing values
        features_array = np.nan_to_num(features_array, nan=0.0)

        # Normalize features with zero-variance handling
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_array)

        # Replace NaN/inf values that can occur with zero-variance features
        features_scaled = np.nan_to_num(features_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        # DBSCAN clustering
        # eps is related to similarity_threshold (lower threshold = higher eps)
        eps = (1 - similarity_threshold) * 2
        dbscan = DBSCAN(eps=eps, min_samples=2, metric='euclidean')
        cluster_labels = dbscan.fit_predict(features_scaled)

        # Group wallets by cluster label
        clusters_dict = defaultdict(list)
        for wallet, label in zip(wallets, cluster_labels):
            if label != -1:  # -1 is noise in DBSCAN
                clusters_dict[label].append(wallet)

        # Build WalletCluster objects
        clusters = []
        for label, cluster_wallets in clusters_dict.items():
            if len(cluster_wallets) < 2:
                continue

            # Calculate total volume and trades
            if not self.trades.empty:
                cluster_trades = self.trades[self.trades['wallet'].isin(cluster_wallets)]
                total_volume = cluster_trades['size'].sum() if 'size' in cluster_trades.columns else 0
                combined_trades = len(cluster_trades)
            else:
                total_volume = sum(self.wallet_volumes.get(w, 0) for w in cluster_wallets)
                combined_trades = sum(profiles[w].avg_trades_per_day * 30 for w in cluster_wallets)

            # Find primary wallet
            wallet_vols = {w: self.wallet_volumes.get(w, 0) for w in cluster_wallets}
            primary_wallet = max(wallet_vols, key=wallet_vols.get) if wallet_vols else cluster_wallets[0]

            # Calculate confidence based on cluster cohesion
            # Use silhouette-like score
            cluster_indices = [wallets.index(w) for w in cluster_wallets]
            cluster_features = features_scaled[cluster_indices]

            if len(cluster_features) > 1:
                # Calculate average intra-cluster distance
                from scipy.spatial.distance import pdist
                distances = pdist(cluster_features, metric='euclidean')
                avg_distance = np.mean(distances) if len(distances) > 0 else 0
                confidence = max(0.0, min(0.99, 1 - avg_distance / 4))
            else:
                confidence = 0.5

            cluster = WalletCluster(
                cluster_id=str(uuid.uuid4()),
                wallets=cluster_wallets,
                primary_wallet=primary_wallet,
                clustering_method="behavior",
                confidence=confidence,
                total_volume=float(total_volume),
                combined_trades=int(combined_trades)
            )
            clusters.append(cluster)

        logger.info(f"Found {len(clusters)} behavioral clusters ({len(cluster_labels[cluster_labels == -1])} noise points)")
        return clusters

    def cluster_by_position_correlation(
        self,
        correlation_threshold: float = 0.7,
        min_shared_markets: int = 2
    ) -> List[WalletCluster]:
        """
        Cluster wallets with correlated position changes.

        Method:
        - Build position time series per wallet per market
        - Compute pairwise Pearson correlation
        - Cluster highly correlated wallets using hierarchical clustering
        """
        if self.trades.empty:
            logger.warning("No trades data available for correlation clustering")
            return []

        required_cols = ['wallet', 'market_id', 'timestamp', 'size']
        if not all(col in self.trades.columns for col in required_cols):
            logger.warning(f"Missing required columns for correlation clustering: {required_cols}")
            return []

        df = self.trades.copy()

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Add side indicator (assume 'side' column or default to 1)
        if 'side' not in df.columns:
            df['side'] = 1  # Default to buy
        else:
            # Convert side to numeric (-1 for sell, 1 for buy)
            df['side'] = df['side'].map({'BUY': 1, 'SELL': -1, 'buy': 1, 'sell': -1})
            df['side'] = df['side'].fillna(1)

        # Calculate signed position change
        df['position_change'] = df['size'] * df['side']

        # Create time buckets (hourly)
        df['time_bucket'] = df['timestamp'].dt.floor('H')

        # Aggregate position changes per wallet per market per time bucket
        position_series = df.groupby(['wallet', 'market_id', 'time_bucket'])['position_change'].sum().reset_index()

        # Find wallets with shared markets using vectorized operations
        market_wallets = defaultdict(set)
        for wallet, market_id in zip(position_series['wallet'], position_series['market_id']):
            market_wallets[market_id].add(wallet)

        # Find wallet pairs with enough shared markets
        wallet_pairs = defaultdict(set)
        for market_id, wallets in market_wallets.items():
            if len(wallets) < 2:
                continue
            for w1 in wallets:
                for w2 in wallets:
                    if w1 < w2:
                        wallet_pairs[tuple(sorted([w1, w2]))].add(market_id)

        valid_pairs = {
            pair: markets for pair, markets in wallet_pairs.items()
            if len(markets) >= min_shared_markets
        }

        if not valid_pairs:
            logger.info("No wallet pairs with sufficient shared markets")
            return []

        # Build correlation matrix
        all_wallets = set()
        for w1, w2 in valid_pairs.keys():
            all_wallets.add(w1)
            all_wallets.add(w2)

        wallets = sorted(all_wallets)
        n = len(wallets)

        if n < 2:
            logger.info("Not enough wallets for correlation clustering")
            return []

        # Create position time series for each wallet
        wallet_series = {}
        for wallet in wallets:
            wallet_data = position_series[position_series['wallet'] == wallet]
            # Pivot to get time series per market
            series = wallet_data.pivot_table(
                index='time_bucket',
                columns='market_id',
                values='position_change',
                aggfunc='sum',
                fill_value=0
            )
            # Flatten to single time series (sum across markets per time bucket)
            wallet_series[wallet] = series.sum(axis=1)

        # Align time series and compute correlations
        correlation_matrix = np.eye(n)

        for i, w1 in enumerate(wallets):
            for j, w2 in enumerate(wallets):
                if i >= j:
                    continue

                pair = tuple(sorted([w1, w2]))
                if pair not in valid_pairs:
                    correlation_matrix[i, j] = 0
                    correlation_matrix[j, i] = 0
                    continue

                # Align time series
                s1 = wallet_series[w1]
                s2 = wallet_series[w2]

                # Get common time buckets
                common_index = s1.index.intersection(s2.index)

                if len(common_index) < 2:
                    correlation_matrix[i, j] = 0
                    correlation_matrix[j, i] = 0
                    continue

                s1_aligned = s1.loc[common_index]
                s2_aligned = s2.loc[common_index]

                # Compute Pearson correlation
                if s1_aligned.std() > 0 and s2_aligned.std() > 0:
                    corr = np.corrcoef(s1_aligned, s2_aligned)[0, 1]
                    corr = 0 if np.isnan(corr) else corr
                else:
                    corr = 0

                correlation_matrix[i, j] = corr
                correlation_matrix[j, i] = corr

        # Convert correlation to distance (1 - correlation)
        distance_matrix = 1 - np.abs(correlation_matrix)

        # Hierarchical clustering
        # Convert to condensed distance matrix
        condensed_dist = squareform(distance_matrix, checks=False)

        # Perform hierarchical clustering
        linkage_matrix = linkage(condensed_dist, method='average')

        # Cut tree at threshold
        cluster_labels = fcluster(linkage_matrix, 1 - correlation_threshold, criterion='distance')

        # Group wallets by cluster
        clusters_dict = defaultdict(list)
        for wallet, label in zip(wallets, cluster_labels):
            clusters_dict[label].append(wallet)

        # Build WalletCluster objects
        clusters = []
        for label, cluster_wallets in clusters_dict.items():
            if len(cluster_wallets) < 2:
                continue

            # Calculate total volume and trades
            cluster_trades = df[df['wallet'].isin(cluster_wallets)]
            total_volume = cluster_trades['size'].sum()
            combined_trades = len(cluster_trades)

            # Find primary wallet
            wallet_vols = {w: self.wallet_volumes.get(w, 0) for w in cluster_wallets}
            primary_wallet = max(wallet_vols, key=wallet_vols.get) if wallet_vols else cluster_wallets[0]

            # Calculate confidence based on average correlation
            cluster_indices = [wallets.index(w) for w in cluster_wallets]
            cluster_corr = []
            for i in cluster_indices:
                for j in cluster_indices:
                    if i < j:
                        cluster_corr.append(abs(correlation_matrix[i, j]))

            avg_correlation = np.mean(cluster_corr) if cluster_corr else 0
            confidence = min(0.99, avg_correlation)

            cluster = WalletCluster(
                cluster_id=str(uuid.uuid4()),
                wallets=cluster_wallets,
                primary_wallet=primary_wallet,
                clustering_method="correlation",
                confidence=confidence,
                total_volume=float(total_volume),
                combined_trades=combined_trades
            )
            clusters.append(cluster)

        logger.info(f"Found {len(clusters)} correlation-based clusters")
        return clusters

    def merge_clusters(
        self,
        cluster_lists: List[List[WalletCluster]]
    ) -> List[WalletCluster]:
        """
        Merge clusters from different methods using Union-Find.

        If a wallet appears in multiple clusters from different methods,
        those clusters are merged together.
        """
        if not cluster_lists:
            return []

        # Flatten all clusters
        all_clusters = [cluster for sublist in cluster_lists for cluster in sublist]

        if not all_clusters:
            return []

        # Use Union-Find to merge clusters with overlapping wallets
        uf = UnionFind()

        # Map each wallet to all clusters it appears in
        wallet_to_clusters = defaultdict(list)
        for cluster in all_clusters:
            for wallet in cluster.wallets:
                wallet_to_clusters[wallet].append(cluster.cluster_id)

        # Merge clusters that share wallets
        for wallet, cluster_ids in wallet_to_clusters.items():
            if len(cluster_ids) > 1:
                # Merge all clusters containing this wallet
                for i in range(1, len(cluster_ids)):
                    uf.union(cluster_ids[0], cluster_ids[i])

        # Group clusters by their root
        cluster_groups = defaultdict(list)
        for cluster in all_clusters:
            root = uf.find(cluster.cluster_id)
            cluster_groups[root].append(cluster)

        # Build merged clusters
        merged_clusters = []
        for group in cluster_groups.values():
            # Collect all unique wallets
            all_wallets = set()
            total_volume = 0
            combined_trades = 0
            methods = set()
            confidences = []

            for cluster in group:
                all_wallets.update(cluster.wallets)
                total_volume += cluster.total_volume
                combined_trades += cluster.combined_trades
                methods.add(cluster.clustering_method)
                confidences.append(cluster.confidence)

            # Average confidence weighted by cluster size
            weights = [len(c.wallets) for c in group]
            avg_confidence = np.average(confidences, weights=weights)

            # Find primary wallet (highest volume)
            wallet_vols = {w: self.wallet_volumes.get(w, 0) for w in all_wallets}
            primary_wallet = max(wallet_vols, key=wallet_vols.get) if wallet_vols else list(all_wallets)[0]

            # Determine clustering method
            if len(methods) > 1:
                method = "merged"
            else:
                method = list(methods)[0]

            merged_cluster = WalletCluster(
                cluster_id=str(uuid.uuid4()),
                wallets=sorted(all_wallets),
                primary_wallet=primary_wallet,
                clustering_method=method,
                confidence=avg_confidence,
                total_volume=total_volume,
                combined_trades=combined_trades
            )
            merged_clusters.append(merged_cluster)

        logger.info(f"Merged {len(all_clusters)} clusters into {len(merged_clusters)} final clusters")
        return merged_clusters

    def run_full_clustering(
        self,
        methods: Optional[List[str]] = None
    ) -> ClusteringResult:
        """
        Run all clustering methods and merge results.

        Args:
            methods: List of methods to use. Options: "temporal", "behavior", "correlation"
                    Default: all methods

        Returns:
            ClusteringResult with all clusters and wallet-to-cluster mapping
        """
        if methods is None:
            methods = ["temporal", "behavior", "correlation"]

        cluster_lists = []

        if "temporal" in methods:
            logger.info("Running temporal clustering...")
            temporal_clusters = self.cluster_by_temporal_proximity()
            if temporal_clusters:
                cluster_lists.append(temporal_clusters)

        if "behavior" in methods:
            logger.info("Running behavioral clustering...")
            behavior_clusters = self.cluster_by_behavior()
            if behavior_clusters:
                cluster_lists.append(behavior_clusters)

        if "correlation" in methods:
            logger.info("Running correlation clustering...")
            correlation_clusters = self.cluster_by_position_correlation()
            if correlation_clusters:
                cluster_lists.append(correlation_clusters)

        # Merge all clusters
        logger.info("Merging clusters from different methods...")
        merged_clusters = self.merge_clusters(cluster_lists)

        # Build wallet-to-cluster mapping
        wallet_to_cluster = {}
        for cluster in merged_clusters:
            for wallet in cluster.wallets:
                wallet_to_cluster[wallet] = cluster.cluster_id

        # Find unclustered wallets
        all_wallets = set()
        if not self.trades.empty and 'wallet' in self.trades.columns:
            all_wallets = set(self.trades['wallet'].unique())
        elif self.behavior_profiles:
            all_wallets = set(self.behavior_profiles.keys())

        clustered_wallets = set(wallet_to_cluster.keys())
        unclustered_wallets = sorted(all_wallets - clustered_wallets)

        result = ClusteringResult(
            clusters=merged_clusters,
            wallet_to_cluster=wallet_to_cluster,
            unclustered_wallets=unclustered_wallets
        )

        logger.info(
            f"Clustering complete: {len(merged_clusters)} clusters, "
            f"{len(clustered_wallets)} clustered wallets, "
            f"{len(unclustered_wallets)} unclustered wallets"
        )

        return result

    def get_suspicious_clusters(
        self,
        result: ClusteringResult,
        min_wallets: int = 2,
        min_volume: float = 1000
    ) -> List[WalletCluster]:
        """
        Get clusters that are most suspicious for coordinated activity.

        Prioritize clusters with:
        - Multiple wallets
        - High total volume
        - High confidence
        """
        suspicious = []

        for cluster in result.clusters:
            # Filter by criteria
            if len(cluster.wallets) < min_wallets:
                continue

            if cluster.total_volume < min_volume:
                continue

            suspicious.append(cluster)

        # Sort by suspicion score (combination of confidence, volume, and size)
        def suspicion_score(cluster: WalletCluster) -> float:
            # Normalize components
            confidence_score = cluster.confidence
            volume_score = min(1.0, cluster.total_volume / 100000)  # Cap at 100k
            size_score = min(1.0, len(cluster.wallets) / 10)  # Cap at 10 wallets

            # Weighted combination
            return (
                0.4 * confidence_score +
                0.3 * volume_score +
                0.3 * size_score
            )

        suspicious.sort(key=suspicion_score, reverse=True)

        logger.info(f"Found {len(suspicious)} suspicious clusters")
        return suspicious
