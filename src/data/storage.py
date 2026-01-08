"""
Data storage utilities for Parquet file I/O.

Handles reading and writing Parquet files with:
- Consistent schemas
- Compression (snappy)
- Incremental updates
"""

import json
from pathlib import Path
from typing import List, Optional, Union
import logging

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class DataStorage:
    """
    Handles reading and writing Parquet files.

    Features:
    - Consistent schemas
    - Compression (snappy)
    - Incremental updates
    """

    # Schema definitions for validation
    MARKETS_SCHEMA = {
        "market_id": str,
        "question": str,
        "category": str,
        "volume": float,
        "liquidity": float,
        "volume_tier": str,
        "is_resolved": bool,
        "resolution": str,
    }

    TRADES_SCHEMA = {
        "trade_id": str,
        "market_id": str,
        "timestamp": "datetime64[ns, UTC]",
        "maker_address": str,
        "taker_address": str,
        "side": str,
        "outcome": str,
        "size": float,
        "price": float,
        "notional": float,
        "tx_hash": str,
    }

    WALLETS_SCHEMA = {
        "address": str,
        "first_seen": "datetime64[ns, UTC]",
        "last_seen": "datetime64[ns, UTC]",
        "total_trades": int,
        "total_volume": float,
        "unique_markets": int,
        "is_whale": bool,
        "volume_percentile": float,
    }

    def __init__(self, base_path: Union[str, Path] = "data/processed"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_path(self, filename: str) -> Path:
        """Get full path for a file."""
        return self.base_path / filename

    def _ensure_timezone(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Ensure datetime columns have timezone info."""
        for col in columns:
            if col in df.columns and pd.api.types.is_datetime64_any_dtype(df[col]):
                if df[col].dt.tz is None:
                    df[col] = df[col].dt.tz_localize("UTC")
                else:
                    df[col] = df[col].dt.tz_convert("UTC")
        return df

    def save_markets(self, df: pd.DataFrame, filename: str = "markets.parquet") -> Path:
        """
        Save markets data to Parquet.

        Args:
            df: DataFrame with market data
            filename: Output filename

        Returns:
            Path to saved file
        """
        if df.empty:
            logger.warning("Empty DataFrame, skipping save")
            return self._get_path(filename)

        # Ensure proper types
        df = df.copy()
        df = self._ensure_timezone(df, ["created_at", "end_date"])

        # Convert lists to JSON strings for Parquet compatibility
        if "outcome_prices" in df.columns:
            df["outcome_prices"] = df["outcome_prices"].apply(
                lambda x: json.dumps(x) if isinstance(x, list) else x
            )
        if "markets_list" in df.columns:
            df["markets_list"] = df["markets_list"].apply(
                lambda x: json.dumps(x) if isinstance(x, list) else x
            )

        path = self._get_path(filename)
        df.to_parquet(path, compression="snappy", index=False)
        logger.info(f"Saved {len(df)} markets to {path}")
        return path

    def save_trades(
        self,
        df: pd.DataFrame,
        filename: str = "trades.parquet",
        market_id: Optional[str] = None,
    ) -> Path:
        """
        Save trades data to Parquet.

        Args:
            df: DataFrame with trade data
            filename: Output filename (ignored if market_id provided)
            market_id: If provided, save to trades_{market_id}.parquet

        Returns:
            Path to saved file
        """
        if df.empty:
            logger.warning("Empty DataFrame, skipping save")
            return self._get_path(filename)

        # Ensure proper types
        df = df.copy()
        df = self._ensure_timezone(df, ["timestamp"])

        if market_id:
            # Sanitize market_id for filename
            safe_id = market_id[:16].replace("/", "_").replace("\\", "_")
            filename = f"trades_{safe_id}.parquet"

        path = self._get_path(filename)
        df.to_parquet(path, compression="snappy", index=False)
        logger.info(f"Saved {len(df)} trades to {path}")
        return path

    def save_wallets(self, df: pd.DataFrame, filename: str = "wallets.parquet") -> Path:
        """
        Save wallet index to Parquet.

        Args:
            df: DataFrame with wallet data
            filename: Output filename

        Returns:
            Path to saved file
        """
        if df.empty:
            logger.warning("Empty DataFrame, skipping save")
            return self._get_path(filename)

        # Ensure proper types
        df = df.copy()
        df = self._ensure_timezone(df, ["first_seen", "last_seen"])

        # Convert lists to JSON strings
        if "markets_list" in df.columns:
            df["markets_list"] = df["markets_list"].apply(
                lambda x: json.dumps(x) if isinstance(x, list) else x
            )

        path = self._get_path(filename)
        df.to_parquet(path, compression="snappy", index=False)
        logger.info(f"Saved {len(df)} wallets to {path}")
        return path

    def load_markets(self, filename: str = "markets.parquet") -> pd.DataFrame:
        """
        Load markets data from Parquet.

        Returns:
            DataFrame with market data
        """
        path = self._get_path(filename)
        if not path.exists():
            logger.warning(f"File not found: {path}")
            return pd.DataFrame()

        df = pd.read_parquet(path)

        # Parse JSON columns back to lists
        if "outcome_prices" in df.columns:
            df["outcome_prices"] = df["outcome_prices"].apply(
                lambda x: json.loads(x) if isinstance(x, str) else x
            )
        if "markets_list" in df.columns:
            df["markets_list"] = df["markets_list"].apply(
                lambda x: json.loads(x) if isinstance(x, str) else x
            )

        logger.info(f"Loaded {len(df)} markets from {path}")
        return df

    def load_trades(
        self,
        filename: str = "trades.parquet",
        market_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load trades data from Parquet.

        Args:
            filename: Filename to load (ignored if market_id provided)
            market_id: If provided, load trades_{market_id}.parquet

        Returns:
            DataFrame with trade data
        """
        if market_id:
            safe_id = market_id[:16].replace("/", "_").replace("\\", "_")
            filename = f"trades_{safe_id}.parquet"

        path = self._get_path(filename)
        if not path.exists():
            logger.warning(f"File not found: {path}")
            return pd.DataFrame()

        df = pd.read_parquet(path)
        logger.info(f"Loaded {len(df)} trades from {path}")
        return df

    def load_wallets(self, filename: str = "wallets.parquet") -> pd.DataFrame:
        """
        Load wallet index from Parquet.

        Returns:
            DataFrame with wallet data
        """
        path = self._get_path(filename)
        if not path.exists():
            logger.warning(f"File not found: {path}")
            return pd.DataFrame()

        df = pd.read_parquet(path)

        # Parse JSON columns back to lists
        if "markets_list" in df.columns:
            df["markets_list"] = df["markets_list"].apply(
                lambda x: json.loads(x) if isinstance(x, str) else x
            )

        logger.info(f"Loaded {len(df)} wallets from {path}")
        return df

    def load_all_trade_files(self) -> pd.DataFrame:
        """
        Load all trade files matching trades_*.parquet pattern.

        Returns:
            Combined DataFrame with all trades
        """
        trade_files = list(self.base_path.glob("trades_*.parquet"))

        if not trade_files:
            # Try loading single trades.parquet
            return self.load_trades()

        all_trades = []
        for path in trade_files:
            try:
                df = pd.read_parquet(path)
                all_trades.append(df)
            except Exception as e:
                logger.error(f"Error loading {path}: {e}")
                continue

        if not all_trades:
            return pd.DataFrame()

        combined = pd.concat(all_trades, ignore_index=True)
        logger.info(f"Loaded {len(combined)} trades from {len(trade_files)} files")
        return combined

    def file_exists(self, filename: str) -> bool:
        """Check if a file exists."""
        return self._get_path(filename).exists()

    def list_files(self, pattern: str = "*.parquet") -> List[Path]:
        """List files matching pattern."""
        return list(self.base_path.glob(pattern))

    def get_file_info(self, filename: str) -> Optional[dict]:
        """Get file metadata."""
        path = self._get_path(filename)
        if not path.exists():
            return None

        stat = path.stat()
        return {
            "path": str(path),
            "size_bytes": stat.st_size,
            "size_mb": stat.st_size / (1024 * 1024),
            "modified": stat.st_mtime,
        }
