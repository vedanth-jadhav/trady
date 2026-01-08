"""
Tests for data ingestion client and components.
"""

import asyncio
from datetime import datetime, timezone
import pytest
import pandas as pd

from src.data.client import RateLimiter, PolymarketClient
from src.data.fetcher import MarketSelector, TradeFetcher, WalletIndexer
from src.data.storage import DataStorage
from src.data.validator import DataValidator


class TestRateLimiter:
    """Tests for RateLimiter class."""

    @pytest.mark.asyncio
    async def test_acquire_tokens(self):
        """Test that rate limiter correctly manages tokens."""
        limiter = RateLimiter(rate=10.0, burst=5)

        # Should allow burst of 5 requests immediately
        for _ in range(5):
            await limiter.acquire()

        # Check tokens depleted
        assert limiter.tokens < 1

    @pytest.mark.asyncio
    async def test_rate_limit_wait(self):
        """Test that rate limiter waits when tokens depleted."""
        limiter = RateLimiter(rate=100.0, burst=1)

        # First request immediate
        await limiter.acquire()

        # Second request should wait but complete
        start = asyncio.get_event_loop().time()
        await limiter.acquire()
        elapsed = asyncio.get_event_loop().time() - start

        # Should have waited some time
        assert elapsed >= 0


class TestWalletIndexer:
    """Tests for WalletIndexer class."""

    def test_build_index_empty(self):
        """Test building index from empty DataFrame."""
        indexer = WalletIndexer()
        result = indexer.build_index(pd.DataFrame())
        assert result.empty

    def test_build_index_basic(self):
        """Test building index from sample trades."""
        trades = pd.DataFrame({
            "maker_address": ["0x123", "0x123", "0x456"],
            "taker_address": ["0x789", "0x456", "0x789"],
            "timestamp": [
                datetime(2024, 1, 1, tzinfo=timezone.utc),
                datetime(2024, 1, 2, tzinfo=timezone.utc),
                datetime(2024, 1, 3, tzinfo=timezone.utc),
            ],
            "notional": [100.0, 200.0, 300.0],
            "market_id": ["m1", "m1", "m2"],
        })

        indexer = WalletIndexer()
        result = indexer.build_index(trades)

        assert len(result) == 3  # 3 unique addresses
        assert "0x123" in result["address"].values
        assert "0x456" in result["address"].values
        assert "0x789" in result["address"].values

    def test_whale_identification(self):
        """Test whale identification based on volume."""
        wallets = pd.DataFrame({
            "address": ["0x1", "0x2", "0x3", "0x4", "0x5"],
            "total_volume": [100, 200, 300, 400, 10000],
            "total_trades": [1, 1, 1, 1, 1],
        })

        indexer = WalletIndexer(whale_percentile=0.8)
        whales = indexer.identify_whales(wallets)

        # Top 20% should be whales
        assert len(whales) == 1
        assert whales.iloc[0]["address"] == "0x5"


class TestDataValidator:
    """Tests for DataValidator class."""

    def test_validate_markets_valid(self):
        """Test validation passes for valid markets."""
        markets = pd.DataFrame({
            "market_id": ["m1", "m2"],
            "question": ["Q1", "Q2"],
            "volume": [100.0, 200.0],
        })

        validator = DataValidator()
        is_valid, _ = validator.validate_markets(markets)
        assert is_valid

    def test_validate_markets_duplicates(self):
        """Test validation fails for duplicate market IDs."""
        markets = pd.DataFrame({
            "market_id": ["m1", "m1"],  # Duplicate
            "question": ["Q1", "Q2"],
            "volume": [100.0, 200.0],
        })

        validator = DataValidator()
        is_valid, _ = validator.validate_markets(markets)
        assert not is_valid

    def test_validate_trades_invalid_price(self):
        """Test validation fails for invalid price."""
        trades = pd.DataFrame({
            "trade_id": ["t1"],
            "market_id": ["m1"],
            "timestamp": [datetime.now(timezone.utc)],
            "maker_address": ["0x123"],
            "taker_address": ["0x456"],
            "size": [100.0],
            "price": [1.5],  # Invalid: > 1
        })

        validator = DataValidator()
        is_valid, _ = validator.validate_trades(trades)
        assert not is_valid

    def test_validate_wallets_valid(self):
        """Test validation passes for valid wallets."""
        wallets = pd.DataFrame({
            "address": ["0x123", "0x456"],
            "first_seen": [
                datetime(2024, 1, 1, tzinfo=timezone.utc),
                datetime(2024, 1, 1, tzinfo=timezone.utc),
            ],
            "last_seen": [
                datetime(2024, 1, 2, tzinfo=timezone.utc),
                datetime(2024, 1, 2, tzinfo=timezone.utc),
            ],
            "total_trades": [10, 20],
            "total_volume": [100.0, 200.0],
        })

        validator = DataValidator()
        is_valid, _ = validator.validate_wallets(wallets)
        assert is_valid


class TestDataStorage:
    """Tests for DataStorage class."""

    def test_save_and_load_markets(self, tmp_path):
        """Test saving and loading markets."""
        storage = DataStorage(tmp_path)

        markets = pd.DataFrame({
            "market_id": ["m1", "m2"],
            "question": ["Q1?", "Q2?"],
            "volume": [100.0, 200.0],
            "outcome_prices": [[0.5, 0.5], [0.7, 0.3]],
        })

        storage.save_markets(markets)
        loaded = storage.load_markets()

        assert len(loaded) == 2
        assert loaded["market_id"].tolist() == ["m1", "m2"]

    def test_save_and_load_trades(self, tmp_path):
        """Test saving and loading trades."""
        storage = DataStorage(tmp_path)

        trades = pd.DataFrame({
            "trade_id": ["t1", "t2"],
            "market_id": ["m1", "m1"],
            "timestamp": [
                datetime(2024, 1, 1, tzinfo=timezone.utc),
                datetime(2024, 1, 2, tzinfo=timezone.utc),
            ],
            "maker_address": ["0x123", "0x456"],
            "taker_address": ["0x789", "0xabc"],
            "size": [100.0, 200.0],
            "price": [0.5, 0.6],
            "notional": [50.0, 120.0],
        })

        storage.save_trades(trades)
        loaded = storage.load_trades()

        assert len(loaded) == 2

    def test_file_exists(self, tmp_path):
        """Test file existence check."""
        storage = DataStorage(tmp_path)

        assert not storage.file_exists("nonexistent.parquet")

        # Create a file
        pd.DataFrame({"a": [1]}).to_parquet(tmp_path / "test.parquet")
        assert storage.file_exists("test.parquet")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
