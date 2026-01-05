# Phase 1: Data Ingestion

## Objective

Build a robust data ingestion pipeline that scrapes market and trade data from Polymarket's API, processes it into analysis-ready formats, and stores it efficiently in Parquet files.

---

## Scope

| Item | Details |
|------|---------|
| Markets | ~100 markets (33 low, 34 medium, 33 high volume) |
| Timeframe | 90 days historical |
| Data Types | Markets, Trades, Wallet activity |
| Storage | Parquet files (columnar, efficient) |
| Rate Limiting | Respectful API usage with backoff |

---

## API Endpoints

### 1. Markets Endpoint

```
GET https://clob.polymarket.com/markets
```

**Response Fields:**
```python
{
    "id": str,                    # Market ID (condition_id)
    "question": str,              # Market question text
    "description": str,           # Detailed description
    "outcomes": List[str],        # ["Yes", "No"] typically
    "outcome_prices": List[str],  # Current prices as strings
    "volume": str,                # Total volume traded
    "liquidity": str,             # Current liquidity
    "end_date_iso": str,          # Resolution date
    "active": bool,               # Is market active
    "closed": bool,               # Is market closed
    "resolved": bool,             # Is market resolved
    "resolution": str,            # Resolution outcome if resolved
    "created_at": str,            # Creation timestamp
    "category": str,              # Market category
}
```

### 2. Trades Endpoint

```
GET https://clob.polymarket.com/trades?market={market_id}&limit=500&offset=0
```

**Response Fields:**
```python
{
    "id": str,                    # Trade ID
    "market": str,                # Market ID
    "asset_id": str,              # Token ID (Yes or No token)
    "side": str,                  # "BUY" or "SELL"
    "size": str,                  # Number of shares
    "price": str,                 # Price per share (0-1)
    "timestamp": int,             # Unix timestamp (ms)
    "maker_address": str,         # Maker wallet address
    "taker_address": str,         # Taker wallet address
    "transaction_hash": str,      # On-chain tx hash
    "outcome": str,               # "Yes" or "No"
}
```

### 3. Trades by User

```
GET https://clob.polymarket.com/trades?user={wallet_address}&limit=500&offset=0
```

Same response format as above, filtered by wallet address.

---

## Data Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      DATA INGESTION PIPELINE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐                                               │
│  │   Market     │                                               │
│  │   Selector   │  Select ~100 markets by volume tier           │
│  └──────┬───────┘                                               │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────┐    ┌──────────────┐                           │
│  │   Market     │───▶│   Market     │  markets.parquet          │
│  │   Fetcher    │    │   Storage    │                           │
│  └──────┬───────┘    └──────────────┘                           │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────┐    ┌──────────────┐                           │
│  │   Trade      │───▶│   Trade      │  trades_{market}.parquet  │
│  │   Fetcher    │    │   Storage    │                           │
│  └──────┬───────┘    └──────────────┘                           │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────┐    ┌──────────────┐                           │
│  │   Wallet     │───▶│   Wallet     │  wallets.parquet          │
│  │   Extractor  │    │   Index      │                           │
│  └──────────────┘    └──────────────┘                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Module Specifications

### 1. MarketSelector

**Purpose**: Select a diverse set of ~100 markets across volume tiers.

```python
class MarketSelector:
    """
    Selects markets for analysis based on volume distribution.

    Volume Tiers:
    - Low: Bottom 33% by volume
    - Medium: Middle 34% by volume
    - High: Top 33% by volume

    Selection Criteria:
    - Active markets only (not closed/resolved for recent data)
    - Include some resolved markets (for backtest ground truth)
    - Exclude near-100% probability markets (no signal value)
    - Prioritize high-insider categories (political, corporate)
    """

    def __init__(self, api_client: PolymarketClient):
        self.client = api_client

    def fetch_all_markets(self) -> pd.DataFrame:
        """Fetch all markets from API."""
        pass

    def categorize_by_volume(self, markets: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Split markets into low/medium/high volume tiers."""
        pass

    def select_markets(
        self,
        n_total: int = 100,
        include_resolved: bool = True,
        min_trades: int = 50,
        exclude_settled: bool = True,  # >95% probability
        priority_categories: List[str] = ["Politics", "Business"]
    ) -> List[str]:
        """
        Select market IDs for analysis.

        Returns:
            List of market IDs (condition_ids)
        """
        pass
```

**Output Schema** (`selected_markets.parquet`):
```python
{
    "market_id": str,
    "question": str,
    "category": str,
    "volume_tier": str,        # "low", "medium", "high"
    "total_volume": float,
    "trade_count": int,
    "created_at": datetime,
    "end_date": datetime,
    "is_resolved": bool,
    "resolution": Optional[str],
}
```

---

### 2. PolymarketClient

**Purpose**: Handle all API communication with rate limiting and error handling.

```python
class PolymarketClient:
    """
    API client for Polymarket CLOB API.

    Features:
    - Automatic rate limiting (respect API limits)
    - Exponential backoff on errors
    - Pagination handling
    - Response caching (optional)
    """

    BASE_URL = "https://clob.polymarket.com"

    def __init__(
        self,
        requests_per_second: float = 2.0,
        max_retries: int = 3,
        cache_enabled: bool = True
    ):
        self.rate_limiter = RateLimiter(requests_per_second)
        self.max_retries = max_retries
        self.session = requests.Session()
        self.cache = {} if cache_enabled else None

    async def get_markets(self, next_cursor: str = None) -> Dict:
        """Fetch markets with pagination."""
        pass

    async def get_all_markets(self) -> List[Dict]:
        """Fetch all markets, handling pagination."""
        pass

    async def get_trades(
        self,
        market_id: str = None,
        user_address: str = None,
        limit: int = 500,
        offset: int = 0
    ) -> List[Dict]:
        """Fetch trades for a market or user."""
        pass

    async def get_all_trades_for_market(self, market_id: str) -> List[Dict]:
        """Fetch all trades for a market, handling pagination."""
        pass

    async def get_all_trades_for_user(self, user_address: str) -> List[Dict]:
        """Fetch all trades for a user, handling pagination."""
        pass
```

**Rate Limiting Strategy**:
```python
class RateLimiter:
    """
    Token bucket rate limiter.

    - 2 requests per second baseline
    - Burst allowance of 10 requests
    - Exponential backoff on 429 errors
    """

    def __init__(self, rate: float = 2.0, burst: int = 10):
        self.rate = rate
        self.burst = burst
        self.tokens = burst
        self.last_update = time.time()

    async def acquire(self):
        """Wait until a request can be made."""
        pass

    def handle_rate_limit_error(self, retry_after: int = None):
        """Handle 429 response with backoff."""
        pass
```

---

### 3. TradeFetcher

**Purpose**: Fetch and process all trades for selected markets.

```python
class TradeFetcher:
    """
    Fetches all trades for selected markets.

    Process:
    1. For each market, paginate through all trades
    2. Filter trades within 90-day window
    3. Extract unique wallet addresses
    4. Store in Parquet format
    """

    def __init__(
        self,
        client: PolymarketClient,
        lookback_days: int = 90
    ):
        self.client = client
        self.lookback_days = lookback_days
        self.cutoff_date = datetime.now() - timedelta(days=lookback_days)

    async def fetch_trades_for_market(self, market_id: str) -> pd.DataFrame:
        """
        Fetch all trades for a single market.

        Returns DataFrame with columns:
        - trade_id, market_id, timestamp,
        - maker_address, taker_address,
        - side, outcome, size, price,
        - tx_hash
        """
        pass

    async def fetch_all_trades(
        self,
        market_ids: List[str],
        progress_callback: Callable = None
    ) -> pd.DataFrame:
        """
        Fetch trades for all markets concurrently.

        Uses asyncio.gather with semaphore to limit concurrency.
        """
        pass

    def extract_unique_wallets(self, trades: pd.DataFrame) -> Set[str]:
        """Extract all unique wallet addresses from trades."""
        pass
```

**Output Schema** (`trades.parquet`):
```python
{
    "trade_id": str,
    "market_id": str,
    "timestamp": datetime,
    "maker_address": str,
    "taker_address": str,
    "side": str,              # "BUY" or "SELL"
    "outcome": str,           # "Yes" or "No"
    "size": float,            # Number of shares
    "price": float,           # Price 0-1
    "notional": float,        # size * price (USD value)
    "tx_hash": str,
}
```

---

### 4. WalletIndexer

**Purpose**: Build an index of all wallets seen in the data.

```python
class WalletIndexer:
    """
    Creates an index of all wallets with summary statistics.

    For each wallet, compute:
    - First seen date
    - Last seen date
    - Total trade count
    - Total volume traded
    - Markets participated in
    - Unique markets count
    """

    def __init__(self):
        pass

    def build_index(self, trades: pd.DataFrame) -> pd.DataFrame:
        """
        Build wallet index from trades data.

        Aggregates both maker and taker sides.
        """
        pass

    def identify_whales(
        self,
        wallet_index: pd.DataFrame,
        percentile: float = 0.95
    ) -> pd.DataFrame:
        """
        Identify whale wallets by volume.

        Dynamic threshold based on percentile of volume distribution.
        """
        pass
```

**Output Schema** (`wallets.parquet`):
```python
{
    "address": str,
    "first_seen": datetime,
    "last_seen": datetime,
    "total_trades": int,
    "total_volume": float,
    "unique_markets": int,
    "markets_list": List[str],
    "is_whale": bool,
    "volume_percentile": float,
}
```

---

### 5. DataStorage

**Purpose**: Handle Parquet file I/O with proper schemas.

```python
class DataStorage:
    """
    Handles reading and writing Parquet files.

    Features:
    - Consistent schemas
    - Compression (snappy)
    - Partitioning by date (optional)
    - Incremental updates
    """

    def __init__(self, base_path: Path = Path("data/processed")):
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save_markets(self, df: pd.DataFrame):
        """Save markets data."""
        pass

    def save_trades(self, df: pd.DataFrame, market_id: str = None):
        """
        Save trades data.

        If market_id provided, save to trades_{market_id}.parquet
        Otherwise, save all to trades.parquet
        """
        pass

    def save_wallets(self, df: pd.DataFrame):
        """Save wallet index."""
        pass

    def load_markets(self) -> pd.DataFrame:
        """Load markets data."""
        pass

    def load_trades(self, market_id: str = None) -> pd.DataFrame:
        """Load trades data."""
        pass

    def load_wallets(self) -> pd.DataFrame:
        """Load wallet index."""
        pass
```

---

## Data Quality Checks

```python
class DataValidator:
    """
    Validates ingested data for quality issues.
    """

    def validate_markets(self, df: pd.DataFrame) -> List[str]:
        """
        Check:
        - No duplicate market IDs
        - All required fields present
        - Volume >= 0
        - Dates are valid
        """
        pass

    def validate_trades(self, df: pd.DataFrame) -> List[str]:
        """
        Check:
        - No duplicate trade IDs
        - Price in range [0, 1]
        - Size > 0
        - Valid wallet addresses (0x...)
        - Timestamps within expected range
        """
        pass

    def validate_wallets(self, df: pd.DataFrame) -> List[str]:
        """
        Check:
        - No duplicate addresses
        - first_seen <= last_seen
        - total_trades > 0
        """
        pass
```

---

## CLI Interface

```python
# src/data/cli.py

import typer

app = typer.Typer()

@app.command()
def fetch_markets(
    output_dir: Path = Path("data/processed"),
    min_volume: float = 0,
    categories: List[str] = None
):
    """Fetch and save all markets."""
    pass

@app.command()
def select_markets(
    n_markets: int = 100,
    output_dir: Path = Path("data/processed")
):
    """Select markets for analysis."""
    pass

@app.command()
def fetch_trades(
    market_file: Path = Path("data/processed/selected_markets.parquet"),
    output_dir: Path = Path("data/processed"),
    lookback_days: int = 90
):
    """Fetch trades for selected markets."""
    pass

@app.command()
def build_wallet_index(
    trades_file: Path = Path("data/processed/trades.parquet"),
    output_dir: Path = Path("data/processed")
):
    """Build wallet index from trades."""
    pass

@app.command()
def run_full_pipeline(
    n_markets: int = 100,
    lookback_days: int = 90,
    output_dir: Path = Path("data/processed")
):
    """Run complete data ingestion pipeline."""
    pass

if __name__ == "__main__":
    app()
```

---

## File Outputs

After Phase 1 completion:

```
data/
├── raw/
│   └── api_responses/         # Optional: raw JSON for debugging
├── processed/
│   ├── markets.parquet        # All fetched markets
│   ├── selected_markets.parquet  # 100 selected markets
│   ├── trades.parquet         # All trades for selected markets
│   └── wallets.parquet        # Wallet index
```

---

## Dependencies

```python
# requirements.txt additions for Phase 1

aiohttp>=3.9.0          # Async HTTP client
pandas>=2.0.0           # Data manipulation
pyarrow>=14.0.0         # Parquet support
typer>=0.9.0            # CLI framework
tqdm>=4.66.0            # Progress bars
tenacity>=8.2.0         # Retry logic
python-dateutil>=2.8.0  # Date parsing
```

---

## Success Criteria

- [ ] Successfully fetch metadata for all active Polymarket markets
- [ ] Select 100 markets with proper volume tier distribution
- [ ] Fetch all trades for selected markets (90 days)
- [ ] Build wallet index with summary statistics
- [ ] All data passes validation checks
- [ ] Parquet files are properly compressed and readable
- [ ] CLI commands work as documented

---

## Estimated Data Volumes

| Data Type | Estimated Rows | Estimated Size |
|-----------|----------------|----------------|
| Markets | ~500-1000 | ~1 MB |
| Selected Markets | 100 | ~100 KB |
| Trades | 100K - 1M | 50-500 MB |
| Wallets | 10K - 100K | 5-50 MB |

---

## Next Phase

After completing Phase 1, proceed to:
→ **Phase 2: Wallet Analysis** (`02_wallet_analysis/spec.md`)

The wallet index created here will be enriched with:
- Wallet age and history analysis
- Funding source tracking
- Behavioral clustering
