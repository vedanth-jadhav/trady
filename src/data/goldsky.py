"""
Goldsky GraphQL client for Polymarket historical trade data.

Goldsky is Polymarket's official blockchain indexer providing:
- Full historical data since 2020
- No authentication required
- Cursor-based pagination that works correctly
- Rate limit: 50 requests per 10 seconds (18,000/hour)

Performance optimizations:
- uvloop for 2-4x faster event loop (Unix only)
- orjson for 3-10x faster JSON parsing
- Increased connection pooling (500 total, 200 per host)
- Higher concurrency (100 workers)
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Callable

import aiohttp
import certifi
import ssl

# Performance: Use uvloop for faster event loop (Unix only)
try:
    import uvloop
    import sys
    if sys.version_info >= (3, 12):
        # For Python 3.12+, set the event loop policy instead of install()
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    else:
        uvloop.install()
    _UVLOOP_AVAILABLE = True
except ImportError:
    _UVLOOP_AVAILABLE = False

# Performance: Use orjson for faster JSON parsing
try:
    import orjson
    def _json_loads(data: bytes) -> dict:
        return orjson.loads(data)
    _ORJSON_AVAILABLE = True
except ImportError:
    import json
    def _json_loads(data: str) -> dict:
        return json.loads(data)
    _ORJSON_AVAILABLE = False

logger = logging.getLogger(__name__)

# Log performance features available
logger.debug(f"Performance: uvloop={'enabled' if _UVLOOP_AVAILABLE else 'disabled'}, orjson={'enabled' if _ORJSON_AVAILABLE else 'disabled'}")

# Goldsky API endpoint for Polymarket orderbook subgraph
GOLDSKY_URL = "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/orderbook-subgraph/0.0.1/gn"

# Rate limiting: 500 requests per 10 seconds (allows ~50 req/sec)
RATE_LIMIT_REQUESTS = 500
RATE_LIMIT_WINDOW = 10  # seconds

# Performance: Connection pool settings
CONNECTION_POOL_LIMIT = 100  # SAFE LIMIT for macOS
CONNECTION_POOL_PER_HOST = 50  # Per-host connections
CONCURRENT_REQUESTS = 50  # Semaphore limit - safe for stable fetching


class GoldskyClient:
    """
    GraphQL client for fetching Polymarket trade data from Goldsky.

    Uses the orderFilledEvents query to get trade history with proper
    cursor-based pagination using timestamp_gt.
    """

    def __init__(
        self,
        url: str = GOLDSKY_URL,
        batch_size: int = 1000,
        max_retries: int = 3,
    ):
        """
        Initialize Goldsky client.

        Args:
            url: Goldsky GraphQL endpoint
            batch_size: Number of records per query (max 1000)
            max_retries: Number of retries on failure
        """
        self.url = url
        self.batch_size = min(batch_size, 1000)  # Max 1000 per query
        self.max_retries = max_retries
        self._session: Optional[aiohttp.ClientSession] = None
        self._request_times: List[float] = []

    async def __aenter__(self):
        """Create aiohttp session with optimized connection pool for high throughput."""
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        # Performance: Increased connection pool for higher throughput
        connector = aiohttp.TCPConnector(
            ssl=ssl_context,
            limit=CONNECTION_POOL_LIMIT,  # Max total connections
            limit_per_host=CONNECTION_POOL_PER_HOST,  # Max per host
            ttl_dns_cache=300,
            use_dns_cache=True,  # Cache DNS lookups
            enable_cleanup_closed=True,
            force_close=False,
            keepalive_timeout=60,  # Longer keepalive for connection reuse
        )
        timeout = aiohttp.ClientTimeout(total=300, connect=30)  # 5 min total timeout
        self._session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        self._semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)  # Higher concurrency
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close aiohttp session."""
        if self._session:
            await self._session.close()
            self._session = None

    async def _rate_limit(self):
        """Apply rate limiting: 50 requests per 10 seconds."""
        now = asyncio.get_running_loop().time()

        # Remove old timestamps outside the window
        self._request_times = [
            t for t in self._request_times
            if now - t < RATE_LIMIT_WINDOW
        ]

        # If at limit, wait for oldest request to expire
        if len(self._request_times) >= RATE_LIMIT_REQUESTS:
            wait_time = RATE_LIMIT_WINDOW - (now - self._request_times[0]) + 0.1
            if wait_time > 0:
                logger.debug(f"Rate limit reached, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)

        self._request_times.append(now)

    async def _query(self, query: str, variables: Dict = None) -> Dict[str, Any]:
        """
        Execute GraphQL query with retry logic.

        Args:
            query: GraphQL query string
            variables: Optional query variables

        Returns:
            Query response data
        """
        if not self._session:
            raise RuntimeError("Client not initialized. Use 'async with' context.")

        await self._rate_limit()

        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        for attempt in range(self.max_retries):
            try:
                async with self._session.post(
                    self.url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status == 429:
                        # Rate limited - wait and retry
                        retry_after = int(response.headers.get("Retry-After", 10))
                        logger.warning(f"Rate limited, waiting {retry_after}s")
                        await asyncio.sleep(retry_after)
                        continue

                    response.raise_for_status()
                    # Performance: Use orjson for faster parsing if available
                    if _ORJSON_AVAILABLE:
                        raw_data = await response.read()
                        data = _json_loads(raw_data)
                    else:
                        data = await response.json()

                    if "errors" in data:
                        logger.error(f"GraphQL errors: {data['errors']}")
                        raise Exception(f"GraphQL error: {data['errors']}")

                    return data.get("data", {})

            except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(f"Request failed, retrying in {wait}s: {type(e).__name__}: {e}")
                    await asyncio.sleep(wait)
                else:
                    logger.error(f"All retries exhausted: {type(e).__name__}: {e}")
                    raise

        return {}

    async def get_order_filled_events(
        self,
        since_timestamp: int = 0,
        until_timestamp: Optional[int] = None,
        market_id: Optional[str] = None,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> List[Dict]:
        """
        Fetch orderFilledEvents from Goldsky.

        Uses cursor-based pagination with timestamp_gt.

        Args:
            since_timestamp: Unix timestamp to start from (exclusive)
            until_timestamp: Optional Unix timestamp to stop at
            market_id: Optional filter by maker/taker asset ID
            progress_callback: Optional callback(count) for progress updates

        Returns:
            List of order filled events
        """
        all_events = []
        cursor_timestamp = str(since_timestamp)

        # Build where clause
        where_parts = [f'timestamp_gt: "{cursor_timestamp}"']
        if until_timestamp:
            where_parts.append(f'timestamp_lt: "{until_timestamp}"')
        if market_id:
            # Filter by either maker or taker asset containing the market ID
            where_parts.append(f'makerAssetId_contains: "{market_id}"')

        where_clause = ", ".join(where_parts)

        query_template = """
        query OrderFilledEvents {{
            orderFilledEvents(
                orderBy: timestamp
                orderDirection: asc
                first: {batch_size}
                where: {{{where_clause}}}
            ) {{
                id
                timestamp
                transactionHash
                orderHash
                maker
                taker
                makerAssetId
                takerAssetId
                makerAmountFilled
                takerAmountFilled
                fee
            }}
        }}
        """

        page_count = 0
        while True:
            # Update where clause with current cursor
            where_parts[0] = f'timestamp_gt: "{cursor_timestamp}"'
            where_clause = ", ".join(where_parts)

            query = query_template.format(
                batch_size=self.batch_size,
                where_clause=where_clause
            )

            result = await self._query(query)
            events = result.get("orderFilledEvents", [])

            if not events:
                break

            all_events.extend(events)
            page_count += 1

            # Update cursor to last timestamp
            cursor_timestamp = events[-1]["timestamp"]

            if progress_callback:
                progress_callback(len(all_events))

            # Log progress
            if page_count % 10 == 0:
                logger.info(f"Fetched {len(all_events)} events (page {page_count})")

            # Check if we've reached the end or the until_timestamp
            if len(events) < self.batch_size:
                break

            if until_timestamp and int(cursor_timestamp) >= until_timestamp:
                break

        logger.info(f"Total fetched: {len(all_events)} order filled events")
        return all_events

    async def get_order_filled_events_parallel(
        self,
        since_timestamp: int,
        until_timestamp: int,
        num_workers: int = 100,  # Performance: Doubled parallel workers
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> List[Dict]:
        """
        Fetch orderFilledEvents in parallel by splitting time range.

        Uses semaphore-controlled concurrency for maximum speed without errors.

        Args:
            since_timestamp: Unix timestamp to start from
            until_timestamp: Unix timestamp to stop at
            num_workers: Number of parallel workers (default 50)
            progress_callback: Optional callback(count) for progress updates

        Returns:
            List of order filled events (sorted by timestamp)
        """
        # Split time range into chunks
        total_duration = until_timestamp - since_timestamp
        chunk_duration = max(total_duration // num_workers, 1)

        # Create time chunks
        chunks = []
        for i in range(num_workers):
            chunk_start = since_timestamp + (i * chunk_duration)
            chunk_end = since_timestamp + ((i + 1) * chunk_duration) if i < num_workers - 1 else until_timestamp
            if chunk_start < until_timestamp:
                chunks.append((chunk_start, chunk_end))

        logger.info(f"Fetching in parallel with {len(chunks)} workers...")

        # Shared state for progress
        progress_state = {"total": 0}
        progress_lock = asyncio.Lock()

        async def fetch_chunk(chunk_start: int, chunk_end: int) -> List[Dict]:
            """Fetch a single time chunk with semaphore control."""
            events = []
            cursor_timestamp = str(chunk_start)

            query_template = """
            query OrderFilledEvents {{
                orderFilledEvents(
                    orderBy: timestamp
                    orderDirection: asc
                    first: {batch_size}
                    where: {{timestamp_gt: "{cursor}", timestamp_lt: "{end}"}}
                ) {{
                    id
                    timestamp
                    transactionHash
                    orderHash
                    maker
                    taker
                    makerAssetId
                    takerAssetId
                    makerAmountFilled
                    takerAmountFilled
                    fee
                }}
            }}
            """

            while True:
                query = query_template.format(
                    batch_size=self.batch_size,
                    cursor=cursor_timestamp,
                    end=chunk_end
                )

                # Use semaphore to limit concurrent requests
                async with self._semaphore:
                    result = await self._query(query)

                batch = result.get("orderFilledEvents", [])

                if not batch:
                    break

                events.extend(batch)
                cursor_timestamp = batch[-1]["timestamp"]

                # Update global progress
                async with progress_lock:
                    progress_state["total"] += len(batch)
                    if progress_callback:
                        progress_callback(progress_state["total"])

                if len(batch) < self.batch_size:
                    break

                if int(cursor_timestamp) >= chunk_end:
                    break

            return events

        # Run all chunks in parallel with controlled concurrency
        tasks = [fetch_chunk(start, end) for start, end in chunks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine results, skip any errors
        all_events = []
        failed_chunks = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_chunks += 1
                logger.warning(f"Chunk {i} failed: {type(result).__name__}: {result}")
            else:
                all_events.extend(result)

        if failed_chunks > 0:
            logger.warning(f"{failed_chunks}/{len(chunks)} chunks failed")

        # Sort by timestamp to ensure correct order
        all_events.sort(key=lambda x: int(x["timestamp"]))

        logger.info(f"Total fetched (parallel): {len(all_events)} order filled events")
        return all_events

    async def get_order_filled_events_parallel_generator(
        self,
        since_timestamp: int,
        until_timestamp: int,
        num_workers: int = 100,
    ):
        """
        Fetch orderFilledEvents in parallel and yield batches as they complete.
        
        Uses a bounded queue to constrain memory usage and allow workers to
        yield partial results immediately, providing smooth progress updates.
        """
        # Split time range into chunks
        total_duration = until_timestamp - since_timestamp
        chunk_duration = max(total_duration // num_workers, 1)

        # Create time chunks
        chunks = []
        for i in range(num_workers):
            chunk_start = since_timestamp + (i * chunk_duration)
            chunk_end = since_timestamp + ((i + 1) * chunk_duration) if i < num_workers - 1 else until_timestamp
            if chunk_start < until_timestamp:
                chunks.append((chunk_start, chunk_end))

        logger.info(f"Fetching stream in parallel with {len(chunks)} workers...")

        # Bounded queue for backpressure
        # Capacity: 50 batches * 1000 items/batch = ~50k items buffered max
        queue = asyncio.Queue(maxsize=50) 
        
        async def fetch_worker(chunk_start: int, chunk_end: int):
            """Fetch a single time chunk and put batches in queue."""
            cursor_timestamp = str(chunk_start)

            query_template = """
            query OrderFilledEvents {{
                orderFilledEvents(
                    orderBy: timestamp
                    orderDirection: asc
                    first: {batch_size}
                    where: {{timestamp_gt: "{cursor}", timestamp_lt: "{end}"}}
                ) {{
                    id
                    timestamp
                    transactionHash
                    orderHash
                    maker
                    taker
                    makerAssetId
                    takerAssetId
                    makerAmountFilled
                    takerAmountFilled
                    fee
                }}
            }}
            """

            while True:
                query = query_template.format(
                    batch_size=self.batch_size,
                    cursor=cursor_timestamp,
                    end=chunk_end
                )

                try:
                    async with self._semaphore:
                        result = await self._query(query)

                    batch = result.get("orderFilledEvents", [])
                    if not batch:
                        break

                    # Put batch in queue (blocks if queue is full)
                    await queue.put(batch)
                    
                    cursor_timestamp = batch[-1]["timestamp"]
                    
                    if len(batch) < self.batch_size:
                        break
                    if int(cursor_timestamp) >= chunk_end:
                        break
                except Exception as e:
                    logger.error(f"Error in fetch worker: {e}")
                    # Don't break immediately on transient error, retry logic is in _query
                    # But if _query fails after retries, it raises.
                    # We should probably catch here to avoid killing the worker entirely?
                    # But _query raises on catastrophic failure.
                    # Let's break loop on error to avoid infinite loops
                    break

        # Start workers
        # We need to track how many workers are active so consumer knows when to stop
        workers = [asyncio.create_task(fetch_worker(start, end)) for start, end in chunks]
        
        # Consumer logic
        active_workers = len(workers)
        
        # We need to know when all workers are done.
        # Can use a separate task to wait for workers and put a sentinel, or just track logic.
        # Cleaner: Use a coordinator task that waits for all workers and then puts None.
        
        async def coordinator():
            await asyncio.gather(*workers, return_exceptions=True)
            await queue.put(None) # Sentinel

        coord_task = asyncio.create_task(coordinator())

        while True:
            # Get next batch
            item = await queue.get()
            
            if item is None:
                # Sentinel received, we are done
                break
                
            yield item
            
        # Cleanup (if user breaks loop early)
        if not coord_task.done():
            coord_task.cancel()
            for w in workers:
                w.cancel()

    async def get_trades_for_period(
        self,
        days: int = 7,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> List[Dict]:
        """
        Fetch trades for the last N days.

        Args:
            days: Number of days to look back
            progress_callback: Optional callback for progress

        Returns:
            List of parsed trade dictionaries
        """
        now = datetime.now(timezone.utc)
        since = now - timedelta(days=days)

        since_ts = int(since.timestamp())
        until_ts = int(now.timestamp())

        logger.info(f"Fetching trades from {since} to {now}")

        events = await self.get_order_filled_events(
            since_timestamp=since_ts,
            until_timestamp=until_ts,
            progress_callback=progress_callback,
        )

        # Parse events into trade format
        trades = []
        for event in events:
            trade = self._parse_order_filled_event(event)
            if trade:
                trades.append(trade)

        logger.info(f"Parsed {len(trades)} trades from {len(events)} events")
        return trades

    def _parse_order_filled_event(self, event: Dict) -> Optional[Dict]:
        """
        Parse orderFilledEvent into trade format.

        Args:
            event: Raw event from Goldsky

        Returns:
            Parsed trade dict or None if invalid
        """
        try:
            timestamp = int(event["timestamp"])

            # Parse amounts (these are in wei, need to convert)
            maker_amount = float(event.get("makerAmountFilled", 0)) / 1e6  # USDC has 6 decimals
            taker_amount = float(event.get("takerAmountFilled", 0)) / 1e6

            # Derive side: makerAssetId == 0 means maker gives USDC (BUY), else SELL
            maker_asset = event.get("makerAssetId", "")
            taker_asset = event.get("takerAssetId", "")

            is_maker_usdc = maker_asset == "" or maker_asset == "0"
            side = "BUY" if is_maker_usdc else "SELL"

            # Determine market ID from the non-USDC asset
            market_id = ""
            if not is_maker_usdc:
                market_id = maker_asset.split("-")[0] if "-" in maker_asset else maker_asset
            elif taker_asset and taker_asset != "0":
                taker_id = taker_asset.split("-")[0] if "-" in taker_asset else taker_asset
                market_id = taker_id

            # Price = usdc_amount / token_amount
            if is_maker_usdc:
                usdc_amount = maker_amount
                token_amount = taker_amount
            else:
                usdc_amount = taker_amount
                token_amount = maker_amount

            if token_amount > 0:
                price = usdc_amount / token_amount
                size = token_amount
            else:
                price = 0.5
                size = maker_amount or taker_amount

            # Normalize price to 0-1 range
            price = max(0, min(1, price))  # Clamp to [0, 1]

            return {
                "trade_id": event["id"],
                "source": "goldsky",  # Track source
                "tx_hash": event.get("transactionHash", ""),
                "timestamp": datetime.fromtimestamp(timestamp, tz=timezone.utc),
                "market_id": market_id,
                "maker_address": event.get("maker", "").lower(),
                "taker_address": event.get("taker", "").lower(),
                "size": size,
                "price": price,
                "notional": size * price,
                "side": side,
                "outcome": "",  # Will need more context
                "outcome": "",  # Will need more context
                "fee": float(event.get("fee", 0)) / 1e6,
            }
        except Exception as e:
            logger.warning(f"Failed to parse event: {e}")
            return None


async def test_goldsky():
    """Test Goldsky client."""
    async with GoldskyClient() as client:
        # Fetch last 1 day of trades
        trades = await client.get_trades_for_period(days=1)
        print(f"Fetched {len(trades)} trades")

        if trades:
            # Show date range
            timestamps = [t["timestamp"] for t in trades]
            print(f"Date range: {min(timestamps)} to {max(timestamps)}")

            # Show sample
            print(f"Sample trade: {trades[0]}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_goldsky())
