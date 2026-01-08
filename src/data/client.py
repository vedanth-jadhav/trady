"""
Polymarket API Client with rate limiting and error handling.

Handles all communication with the Polymarket CLOB API including:
- Automatic rate limiting (token bucket algorithm)
- Exponential backoff on errors
- Pagination handling
- Response caching (optional)
"""

import asyncio
import socket
import ssl
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging

import aiohttp
import certifi
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    """Raised when rate limit is exceeded."""

    def __init__(self, retry_after: Optional[int] = None):
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded. Retry after {retry_after}s")


class APIError(Exception):
    """General API error."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"API Error {status_code}: {message}")


@dataclass
class RateLimiter:
    """
    Token bucket rate limiter for API requests.

    - Baseline: 2 requests per second
    - Burst allowance: 10 requests
    - Exponential backoff on 429 errors
    """

    rate: float = 2.0  # Requests per second
    burst: int = 10  # Maximum burst capacity
    tokens: float = field(init=False)
    last_update: float = field(init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def __post_init__(self):
        self.tokens = float(self.burst)
        self.last_update = time.monotonic()

    async def acquire(self) -> None:
        """Wait until a request can be made."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens < 1:
                wait_time = (1 - self.tokens) / self.rate
                logger.debug(f"Rate limit: waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1

    async def handle_rate_limit_error(self, retry_after: Optional[int] = None) -> None:
        """Handle 429 response with backoff."""
        wait_time = retry_after if retry_after else 5
        logger.warning(f"Rate limited. Waiting {wait_time}s before retry.")
        async with self._lock:
            self.tokens = 0  # Reset tokens
        await asyncio.sleep(wait_time)


class PolymarketClient:
    """
    API client for Polymarket CLOB API.

    Features:
    - Automatic rate limiting (respects API limits)
    - Exponential backoff on errors
    - Pagination handling
    - Response caching (optional)

    Example:
        async with PolymarketClient() as client:
            markets = await client.get_all_markets()
            trades = await client.get_all_trades_for_market(market_id)
    """

    BASE_URL = "https://clob.polymarket.com"
    GAMMA_API_URL = "https://gamma-api.polymarket.com"
    DATA_API_URL = "https://data-api.polymarket.com"  # Public trades endpoint

    def __init__(
        self,
        requests_per_second: float = 2.0,
        max_retries: int = 3,
        cache_enabled: bool = True,
        timeout: int = 60,
    ):
        self.rate_limiter = RateLimiter(rate=requests_per_second)
        self.max_retries = max_retries
        self.cache_enabled = cache_enabled
        self.timeout = aiohttp.ClientTimeout(total=timeout, connect=30)
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache: Dict[str, Any] = {} if cache_enabled else None

    async def __aenter__(self) -> "PolymarketClient":
        # Force IPv4 and use certifi for SSL certificates
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        connector = aiohttp.TCPConnector(family=socket.AF_INET, ssl=ssl_ctx)
        self._session = aiohttp.ClientSession(timeout=self.timeout, connector=connector)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    @property
    def session(self) -> aiohttp.ClientSession:
        if self._session is None:
            raise RuntimeError("Client not initialized. Use 'async with' context manager.")
        return self._session

    def _get_cache_key(self, endpoint: str, params: Dict) -> str:
        """Generate cache key from endpoint and params."""
        param_str = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        return f"{endpoint}?{param_str}"

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
    )
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        base_url: Optional[str] = None,
    ) -> Any:
        """Make an API request with rate limiting and error handling."""
        await self.rate_limiter.acquire()

        url = f"{base_url or self.BASE_URL}{endpoint}"
        params = params or {}

        # Check cache
        if self.cache_enabled and method == "GET":
            cache_key = self._get_cache_key(endpoint, params)
            if cache_key in self._cache:
                logger.debug(f"Cache hit: {cache_key}")
                return self._cache[cache_key]

        logger.debug(f"Request: {method} {url} params={params}")

        async with self.session.request(method, url, params=params) as response:
            if response.status == 429:
                retry_after = response.headers.get("Retry-After")
                retry_after = int(retry_after) if retry_after else None
                await self.rate_limiter.handle_rate_limit_error(retry_after)
                raise RateLimitError(retry_after)

            if response.status >= 400:
                text = await response.text()
                raise APIError(response.status, text)

            data = await response.json()

            # Cache response
            if self.cache_enabled and method == "GET":
                cache_key = self._get_cache_key(endpoint, params)
                self._cache[cache_key] = data

            return data

    async def get_markets(self, next_cursor: Optional[str] = None) -> Dict:
        """
        Fetch markets with pagination.

        Args:
            next_cursor: Pagination cursor for next page

        Returns:
            Dict with 'data' (list of markets) and 'next_cursor'
        """
        params = {}
        if next_cursor:
            params["next_cursor"] = next_cursor

        return await self._request("GET", "/markets", params=params)

    async def get_all_markets(self) -> List[Dict]:
        """
        Fetch all markets, handling pagination.

        Returns:
            List of all market dictionaries
        """
        all_markets = []
        next_cursor = None

        while True:
            response = await self.get_markets(next_cursor)

            # Handle different response formats
            if isinstance(response, list):
                all_markets.extend(response)
                break
            elif isinstance(response, dict):
                markets = response.get("data", response.get("markets", []))
                if isinstance(markets, list):
                    all_markets.extend(markets)
                next_cursor = response.get("next_cursor")
                # "LTE=" is base64 for -1, indicating end of pagination
                if not next_cursor or next_cursor == "LTE=":
                    break
            else:
                break

        logger.info(f"Fetched {len(all_markets)} markets")
        return all_markets

    async def get_events(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """
        Fetch events from Gamma API (contains market metadata).

        Args:
            limit: Number of events to fetch
            offset: Pagination offset

        Returns:
            List of event dictionaries with nested markets
        """
        params = {"limit": limit, "offset": offset, "active": "true"}
        return await self._request(
            "GET", "/events", params=params, base_url=self.GAMMA_API_URL
        )

    async def get_all_events(self, max_events: int = 1000) -> List[Dict]:
        """
        Fetch all events with pagination.

        Args:
            max_events: Maximum number of events to fetch

        Returns:
            List of all event dictionaries
        """
        all_events = []
        offset = 0
        limit = 100

        while len(all_events) < max_events:
            events = await self.get_events(limit=limit, offset=offset)

            if not events:
                break

            all_events.extend(events)
            offset += limit

            if len(events) < limit:
                break

        logger.info(f"Fetched {len(all_events)} events")
        return all_events

    async def get_trades(
        self,
        market_id: Optional[str] = None,
        user_address: Optional[str] = None,
        limit: int = 500,
        offset: int = 0,
    ) -> List[Dict]:
        """
        Fetch trades for a market or user using the public Data API.

        Args:
            market_id: Filter by market ID (conditionId)
            user_address: Filter by user wallet address
            limit: Number of trades to fetch (max 500)
            offset: Pagination offset

        Returns:
            List of trade dictionaries
        """
        params = {"limit": min(limit, 500)}

        if offset > 0:
            params["offset"] = offset
        if market_id:
            params["conditionId"] = market_id  # Data API uses conditionId
        if user_address:
            params["proxyWallet"] = user_address  # Data API uses proxyWallet

        # Use Data API for public trades access
        response = await self._request("GET", "/trades", params=params, base_url=self.DATA_API_URL)

        # Handle different response formats
        if isinstance(response, list):
            return response
        elif isinstance(response, dict):
            return response.get("data", response.get("trades", []))
        return []

    async def get_all_trades_for_market(
        self,
        market_id: str,
        max_trades: int = 100000,
        cutoff_timestamp: Optional[float] = None,
    ) -> List[Dict]:
        """
        Fetch all trades for a market, handling pagination.

        Continues paginating until:
        1. We reach max_trades, OR
        2. We get an empty response, OR
        3. All trades in a page are older than cutoff_timestamp

        Args:
            market_id: Market ID to fetch trades for
            max_trades: Maximum number of trades to fetch
            cutoff_timestamp: Optional Unix timestamp - stop when all trades are older than this

        Returns:
            List of all trade dictionaries for the market
        """
        all_trades = []
        offset = 0
        limit = 500
        consecutive_empty = 0

        while len(all_trades) < max_trades:
            trades = await self.get_trades(
                market_id=market_id,
                limit=limit,
                offset=offset,
            )

            if not trades:
                consecutive_empty += 1
                if consecutive_empty >= 3:
                    # API consistently returning empty, stop
                    break
                offset += limit
                continue

            consecutive_empty = 0
            all_trades.extend(trades)
            offset += limit

            # Check if we've reached trades older than cutoff
            if cutoff_timestamp:
                # Get oldest trade timestamp in this batch
                oldest_ts = None
                for t in trades:
                    ts = t.get("timestamp")
                    if isinstance(ts, (int, float)):
                        if ts > 1e12:
                            ts = ts / 1000
                        if oldest_ts is None or ts < oldest_ts:
                            oldest_ts = ts

                if oldest_ts and oldest_ts < cutoff_timestamp:
                    logger.debug(f"Reached cutoff date for market {market_id[:8]}, stopping pagination")
                    break

            # If we got fewer trades than requested, we've reached the end
            if len(trades) < limit:
                break

            # Progress logging for large fetches
            if len(all_trades) % 5000 == 0:
                logger.info(f"Fetched {len(all_trades)} trades for market {market_id[:8]}...")

        logger.debug(f"Fetched {len(all_trades)} trades for market {market_id[:8]}")
        return all_trades

    async def get_all_trades_for_user(
        self,
        user_address: str,
        max_trades: int = 100000,
    ) -> List[Dict]:
        """
        Fetch all trades for a user, handling pagination.

        Args:
            user_address: User wallet address
            max_trades: Maximum number of trades to fetch

        Returns:
            List of all trade dictionaries for the user
        """
        all_trades = []
        offset = 0
        limit = 500

        while len(all_trades) < max_trades:
            trades = await self.get_trades(
                user_address=user_address,
                limit=limit,
                offset=offset,
            )

            if not trades:
                break

            all_trades.extend(trades)
            offset += limit

            if len(trades) < limit:
                break

        logger.debug(f"Fetched {len(all_trades)} trades for user {user_address[:10]}...")
        return all_trades

    def clear_cache(self) -> None:
        """Clear the response cache."""
        if self._cache is not None:
            self._cache.clear()
            logger.debug("Cache cleared")
