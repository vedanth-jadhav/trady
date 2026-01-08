"""
Data validation utilities for quality checks.

Validates ingested data for:
- Schema compliance
- Data quality issues
- Logical consistency
"""

from datetime import datetime, timezone
from typing import List, Tuple
import logging

import pandas as pd

logger = logging.getLogger(__name__)


class ValidationError:
    """Represents a validation error."""

    def __init__(self, level: str, field: str, message: str, count: int = 1):
        self.level = level  # "error", "warning", "info"
        self.field = field
        self.message = message
        self.count = count

    def __str__(self):
        return f"[{self.level.upper()}] {self.field}: {self.message} (count: {self.count})"


class DataValidator:
    """
    Validates ingested data for quality issues.

    Provides validation methods for:
    - Markets data
    - Trades data
    - Wallets data
    """

    def __init__(self):
        self.errors: List[ValidationError] = []

    def clear_errors(self):
        """Clear accumulated errors."""
        self.errors = []

    def get_errors(self, level: str = None) -> List[ValidationError]:
        """Get errors, optionally filtered by level."""
        if level:
            return [e for e in self.errors if e.level == level]
        return self.errors

    def is_valid(self) -> bool:
        """Check if validation passed (no errors)."""
        return not any(e.level == "error" for e in self.errors)

    def validate_markets(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate markets data.

        Checks:
        - No duplicate market IDs
        - All required fields present
        - Volume >= 0
        - Dates are valid
        """
        self.clear_errors()
        messages = []

        if df.empty:
            self.errors.append(ValidationError("warning", "markets", "Empty DataFrame"))
            return True, ["Warning: Empty markets DataFrame"]

        # Check required columns
        required = ["market_id", "question", "volume"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            self.errors.append(ValidationError("error", "schema", f"Missing columns: {missing}"))
            messages.append(f"Error: Missing columns: {missing}")

        # Check for duplicates
        if "market_id" in df.columns:
            duplicates = df["market_id"].duplicated().sum()
            if duplicates > 0:
                self.errors.append(ValidationError("error", "market_id", f"Duplicate IDs found", duplicates))
                messages.append(f"Error: {duplicates} duplicate market IDs")

        # Check volume >= 0
        if "volume" in df.columns:
            negative_volume = (df["volume"] < 0).sum()
            if negative_volume > 0:
                self.errors.append(ValidationError("error", "volume", "Negative volume found", negative_volume))
                messages.append(f"Error: {negative_volume} markets with negative volume")

        # Check dates
        date_cols = ["created_at", "end_date"]
        for col in date_cols:
            if col in df.columns:
                null_dates = df[col].isna().sum()
                if null_dates > 0:
                    self.errors.append(ValidationError("warning", col, f"Null dates found", null_dates))
                    messages.append(f"Warning: {null_dates} null values in {col}")

        # Summary
        if self.is_valid():
            messages.append(f"✓ Markets validation passed ({len(df)} records)")
        else:
            messages.append(f"✗ Markets validation failed with {len([e for e in self.errors if e.level == 'error'])} errors")

        logger.info("\n".join(messages))
        return self.is_valid(), messages

    def validate_trades(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate trades data.

        Checks:
        - No duplicate trade IDs
        - Price in range [0, 1]
        - Size > 0
        - Valid wallet addresses (0x...)
        - Timestamps within expected range
        """
        self.clear_errors()
        messages = []

        if df.empty:
            self.errors.append(ValidationError("warning", "trades", "Empty DataFrame"))
            return True, ["Warning: Empty trades DataFrame"]

        # Check required columns
        required = ["trade_id", "market_id", "timestamp", "maker_address", "taker_address", "size", "price"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            self.errors.append(ValidationError("error", "schema", f"Missing columns: {missing}"))
            messages.append(f"Error: Missing columns: {missing}")
            return False, messages

        # Check for duplicates
        duplicates = df["trade_id"].duplicated().sum()
        if duplicates > 0:
            self.errors.append(ValidationError("error", "trade_id", "Duplicate IDs found", duplicates))
            messages.append(f"Error: {duplicates} duplicate trade IDs")

        # Check price range [0, 1]
        invalid_price = ((df["price"] < 0) | (df["price"] > 1)).sum()
        if invalid_price > 0:
            self.errors.append(ValidationError("error", "price", "Price outside [0,1] range", invalid_price))
            messages.append(f"Error: {invalid_price} trades with invalid price")

        # Check size > 0
        invalid_size = (df["size"] <= 0).sum()
        if invalid_size > 0:
            self.errors.append(ValidationError("error", "size", "Non-positive size", invalid_size))
            messages.append(f"Error: {invalid_size} trades with invalid size")

        # Check wallet addresses
        for col in ["maker_address", "taker_address"]:
            if col in df.columns:
                # Check for valid format (starts with 0x)
                invalid_addr = (~df[col].str.startswith("0x", na=True) & df[col].notna() & (df[col] != "")).sum()
                if invalid_addr > 0:
                    self.errors.append(ValidationError("warning", col, "Invalid address format", invalid_addr))
                    messages.append(f"Warning: {invalid_addr} invalid {col} format")

                # Check for empty/null
                empty_addr = (df[col].isna() | (df[col] == "")).sum()
                if empty_addr > 0:
                    self.errors.append(ValidationError("warning", col, "Empty addresses", empty_addr))
                    messages.append(f"Warning: {empty_addr} empty {col}")

        # Check timestamps
        if "timestamp" in df.columns:
            now = datetime.now(timezone.utc)
            future_trades = (df["timestamp"] > now).sum()
            if future_trades > 0:
                self.errors.append(ValidationError("warning", "timestamp", "Future timestamps", future_trades))
                messages.append(f"Warning: {future_trades} trades with future timestamps")

            # Check for very old trades (> 1 year)
            one_year_ago = now.replace(year=now.year - 1)
            old_trades = (df["timestamp"] < one_year_ago).sum()
            if old_trades > 0:
                self.errors.append(ValidationError("info", "timestamp", "Trades older than 1 year", old_trades))
                messages.append(f"Info: {old_trades} trades older than 1 year")

        # Summary
        if self.is_valid():
            messages.append(f"✓ Trades validation passed ({len(df)} records)")
        else:
            messages.append(f"✗ Trades validation failed with {len([e for e in self.errors if e.level == 'error'])} errors")

        logger.info("\n".join(messages))
        return self.is_valid(), messages

    def validate_wallets(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate wallet index data.

        Checks:
        - No duplicate addresses
        - first_seen <= last_seen
        - total_trades > 0
        """
        self.clear_errors()
        messages = []

        if df.empty:
            self.errors.append(ValidationError("warning", "wallets", "Empty DataFrame"))
            return True, ["Warning: Empty wallets DataFrame"]

        # Check required columns
        required = ["address", "first_seen", "last_seen", "total_trades", "total_volume"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            self.errors.append(ValidationError("error", "schema", f"Missing columns: {missing}"))
            messages.append(f"Error: Missing columns: {missing}")
            return False, messages

        # Check for duplicates
        duplicates = df["address"].duplicated().sum()
        if duplicates > 0:
            self.errors.append(ValidationError("error", "address", "Duplicate addresses", duplicates))
            messages.append(f"Error: {duplicates} duplicate wallet addresses")

        # Check first_seen <= last_seen
        if "first_seen" in df.columns and "last_seen" in df.columns:
            invalid_dates = (df["first_seen"] > df["last_seen"]).sum()
            if invalid_dates > 0:
                self.errors.append(ValidationError("error", "dates", "first_seen > last_seen", invalid_dates))
                messages.append(f"Error: {invalid_dates} wallets with first_seen > last_seen")

        # Check total_trades > 0
        if "total_trades" in df.columns:
            zero_trades = (df["total_trades"] <= 0).sum()
            if zero_trades > 0:
                self.errors.append(ValidationError("error", "total_trades", "Non-positive trade count", zero_trades))
                messages.append(f"Error: {zero_trades} wallets with zero trades")

        # Check total_volume >= 0
        if "total_volume" in df.columns:
            negative_volume = (df["total_volume"] < 0).sum()
            if negative_volume > 0:
                self.errors.append(ValidationError("error", "total_volume", "Negative volume", negative_volume))
                messages.append(f"Error: {negative_volume} wallets with negative volume")

        # Summary
        if self.is_valid():
            messages.append(f"✓ Wallets validation passed ({len(df)} records)")
        else:
            messages.append(f"✗ Wallets validation failed with {len([e for e in self.errors if e.level == 'error'])} errors")

        logger.info("\n".join(messages))
        return self.is_valid(), messages

    def validate_all(
        self,
        markets: pd.DataFrame = None,
        trades: pd.DataFrame = None,
        wallets: pd.DataFrame = None,
    ) -> Tuple[bool, List[str]]:
        """
        Validate all data types at once.

        Returns:
            Tuple of (all_valid, all_messages)
        """
        all_valid = True
        all_messages = []

        if markets is not None:
            valid, messages = self.validate_markets(markets)
            all_valid = all_valid and valid
            all_messages.extend(messages)

        if trades is not None:
            valid, messages = self.validate_trades(trades)
            all_valid = all_valid and valid
            all_messages.extend(messages)

        if wallets is not None:
            valid, messages = self.validate_wallets(wallets)
            all_valid = all_valid and valid
            all_messages.extend(messages)

        return all_valid, all_messages

    def generate_report(self) -> str:
        """Generate a validation report."""
        lines = ["=" * 50, "DATA VALIDATION REPORT", "=" * 50, ""]

        error_counts = {"error": 0, "warning": 0, "info": 0}
        for e in self.errors:
            error_counts[e.level] += 1

        lines.append(f"Errors:   {error_counts['error']}")
        lines.append(f"Warnings: {error_counts['warning']}")
        lines.append(f"Info:     {error_counts['info']}")
        lines.append("")

        if self.errors:
            lines.append("Details:")
            lines.append("-" * 40)
            for e in self.errors:
                lines.append(str(e))

        lines.append("")
        lines.append("=" * 50)

        return "\n".join(lines)
