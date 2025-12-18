#!/usr/bin/env python3
"""
ChainRank Event Fetcher
═══════════════════════════════════════════════════════════════════════════════
Downloads on-chain events from specified protocols and saves to CSV files.

Usage:
    python fetch_events.py --config config.toml
    python fetch_events.py --config config.toml --year 2025 --month 1

Environment Variables (in .env file):
    RPC_ETHEREUM=https://eth-mainnet.g.alchemy.com/v2/YOUR_API_KEY
    RPC_ARBITRUM=https://arb-mainnet.g.alchemy.com/v2/YOUR_API_KEY
    RPC_OPTIMISM=https://opt-mainnet.g.alchemy.com/v2/YOUR_API_KEY
"""

import argparse
import calendar
import csv
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import toml
from dotenv import load_dotenv
from web3 import Web3
from web3.exceptions import BlockNotFound

# Load environment variables from .env file
load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════════
# EVENT SIGNATURES (keccak256 hashes of event signatures)
# ═══════════════════════════════════════════════════════════════════════════════

EVENT_SIGNATURES = {
    # Aave V3 Events
    "Supply": "0x2b627736bca15cd5381dcf80b0bf11fd197d01a037c52b927a881a10fb73ba61",
    "Withdraw": "0x3115d1449a7b732c986cba18244e897a450f61e1bb8d589cd2e69e6c8924f9f7",
    "Borrow": "0xb3d084820fb1a9decffb176436bd02558d15fac9b0ddfed8c465bc7359d7dce0",
    "Repay": "0xa534c8dbe71f871f9f3530e97a74601fea17b426cae02e1c5aee42c96c784051",
    "LiquidationCall": "0xe413a321e8681d831f4dbccbca790d2952b56f977908e45be37335533e005286",
    "FlashLoan": "0xefefaba5e921573100900a3ad9cf29f222d995fb3b6045797eaea7521bd8d6f0",
    "ReserveUsedAsCollateralEnabled": "0x00058a56ea94653cdf4f152d227ace22d4c00ad99e2a43f58cb7d9e3feb295f2",
    "ReserveUsedAsCollateralDisabled": "0x44c58d81365b66dd4b1a7f36c25aa97b8c71c361ee4937adc1a00000227db5dd",
    # ERC-20 Events
    "Transfer": "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef",
    "Approval": "0x8c5be1e5ebec7d5bd14f71427d1e84f3dd0314c0f7b2291e5b200ac8c7c3b925",
    # Uniswap V3 Pool Events
    # Swap(address indexed sender, address indexed recipient, int256 amount0, int256 amount1, uint160 sqrtPriceX96, uint128 liquidity, int24 tick)
    "UniswapV3Swap": "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67",
}


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def load_config(config_path: str) -> dict:
    """Load configuration from TOML file."""
    with open(config_path, "r") as f:
        return toml.load(f)


def get_rpc_endpoint(chain_name: str) -> str:
    """Get RPC endpoint from environment variable."""
    env_var = f"RPC_{chain_name.upper()}"
    endpoint = os.getenv(env_var)
    if not endpoint:
        raise ValueError(f"Missing environment variable: {env_var}")
    return endpoint


def get_event_signature(event_name: str) -> str | None:
    """Get the keccak256 hash of an event signature."""
    return EVENT_SIGNATURES.get(event_name)


def get_block_by_timestamp(w3: Web3, target_timestamp: int) -> int:
    """Binary search to find block number closest to target timestamp."""
    latest_block = w3.eth.block_number
    latest_timestamp = w3.eth.get_block(latest_block)["timestamp"]

    if target_timestamp >= latest_timestamp:
        return latest_block

    # Estimate starting point (assuming ~12s block time)
    blocks_back = (latest_timestamp - target_timestamp) // 12
    low = max(0, latest_block - blocks_back - 10000)
    high = latest_block

    while low < high:
        mid = (low + high) // 2
        try:
            mid_timestamp = w3.eth.get_block(mid)["timestamp"]
        except BlockNotFound:
            high = mid - 1
            continue

        if mid_timestamp < target_timestamp:
            low = mid + 1
        else:
            high = mid

    return low


def decode_log_data(log: dict, event_name: str) -> dict[str, Any]:
    """Decode log data based on event type. Returns parsed fields."""
    topics = log.get("topics", [])
    data = log.get("data", "0x")

    decoded = {
        "raw_topics": [t.hex() if isinstance(t, bytes) else t for t in topics],
        "raw_data": data.hex() if isinstance(data, bytes) else data,
    }

    # Decode based on event type
    if event_name == "Transfer" and len(topics) >= 3:
        decoded["from"] = _extract_address(topics[1])
        decoded["to"] = _extract_address(topics[2])
        if len(data) >= 32:
            data_hex = data.hex() if isinstance(data, bytes) else data
            decoded["amount"] = int(data_hex[2:66], 16) if data_hex != "0x" else 0

    elif event_name == "Supply" and len(topics) >= 3:
        decoded["reserve"] = _extract_address(topics[1])
        decoded["on_behalf_of"] = _extract_address(topics[2])

    elif event_name == "Borrow" and len(topics) >= 3:
        decoded["reserve"] = _extract_address(topics[1])
        decoded["on_behalf_of"] = _extract_address(topics[2])

    elif event_name == "Withdraw" and len(topics) >= 3:
        decoded["reserve"] = _extract_address(topics[1])
        decoded["user"] = _extract_address(topics[2])

    elif event_name == "Repay" and len(topics) >= 3:
        decoded["reserve"] = _extract_address(topics[1])
        decoded["user"] = _extract_address(topics[2])

    elif event_name == "LiquidationCall" and len(topics) >= 4:
        decoded["collateral_asset"] = _extract_address(topics[1])
        decoded["debt_asset"] = _extract_address(topics[2])
        decoded["user"] = _extract_address(topics[3])

    elif event_name == "UniswapV3Swap" and len(topics) >= 3:
        # Swap(address indexed sender, address indexed recipient, int256 amount0, int256 amount1, uint160 sqrtPriceX96, uint128 liquidity, int24 tick)
        decoded["sender"] = _extract_address(topics[1])
        decoded["recipient"] = _extract_address(topics[2])
        data_hex = data.hex() if isinstance(data, bytes) else data
        if len(data_hex) >= 322:  # 0x + 5*64 chars
            # amount0 and amount1 are signed int256
            amount0_hex = data_hex[2:66]
            amount1_hex = data_hex[66:130]
            decoded["amount0"] = (
                int(amount0_hex, 16)
                if int(amount0_hex, 16) < 2**255
                else int(amount0_hex, 16) - 2**256
            )
            decoded["amount1"] = (
                int(amount1_hex, 16)
                if int(amount1_hex, 16) < 2**255
                else int(amount1_hex, 16) - 2**256
            )
            decoded["sqrt_price_x96"] = int(data_hex[130:194], 16)
            decoded["liquidity"] = int(data_hex[194:258], 16)
            # tick is int24
            tick_hex = data_hex[258:322]
            tick_val = int(tick_hex, 16)
            decoded["tick"] = tick_val if tick_val < 2**23 else tick_val - 2**24

    return decoded


def _extract_address(topic) -> str:
    """Extract address from topic."""
    if isinstance(topic, bytes):
        return "0x" + topic.hex()[-40:]
    return "0x" + topic[-40:]


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EVENT FETCHER CLASS
# ═══════════════════════════════════════════════════════════════════════════════


class EventFetcher:
    """Fetches and stores on-chain events based on configuration."""

    def __init__(self, config: dict):
        self.config = config
        self.year = config.get("year", datetime.now(timezone.utc).year)
        self.month = config.get("month", datetime.now(timezone.utc).month)
        self.seed_contracts = config.get("seed_contracts", {})
        self.allowlisted_events = config.get("allowlisted_events", {})
        self.output_config = config.get("output", {})
        self.indexer_config = config.get("indexer", {})

        # Derive chains from seed_contracts
        self.chains = self._extract_chains_from_contracts()

        self.batch_size = self.indexer_config.get("batch_size", 2000)
        self.max_retries = self.indexer_config.get("max_retries", 3)
        self.retry_delay = self.indexer_config.get("retry_delay_seconds", 1)
        self.rate_limit = self.indexer_config.get("rate_limit_per_second", 10)

        self.raw_folder = Path(self.output_config.get("raw_folder", "raw"))
        self.raw_folder.mkdir(parents=True, exist_ok=True)

        self._last_request_time = 0

    def _extract_chains_from_contracts(self) -> set[str]:
        """Extract unique chain names from seed_contracts."""
        chains = set()
        for protocol_contracts in self.seed_contracts.values():
            for chain_name in protocol_contracts.keys():
                chains.add(chain_name)
        return chains

    def _rate_limit_wait(self):
        """Enforce rate limiting between requests."""
        now = time.time()
        min_interval = 1.0 / self.rate_limit
        elapsed = now - self._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time = time.time()

    def _get_web3(self, chain_name: str) -> Web3:
        """Get Web3 instance for a chain."""
        rpc_url = get_rpc_endpoint(chain_name)
        return Web3(Web3.HTTPProvider(rpc_url))

    def _fetch_logs_batch(
        self,
        w3: Web3,
        contract_address: str,
        topics: list[str],
        from_block: int,
        to_block: int,
    ) -> list[dict]:
        """Fetch logs for a batch of blocks with retry logic."""
        for attempt in range(self.max_retries):
            try:
                self._rate_limit_wait()
                logs = w3.eth.get_logs(
                    {
                        "address": Web3.to_checksum_address(contract_address),
                        "topics": [topics],
                        "fromBlock": from_block,
                        "toBlock": to_block,
                    }
                )
                return logs
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"  Retry {attempt + 1}/{self.max_retries} after error: {e}")
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    print(f"  Failed after {self.max_retries} attempts: {e}")
                    return []
        return []

    def fetch_events_for_contract(
        self,
        chain_name: str,
        protocol_name: str,
        contract_address: str,
        event_names: list[str],
        from_block: int,
        to_block: int,
    ) -> list[dict]:
        """Fetch all specified events for a contract."""
        w3 = self._get_web3(chain_name)
        all_events = []

        # Get event signatures
        event_topics = []
        event_map = {}
        for event_name in event_names:
            sig = get_event_signature(event_name)
            if sig:
                event_topics.append(sig)
                event_map[sig.lower()] = event_name

        if not event_topics:
            print(f"  No valid event signatures for {protocol_name}")
            return []

        total_blocks = to_block - from_block + 1
        num_batches = (total_blocks + self.batch_size - 1) // self.batch_size
        print(f"  Fetching {len(event_topics)} event types")
        print(
            f"  Block range: {from_block:,} → {to_block:,} ({total_blocks:,} blocks, {num_batches} batches)"
        )

        # Fetch in batches
        current_block = from_block
        total_events = 0
        batch_count = 0
        start_time = time.time()

        while current_block <= to_block:
            batch_end = min(current_block + self.batch_size - 1, to_block)
            batch_count += 1

            print(
                f"    Fetching batch {batch_count}/{num_batches}: blocks {current_block:,} - {batch_end:,}...",
                end="\r",
            )

            logs = self._fetch_logs_batch(
                w3, contract_address, event_topics, current_block, batch_end
            )

            for log in logs:
                topic0 = (
                    log["topics"][0].hex()
                    if isinstance(log["topics"][0], bytes)
                    else log["topics"][0]
                )
                # Ensure 0x prefix for matching
                if not topic0.startswith("0x"):
                    topic0 = "0x" + topic0
                event_name = event_map.get(topic0.lower(), "Unknown")

                # Decode event data
                decoded = decode_log_data(log, event_name)

                event_record = {
                    "chain_name": chain_name,
                    "protocol": protocol_name,
                    "contract_address": contract_address,
                    "event_name": event_name,
                    "block_number": log["blockNumber"],
                    "transaction_hash": (
                        log["transactionHash"].hex()
                        if isinstance(log["transactionHash"], bytes)
                        else log["transactionHash"]
                    ),
                    "log_index": log["logIndex"],
                    "tx_index": log.get("transactionIndex", 0),
                    **decoded,
                }
                all_events.append(event_record)

            total_events += len(logs)
            progress = (batch_end - from_block) / max(1, (to_block - from_block)) * 100
            elapsed = time.time() - start_time
            blocks_done = batch_end - from_block + 1
            blocks_per_sec = blocks_done / max(1, elapsed)
            eta_seconds = (to_block - batch_end) / max(1, blocks_per_sec)

            print(
                f"    Batch {batch_count}/{num_batches} ({progress:.1f}%) | "
                f"{total_events:,} events | {blocks_per_sec:.0f} blk/s | ETA: {eta_seconds:.0f}s"
                + " "
                * 10,
                end="\r",
            )

            current_block = batch_end + 1

        elapsed = time.time() - start_time
        print(f"\n  ✓ Completed: {total_events:,} events in {elapsed:.1f}s")
        return all_events

    def _get_existing_files(self, chain_name: str) -> set[str]:
        """Get set of protocol names that already have files for current year/month."""
        year_month = f"{self.year}_{self.month:02d}"
        pattern = f"{chain_name}_*_{year_month}.csv"
        existing_files = list(self.raw_folder.glob(pattern))

        # Extract protocol names from filenames like "ethereum_usdc_2025_01.csv"
        existing_protocols = set()
        for f in existing_files:
            parts = f.stem.split("_")  # e.g., ["ethereum", "usdc", "2025", "01"]
            if len(parts) >= 4:
                # Protocol name is everything between chain and year_month
                protocol = "_".join(parts[1:-2])
                existing_protocols.add(protocol)

        return existing_protocols

    def fetch_all_events(
        self, chains_filter: list[str] | None = None, contract_filter: str | None = None
    ) -> dict[str, list[dict]]:
        """Fetch events for all configured contracts and chains.

        Args:
            chains_filter: Optional list of chain names to fetch (e.g., ["ethereum", "arbitrum"])
            contract_filter: Optional protocol/contract name to fetch (e.g., "usdc", "aave")
        """
        all_events = {}

        # Calculate time window for the specified month
        start_time = datetime(self.year, self.month, 1, tzinfo=timezone.utc)
        _, last_day = calendar.monthrange(self.year, self.month)
        end_time = datetime(
            self.year, self.month, last_day, 23, 59, 59, tzinfo=timezone.utc
        )

        start_timestamp = int(start_time.timestamp())
        end_timestamp = int(end_time.timestamp())

        print("═" * 79)
        print("ChainRank Event Fetcher")
        print("═" * 79)
        print(f"Time window: {start_time.isoformat()} to {end_time.isoformat()}")
        print(f"Year: {self.year}, Month: {self.month}")
        print()

        # Iterate through chains
        for chain_name in self.chains:
            if chains_filter and chain_name not in chains_filter:
                continue

            print("─" * 79)
            print(f"Chain: {chain_name}")
            print("─" * 79)

            try:
                w3 = self._get_web3(chain_name)
                if not w3.is_connected():
                    print(f"  ✗ Failed to connect to {chain_name}")
                    continue

                # Get block range for the month
                from_block = get_block_by_timestamp(w3, start_timestamp)
                to_block = get_block_by_timestamp(w3, end_timestamp)
                print(f"  Block range: {from_block} to {to_block}")

            except Exception as e:
                print(f"  ✗ Error connecting to {chain_name}: {e}")
                continue

            # Get existing files for this chain
            existing_protocols = self._get_existing_files(chain_name)
            if existing_protocols:
                print(f"  Already fetched: {', '.join(sorted(existing_protocols))}")

            # Iterate through protocols
            for protocol_name, addresses in self.seed_contracts.items():
                if contract_filter and protocol_name.lower() != contract_filter:
                    continue
                if chain_name not in addresses:
                    continue

                # Skip if already fetched
                if protocol_name in existing_protocols:
                    print(f"\n  Skipping {protocol_name}: already exists in raw/")
                    continue

                contract_address = addresses[chain_name]
                event_names = self.allowlisted_events.get(protocol_name, [])

                if not event_names:
                    print(f"  Skipping {protocol_name}: no allowlisted events")
                    continue

                print(f"\n  Protocol: {protocol_name}")
                print(f"  Contract: {contract_address}")
                print(f"  Events: {', '.join(event_names)}")

                events = self.fetch_events_for_contract(
                    chain_name,
                    protocol_name,
                    contract_address,
                    event_names,
                    from_block,
                    to_block,
                )

                key = f"{chain_name}_{protocol_name}"
                all_events[key] = events

        return all_events

    def save_to_csv(self, events: dict[str, list[dict]]):
        """Save fetched events to CSV files."""
        print()
        print("═" * 79)
        print("Saving to CSV")
        print("═" * 79)

        year_month = f"{self.year}_{self.month:02d}"

        # Define field order
        priority_fields = [
            "chain_name",
            "protocol",
            "contract_address",
            "event_name",
            "block_number",
            "transaction_hash",
            "log_index",
            "tx_index",
            "from",
            "to",
            "amount",
            "reserve",
            "user",
            "on_behalf_of",
            "collateral_asset",
            "debt_asset",
        ]

        for key, event_list in events.items():
            if not event_list:
                print(f"  Skipping {key}: no events")
                continue

            filename = f"{key}_{year_month}.csv"
            filepath = self.raw_folder / filename

            # Get all unique fields
            all_fields = set()
            for event in event_list:
                all_fields.update(event.keys())

            fieldnames = [f for f in priority_fields if f in all_fields]
            fieldnames += sorted([f for f in all_fields if f not in priority_fields])

            with open(filepath, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(event_list)

            print(f"  ✓ {filename}: {len(event_list)} events")

        print(f"\nOutput folder: {self.raw_folder.absolute()}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="ChainRank Event Fetcher - Download on-chain events to CSV"
    )
    parser.add_argument(
        "--config", type=str, default="config.toml", help="Path to config.toml file"
    )
    parser.add_argument(
        "--chains",
        type=str,
        default=None,
        help="Comma-separated list of chains to fetch (e.g., ethereum,arbitrum)",
    )
    parser.add_argument(
        "--year", type=int, default=None, help="Year to fetch events for (e.g., 2025)"
    )
    parser.add_argument(
        "--month",
        type=int,
        default=None,
        help="Month to fetch events for (1-12)",
    )
    parser.add_argument(
        "--contract",
        type=str,
        default=None,
        help="Fetch events for a specific protocol/contract only (e.g., usdc, aave)",
    )

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1

    config = load_config(config_path)

    # Override config with CLI args
    if args.year:
        config["year"] = args.year
    if args.month:
        if not 1 <= args.month <= 12:
            print(f"Error: Month must be between 1 and 12, got {args.month}")
            return 1
        config["month"] = args.month

    # Parse chains filter
    chains_filter = None
    if args.chains:
        chains_filter = [c.strip() for c in args.chains.split(",")]

    # Parse contract filter
    contract_filter = None
    if args.contract:
        contract_filter = args.contract.strip().lower()

    # Run fetcher
    fetcher = EventFetcher(config)
    events = fetcher.fetch_all_events(chains_filter, contract_filter)
    fetcher.save_to_csv(events)

    print("\n✓ Done!")
    return 0


if __name__ == "__main__":
    exit(main())
