#!/usr/bin/env python3
"""
ChainRank ENS Names Fetcher
═══════════════════════════════════════════════════════════════════════════════
Downloads ENS names from Alchemy by fetching NameRegistered events from the
ENS registry contracts.

Usage:
    python fetch_ens_names.py --chain ethereum --year 2025
    python fetch_ens_names.py --chain base --year 2024

Environment Variables (in .env file):
    RPC_ETHEREUM=https://eth-mainnet.g.alchemy.com/v2/YOUR_API_KEY
    RPC_BASE=https://base-mainnet.g.alchemy.com/v2/YOUR_API_KEY
"""

import argparse
import csv
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv
from web3 import Web3

load_dotenv()

# ENS Registry contracts and events
# ETH Registrar Controller (for .eth names)
ENS_ETH_REGISTRAR = "0x253553366Da8546fC250F225fe3d25d0C782303b"  # Current controller
ENS_OLD_REGISTRAR = "0x283Af0B28c62C092C9727F1Ee09c02CA627EB7F5"  # Old controller

# NameRegistered event signature
# NameRegistered(string name, bytes32 indexed label, address indexed owner, uint256 baseCost, uint256 premium, uint256 expires)
NAME_REGISTERED_SIG = (
    "0x69e37f151eb98a09618ddaa80c8cfaf1ce5996867c489f45b555b412271ebf27"
)

# Old NameRegistered event (different signature)
# NameRegistered(string name, bytes32 indexed label, address indexed owner, uint256 cost, uint256 expires)
NAME_REGISTERED_OLD_SIG = (
    "0xca6abbe9d7f11422cb6ca7629fbf6fe9efb1c621f71ce8f02b9f2a230097404f"
)

# Approximate blocks per year (assuming ~12s block time for Ethereum)
BLOCKS_PER_DAY = 7200
BLOCKS_PER_YEAR = BLOCKS_PER_DAY * 365

# Request settings
BATCH_SIZE = 2000
MAX_RETRIES = 3
RETRY_DELAY = 2
PARALLEL_REQUESTS = 10


def get_rpc_url(chain: str) -> str:
    """Get RPC URL for the specified chain."""
    if chain == "ethereum":
        return os.getenv("RPC_ETHEREUM", "https://eth.llamarpc.com")
    elif chain == "base":
        return os.getenv("RPC_BASE", "https://base.llamarpc.com")
    else:
        return os.getenv(f"RPC_{chain.upper()}", f"https://{chain}.llamarpc.com")


def get_block_by_timestamp(rpc_url: str, timestamp: int) -> int:
    """Get approximate block number for a given timestamp using binary search."""
    # Get current block
    payload = {
        "jsonrpc": "2.0",
        "method": "eth_blockNumber",
        "params": [],
        "id": 1,
    }
    response = requests.post(rpc_url, json=payload, timeout=30)
    current_block = int(response.json()["result"], 16)

    # Get current block timestamp
    payload = {
        "jsonrpc": "2.0",
        "method": "eth_getBlockByNumber",
        "params": [hex(current_block), False],
        "id": 1,
    }
    response = requests.post(rpc_url, json=payload, timeout=30)
    current_timestamp = int(response.json()["result"]["timestamp"], 16)

    # Estimate blocks difference based on ~12s block time
    time_diff = current_timestamp - timestamp
    blocks_diff = time_diff // 12

    estimated_block = max(1, current_block - blocks_diff)

    # Refine with a few iterations
    for _ in range(5):
        payload = {
            "jsonrpc": "2.0",
            "method": "eth_getBlockByNumber",
            "params": [hex(estimated_block), False],
            "id": 1,
        }
        response = requests.post(rpc_url, json=payload, timeout=30)
        result = response.json().get("result")
        if not result:
            estimated_block = max(1, estimated_block - 1000)
            continue

        block_timestamp = int(result["timestamp"], 16)

        if abs(block_timestamp - timestamp) < 3600:  # Within 1 hour
            break

        # Adjust estimate
        time_diff = block_timestamp - timestamp
        block_adjustment = time_diff // 12
        estimated_block = max(1, estimated_block - block_adjustment)

    return estimated_block


def get_year_block_range(rpc_url: str, year: int) -> tuple[int, int]:
    """Get block range for a specific year."""
    print(f"  Calculating block range for year {year}...")

    # Start of year
    start_timestamp = int(
        datetime(year, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp()
    )
    # End of year
    end_timestamp = int(
        datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc).timestamp()
    )

    # Get current timestamp to check if end_timestamp is in the future
    current_timestamp = int(datetime.now(timezone.utc).timestamp())
    if end_timestamp > current_timestamp:
        end_timestamp = current_timestamp

    start_block = get_block_by_timestamp(rpc_url, start_timestamp)
    end_block = get_block_by_timestamp(rpc_url, end_timestamp)

    print(f"    Start block: {start_block:,}")
    print(f"    End block: {end_block:,}")
    print(f"    Total blocks: {end_block - start_block:,}")

    return start_block, end_block


def fetch_logs_batch(
    rpc_url: str,
    contract: str,
    topic: str,
    from_block: int,
    to_block: int,
) -> list[dict]:
    """Fetch logs for a batch of blocks."""
    payload = {
        "jsonrpc": "2.0",
        "method": "eth_getLogs",
        "params": [
            {
                "address": contract,
                "topics": [topic],
                "fromBlock": hex(from_block),
                "toBlock": hex(to_block),
            }
        ],
        "id": 1,
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(rpc_url, json=payload, timeout=60)
            result = response.json()

            if "error" in result:
                error_msg = result["error"].get("message", "Unknown error")
                if "too many" in error_msg.lower() or "limit" in error_msg.lower():
                    # Reduce batch size by splitting
                    return None
                raise Exception(error_msg)

            return result.get("result", [])

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                print(
                    f"    Warning: Failed to fetch blocks {from_block}-{to_block}: {e}"
                )
                return []

    return []


def decode_name_registered_event(log: dict, is_old_format: bool = False) -> dict | None:
    """Decode a NameRegistered event log."""
    try:
        topics = log.get("topics", [])
        data = log.get("data", "0x")

        if len(topics) < 3:
            return None

        # Owner is indexed (topic 2)
        owner = "0x" + topics[2][-40:]

        # Decode data - name is a dynamic string at the beginning
        data_bytes = (
            bytes.fromhex(data[2:]) if data.startswith("0x") else bytes.fromhex(data)
        )

        if len(data_bytes) < 64:
            return None

        # First 32 bytes is offset to string
        # Next 32 bytes at that offset is string length
        # Then the actual string data
        offset = int.from_bytes(data_bytes[0:32], "big")
        if offset + 32 > len(data_bytes):
            return None

        str_length = int.from_bytes(data_bytes[offset : offset + 32], "big")
        if offset + 32 + str_length > len(data_bytes):
            return None

        name_bytes = data_bytes[offset + 32 : offset + 32 + str_length]

        try:
            name = name_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return None

        # Skip empty or invalid names
        if not name or len(name) < 1:
            return None

        # Add .eth suffix
        ens_name = f"{name}.eth"

        return {
            "address": owner.lower(),
            "ens_name": ens_name,
        }

    except Exception:
        return None


def fetch_ens_names(
    rpc_url: str,
    start_block: int,
    end_block: int,
) -> list[dict]:
    """Fetch all ENS name registrations in the given block range."""
    all_names = {}  # Use dict to deduplicate by address
    total_blocks = end_block - start_block

    # Contracts and their event signatures to check
    contracts_events = [
        (ENS_ETH_REGISTRAR, NAME_REGISTERED_SIG, False),
        (ENS_OLD_REGISTRAR, NAME_REGISTERED_OLD_SIG, True),
    ]

    for contract, event_sig, is_old in contracts_events:
        contract_name = "Old Registrar" if is_old else "Current Registrar"
        print(f"\n  Fetching from {contract_name}...")
        print(f"    Contract: {contract}")

        # Create batches
        batches = []
        current_block = start_block
        while current_block < end_block:
            batch_end = min(current_block + BATCH_SIZE, end_block)
            batches.append((current_block, batch_end))
            current_block = batch_end

        print(f"    Batches: {len(batches)}")

        # Process batches
        processed = 0
        events_found = 0

        # Use thread pool for parallel requests
        with ThreadPoolExecutor(max_workers=PARALLEL_REQUESTS) as executor:
            futures = {
                executor.submit(
                    fetch_logs_batch, rpc_url, contract, event_sig, batch[0], batch[1]
                ): batch
                for batch in batches
            }

            for future in as_completed(futures):
                batch = futures[future]
                processed += 1

                try:
                    logs = future.result()

                    if logs is None:
                        # Need to split batch - fetch sequentially with smaller range
                        mid = (batch[0] + batch[1]) // 2
                        logs1 = fetch_logs_batch(
                            rpc_url, contract, event_sig, batch[0], mid
                        )
                        logs2 = fetch_logs_batch(
                            rpc_url, contract, event_sig, mid, batch[1]
                        )
                        logs = (logs1 or []) + (logs2 or [])

                    if logs:
                        for log in logs:
                            decoded = decode_name_registered_event(log, is_old)
                            if decoded:
                                # Keep the latest registration for each address
                                all_names[decoded["address"]] = decoded["ens_name"]
                                events_found += 1

                except Exception as e:
                    print(f"    Warning: Batch {batch} failed: {e}")

                # Progress update
                if processed % 50 == 0 or processed == len(batches):
                    progress = processed / len(batches) * 100
                    print(
                        f"    Progress: {processed}/{len(batches)} ({progress:.1f}%) - {events_found} names found"
                    )

        print(f"    ✓ Found {events_found} registrations from {contract_name}")

    # Convert to list
    result = [{"address": addr, "ens_name": name} for addr, name in all_names.items()]

    return result


def save_ens_names(ens_names: list[dict], output_path: Path):
    """Save ENS names to CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Sort by address
    ens_names.sort(key=lambda x: x["address"])

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["address", "ens_name"])
        writer.writeheader()
        writer.writerows(ens_names)

    print(f"  ✓ Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="ChainRank ENS Names Fetcher")
    parser.add_argument(
        "--chain",
        type=str,
        default="ethereum",
        help="Chain to fetch ENS names for (default: ethereum)",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2025,
        help="Year to filter ENS registrations (default: 2025)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="raw",
        help="Output directory for ENS names file (default: raw)",
    )

    args = parser.parse_args()

    print("═" * 79)
    print("ChainRank ENS Names Fetcher")
    print("═" * 79)

    # ENS is only on Ethereum mainnet
    if args.chain.lower() != "ethereum":
        print(f"\nNote: ENS registry is on Ethereum mainnet.")
        print(f"      Fetching ENS names registered in {args.year} from Ethereum.")
        chain = "ethereum"
    else:
        chain = args.chain.lower()

    rpc_url = get_rpc_url(chain)

    print(f"\n{'─' * 79}")
    print("Configuration")
    print(f"{'─' * 79}")
    print(f"  Chain: {chain}")
    print(f"  Year: {args.year}")
    print(f"  RPC URL: {rpc_url[:50]}...")

    output_path = Path(args.output_dir) / f"ens_names_{args.chain}_{args.year}.csv"
    print(f"  Output: {output_path}")

    print(f"\n{'─' * 79}")
    print("Calculating Block Range")
    print(f"{'─' * 79}")

    try:
        start_block, end_block = get_year_block_range(rpc_url, args.year)
    except Exception as e:
        print(f"  Error calculating block range: {e}")
        return 1

    print(f"\n{'─' * 79}")
    print("Fetching ENS Names")
    print(f"{'─' * 79}")

    ens_names = fetch_ens_names(rpc_url, start_block, end_block)

    if not ens_names:
        print("\n  No ENS names found. Exiting.")
        return 1

    print(f"\n{'─' * 79}")
    print("Saving Results")
    print(f"{'─' * 79}")

    save_ens_names(ens_names, output_path)

    # Summary
    print(f"\n{'─' * 79}")
    print("Summary")
    print(f"{'─' * 79}")
    print(f"  Total unique addresses with ENS: {len(ens_names):,}")
    print(f"  Output file: {output_path}")

    # Show sample
    print(f"\n  Sample names:")
    for item in ens_names[:5]:
        print(
            f"    {item['address'][:10]}...{item['address'][-6:]}: {item['ens_name']}"
        )

    print(f"\n{'═' * 79}")
    print("Done!")
    print("═" * 79)

    return 0


if __name__ == "__main__":
    exit(main())
