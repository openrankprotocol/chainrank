#!/usr/bin/env python3
"""
ChainRank Seed Candidate Finder
═══════════════════════════════════════════════════════════════════════════════
Finds known protocol contracts that:
1. Interact with existing seed contracts (send trust to them)
2. Also receive interactions from EOAs/wallets (have real users)

These are good candidates to add to the seed list.

Usage:
    python find_seed_candidates.py
    python find_seed_candidates.py --chain ethereum --top 10
"""

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

import toml
from dotenv import load_dotenv
from web3 import Web3

load_dotenv()


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


def get_seed_contracts(config: dict, chain: str) -> set[str]:
    """Get all seed contract addresses for a chain from config."""
    seed_contracts = set()
    for protocol_name, addresses in config.get("seed_contracts", {}).items():
        if chain in addresses:
            seed_contracts.add(addresses[chain].lower())
    return seed_contracts


def is_contract(w3: Web3, address: str) -> bool:
    """Check if an address is a contract."""
    try:
        checksum_addr = Web3.to_checksum_address(address)
        code = w3.eth.get_code(checksum_addr)
        return len(code) > 0
    except Exception:
        return False


# Well-known protocol contracts that would be good seed candidates
KNOWN_PROTOCOLS = {
    # DEX Routers
    "0x7a250d5630b4cf539739df2c5dacb4c659f2488d": {
        "name": "Uniswap V2 Router",
        "protocol": "uniswap_v2",
        "category": "DEX",
    },
    "0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45": {
        "name": "Uniswap V3 Router 2",
        "protocol": "uniswap_v3",
        "category": "DEX",
    },
    "0xe592427a0aece92de3edee1f18e0157c05861564": {
        "name": "Uniswap V3 Router",
        "protocol": "uniswap_v3",
        "category": "DEX",
    },
    "0xef1c6e67703c7bd7107eed8303fbe6ec2554bf6b": {
        "name": "Uniswap Universal Router",
        "protocol": "uniswap",
        "category": "DEX",
    },
    "0x3fc91a3afd70395cd496c647d5a6cc9d4b2b7fad": {
        "name": "Uniswap Universal Router 2",
        "protocol": "uniswap",
        "category": "DEX",
    },
    "0x1111111254fb6c44bac0bed2854e76f90643097d": {
        "name": "1inch V4 Router",
        "protocol": "1inch",
        "category": "DEX Aggregator",
    },
    "0x1111111254eeb25477b68fb85ed929f73a960582": {
        "name": "1inch V5 Router",
        "protocol": "1inch",
        "category": "DEX Aggregator",
    },
    "0x111111125421ca6dc452d289314280a0f8842a65": {
        "name": "1inch V6 Router",
        "protocol": "1inch",
        "category": "DEX Aggregator",
    },
    "0xdef1c0ded9bec7f1a1670819833240f027b25eff": {
        "name": "0x Exchange Proxy",
        "protocol": "0x",
        "category": "DEX Aggregator",
    },
    "0xd9e1ce17f2641f24ae83637ab66a2cca9c378b9f": {
        "name": "SushiSwap Router",
        "protocol": "sushiswap",
        "category": "DEX",
    },
    "0x881d40237659c251811cec9c364ef91dc08d300c": {
        "name": "MetaMask Swap Router",
        "protocol": "metamask",
        "category": "DEX Aggregator",
    },
    # Lending Protocols
    "0x7d2768de32b0b80b7a3454c06bdac94a69ddc7a9": {
        "name": "Aave V2 Pool",
        "protocol": "aave_v2",
        "category": "Lending",
    },
    "0x87870bca3f3fd6335c3f4ce8392d69350b4fa4e2": {
        "name": "Aave V3 Pool",
        "protocol": "aave_v3",
        "category": "Lending",
    },
    "0x398ec7346dcd622edc5ae82352f02be94c62d119": {
        "name": "Aave V1 Lending Pool",
        "protocol": "aave_v1",
        "category": "Lending",
    },
    "0x5d3a536e4d6dbd6114cc1ead35777bab948e3643": {
        "name": "Compound cDAI",
        "protocol": "compound",
        "category": "Lending",
    },
    "0x4ddc2d193948926d02f9b1fe9e1daa0718270ed5": {
        "name": "Compound cETH",
        "protocol": "compound",
        "category": "Lending",
    },
    "0x39aa39c021dfbae8fac545936693ac917d5e7563": {
        "name": "Compound cUSDC",
        "protocol": "compound",
        "category": "Lending",
    },
    "0xc3d688b66703497daa19211eedff47f25384cdc3": {
        "name": "Compound V3 cUSDCv3",
        "protocol": "compound_v3",
        "category": "Lending",
    },
    "0xa17581a9e3356d9a858b789d68b4d866e593ae94": {
        "name": "Compound V3 cWETHv3",
        "protocol": "compound_v3",
        "category": "Lending",
    },
    # Curve
    "0xbebc44782c7db0a1a60cb6fe97d0b483032ff1c7": {
        "name": "Curve 3pool",
        "protocol": "curve",
        "category": "DEX",
    },
    "0xd51a44d3fae010294c616388b506acda1bfaae46": {
        "name": "Curve Tricrypto2",
        "protocol": "curve",
        "category": "DEX",
    },
    "0x99a58482bd75cbab83b27ec03ca68ff489b5788f": {
        "name": "Curve Router",
        "protocol": "curve",
        "category": "DEX",
    },
    # Balancer
    "0xba12222222228d8ba445958a75a0704d566bf2c8": {
        "name": "Balancer Vault",
        "protocol": "balancer",
        "category": "DEX",
    },
    # Maker/DAI
    "0x9759a6ac90977b93b58547b4a71c78317f391a28": {
        "name": "MakerDAO DSR Manager",
        "protocol": "maker",
        "category": "Lending",
    },
    "0x83f20f44975d03b1b09e64809b757c47f942beea": {
        "name": "Spark sDAI",
        "protocol": "spark",
        "category": "Lending",
    },
    # Lido
    "0xae7ab96520de3a18e5e111b5eaab095312d7fe84": {
        "name": "Lido stETH",
        "protocol": "lido",
        "category": "Staking",
    },
    "0x7f39c581f595b53c5cb19bd0b3f8da6c935e2ca0": {
        "name": "Lido wstETH",
        "protocol": "lido",
        "category": "Staking",
    },
    "0x889edc2edab5f40e902b864ad4d7ade8e412f9b1": {
        "name": "Lido Withdrawal Queue",
        "protocol": "lido",
        "category": "Staking",
    },
    # Rocket Pool
    "0xae78736cd615f374d3085123a210448e74fc6393": {
        "name": "Rocket Pool rETH",
        "protocol": "rocketpool",
        "category": "Staking",
    },
    "0xdd3f50f8a6cafbe9b31a427582963f465e745af8": {
        "name": "Rocket Pool Deposit Pool",
        "protocol": "rocketpool",
        "category": "Staking",
    },
    # EigenLayer
    "0x858646372cc42e1a627fce94aa7a7033e7cf075a": {
        "name": "EigenLayer Strategy Manager",
        "protocol": "eigenlayer",
        "category": "Restaking",
    },
    "0x39053d51b77dc0d36036fc1fcc8cb819df8ef37a": {
        "name": "EigenLayer Delegation Manager",
        "protocol": "eigenlayer",
        "category": "Restaking",
    },
    # ether.fi
    "0x35fa164735182de50811e8e2e824cfb9b6118ac2": {
        "name": "ether.fi eETH",
        "protocol": "etherfi",
        "category": "Restaking",
    },
    "0xcd5fe23c85820f7b72d0926fc9b05b43e359b7ee": {
        "name": "ether.fi weETH",
        "protocol": "etherfi",
        "category": "Restaking",
    },
    "0x308861a430be4cce5502d0a12724771fc6daf216": {
        "name": "ether.fi Liquidity Pool",
        "protocol": "etherfi",
        "category": "Restaking",
    },
    "0x7d5706f6ef3f89b3951e23e557cdfbc3239d4e2c": {
        "name": "ether.fi Withdraw Request NFT",
        "protocol": "etherfi",
        "category": "Restaking",
    },
    # Convex
    "0xf403c135812408bfbe8713b5a23a04b3d48aae31": {
        "name": "Convex Booster",
        "protocol": "convex",
        "category": "Yield",
    },
    # Yearn
    "0xa354f35829ae975e850e23e9615b11da1b3dc4de": {
        "name": "Yearn yvUSDC",
        "protocol": "yearn",
        "category": "Yield",
    },
    # Frax
    "0x5e8422345238f34275888049021821e8e08caa1f": {
        "name": "Frax frxETH",
        "protocol": "frax",
        "category": "Staking",
    },
    "0xac3e018457b222d93114458476f3e3416abbe38f": {
        "name": "Frax sfrxETH",
        "protocol": "frax",
        "category": "Staking",
    },
    "0xbafa44efe7901e04e39dad13167d089c559c1138": {
        "name": "Frax ETH Minter",
        "protocol": "frax",
        "category": "Staking",
    },
    # Pendle
    "0x00000000005bbb0ef59571e58418f9a4357b68a0": {
        "name": "Pendle Router V3",
        "protocol": "pendle",
        "category": "Yield",
    },
    "0x888888888889758f76e7103c6cbf23abbf58f946": {
        "name": "Pendle Router V4",
        "protocol": "pendle",
        "category": "Yield",
    },
    # Renzo - Liquid Restaking
    "0x74a09653a083691711cf8215a6ab074bb4e99ef5": {
        "name": "Renzo RestakeManager",
        "protocol": "renzo",
        "category": "Restaking",
    },
    "0xbf5495efe5db9ce00f80364c8b423567e58d2110": {
        "name": "Renzo ezETH",
        "protocol": "renzo",
        "category": "Restaking",
    },
    # Kelp DAO - Multi-LST Restaking
    "0x036676389e48133b63a802f8635ad39e752d375d": {
        "name": "Kelp LRT Deposit Pool",
        "protocol": "kelp",
        "category": "Restaking",
    },
    "0xa1290d69c65a6fe4df752f95823fae25cb99e5a7": {
        "name": "Kelp rsETH",
        "protocol": "kelp",
        "category": "Restaking",
    },
    # Puffer Finance - Native Restaking
    "0xd9a442856c234a39a81a089c06451ebaa4306a72": {
        "name": "Puffer Vault / pufETH",
        "protocol": "puffer",
        "category": "Restaking",
    },
    # Origin OETH - LST Auto-Compounder
    "0x39254033945aa2e4809cc2977e7087bee48bd7ab": {
        "name": "Origin OETH Vault",
        "protocol": "origin",
        "category": "Yield",
    },
    "0x856c4efb76c1d1ae02e20ceb03a2a6a08b0b8dc3": {
        "name": "Origin OETH",
        "protocol": "origin",
        "category": "Yield",
    },
    # Morpho
    "0xbbbbbbbbbb9cc5e90e3b3af64bdaf62c37eeffcb": {
        "name": "Morpho Blue",
        "protocol": "morpho",
        "category": "Lending",
    },
    # EigenLayer
    "0x858646372cc42e1a627fce94aa7a7033e7cf075a": {
        "name": "EigenLayer Strategy Manager",
        "protocol": "eigenlayer",
        "category": "Restaking",
    },
    # Stablecoins
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": {
        "name": "USDC",
        "protocol": "usdc",
        "category": "Stablecoin",
    },
    "0xdac17f958d2ee523a2206206994597c13d831ec7": {
        "name": "USDT",
        "protocol": "usdt",
        "category": "Stablecoin",
    },
    "0x6b175474e89094c44da98b954eedeac495271d0f": {
        "name": "DAI",
        "protocol": "dai",
        "category": "Stablecoin",
    },
    # Wrapped tokens
    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": {
        "name": "WETH",
        "protocol": "weth",
        "category": "Wrapped",
    },
    "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599": {
        "name": "WBTC",
        "protocol": "wbtc",
        "category": "Wrapped",
    },
}


def load_trust_edges(trust_path: Path) -> list[tuple[str, str, float]]:
    """Load all trust edges from CSV file."""
    edges = []
    with open(trust_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            from_addr = row.get("i", "").lower()
            to_addr = row.get("j", "").lower()
            trust = float(row.get("v", 0))
            edges.append((from_addr, to_addr, trust))
    return edges


def main():
    parser = argparse.ArgumentParser(description="ChainRank Seed Candidate Finder")
    parser.add_argument(
        "--config", type=str, default="config.toml", help="Path to config.toml file"
    )
    parser.add_argument(
        "--chain",
        type=str,
        default="ethereum",
        help="Chain to analyze (default: ethereum)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of top candidates to show (default: 5)",
    )

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1

    config = load_config(str(config_path))

    print("═" * 79)
    print("ChainRank Seed Candidate Finder")
    print("═" * 79)

    # Setup
    chain = args.chain.lower()
    trust_folder = Path(config.get("trust_output", {}).get("trust_folder", "trust"))
    trust_path = trust_folder / f"{chain}.csv"

    if not trust_path.exists():
        print(f"Error: Trust file not found: {trust_path}")
        return 1

    # Get current seed contracts
    seed_contracts = get_seed_contracts(config, chain)
    print(f"Current seed contracts: {len(seed_contracts)}")
    for addr in seed_contracts:
        info = KNOWN_PROTOCOLS.get(addr, {})
        name = info.get("name", "Unknown")
        print(f"  - {addr} ({name})")
    print()

    # Connect to chain
    print("─" * 79)
    print("Connecting to Chain")
    print("─" * 79)

    try:
        rpc_url = get_rpc_endpoint(chain)
        w3 = Web3(Web3.HTTPProvider(rpc_url))
        if not w3.is_connected():
            print(f"Error: Failed to connect to {chain}")
            return 1
        print(f"  ✓ Connected to {chain}")
    except Exception as e:
        print(f"Error connecting to {chain}: {e}")
        return 1

    print()

    # Load trust edges
    print("─" * 79)
    print("Loading Trust Data")
    print("─" * 79)

    edges = load_trust_edges(trust_path)
    print(f"  Total edges: {len(edges):,}")

    # Find:
    # 1. Contracts that send trust TO seed contracts
    # 2. Filter to known protocols
    # 3. Check that they also receive trust from EOAs

    # Trust sent TO seeds (from -> seed)
    trust_to_seeds = defaultdict(float)
    for from_addr, to_addr, trust in edges:
        if to_addr in seed_contracts and from_addr not in seed_contracts:
            trust_to_seeds[from_addr] += trust

    print(f"  Addresses sending trust to seeds: {len(trust_to_seeds):,}")

    # Trust received FROM others (other -> addr)
    trust_received = defaultdict(float)
    trust_senders = defaultdict(set)
    for from_addr, to_addr, trust in edges:
        trust_received[to_addr] += trust
        trust_senders[to_addr].add(from_addr)

    print()

    # Find known protocols that interact with seeds
    print("─" * 79)
    print("Finding Known Protocol Candidates")
    print("─" * 79)

    candidates = []

    for addr, trust_given in trust_to_seeds.items():
        # Skip if already a seed
        if addr in seed_contracts:
            continue

        # Check if it's a known protocol
        if addr not in KNOWN_PROTOCOLS:
            continue

        protocol_info = KNOWN_PROTOCOLS[addr]

        # Check how many unique senders (potential EOAs/users)
        senders = trust_senders.get(addr, set())
        trust_recv = trust_received.get(addr, 0)

        # Count EOA senders (check a sample)
        eoa_count = 0
        sample_size = min(50, len(senders))
        sampled_senders = list(senders)[:sample_size]

        for sender in sampled_senders:
            if not is_contract(w3, sender):
                eoa_count += 1

        # Estimate total EOAs
        if sample_size > 0:
            eoa_ratio = eoa_count / sample_size
            estimated_eoas = int(len(senders) * eoa_ratio)
        else:
            estimated_eoas = 0

        candidates.append(
            {
                "address": addr,
                "name": protocol_info["name"],
                "protocol": protocol_info["protocol"],
                "category": protocol_info["category"],
                "trust_to_seeds": trust_given,
                "trust_received": trust_recv,
                "unique_senders": len(senders),
                "estimated_eoas": estimated_eoas,
                "eoa_ratio": eoa_ratio if sample_size > 0 else 0,
            }
        )

    # Sort by combination of trust to seeds and EOA users
    # Score = trust_to_seeds * log(1 + estimated_eoas)
    import math

    for c in candidates:
        c["score"] = c["trust_to_seeds"] * math.log2(1 + c["estimated_eoas"])

    candidates.sort(key=lambda x: -x["score"])

    print(f"  Found {len(candidates)} known protocol candidates")
    print()

    # Display top candidates
    print("─" * 79)
    print(f"Top {args.top} Seed Candidates")
    print("─" * 79)
    print()

    for i, c in enumerate(candidates[: args.top], 1):
        print(f"{i}. {c['name']} ({c['category']})")
        print(f"   Address: {c['address']}")
        print(f"   Protocol: {c['protocol']}")
        print(f"   Trust to seeds: {c['trust_to_seeds']:,.2f}")
        print(f"   Trust received: {c['trust_received']:,.2f}")
        print(f"   Unique senders: {c['unique_senders']:,}")
        print(f"   Estimated EOA users: {c['estimated_eoas']:,} ({c['eoa_ratio']:.0%})")
        print(f"   Score: {c['score']:,.2f}")
        print()

    # Generate config snippet
    print("─" * 79)
    print("Config Snippet (add to config.toml)")
    print("─" * 79)
    print()

    for c in candidates[: args.top]:
        protocol = c["protocol"].replace("_", "")
        print(f"[seed_contracts.{protocol}]")
        print(f'{chain} = "{c["address"]}"')
        print()

    print("═" * 79)
    print("✓ Done!")
    print("═" * 79)

    return 0


if __name__ == "__main__":
    exit(main())
