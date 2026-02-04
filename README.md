# ChainRank

A trust scoring system for Ethereum addresses based on on-chain activity. ChainRank analyzes blockchain events from DeFi protocols to calculate reputation scores using the EigenTrust algorithm.

## Overview

ChainRank fetches events from major DeFi protocols (Aave, Compound, Uniswap, Lido, etc.) and calculates trust scores based on user interactions. The system supports multiple chains (Ethereum, Base) and incorporates identity verification through ENS and Farcaster.

### Key Features

- **Multi-protocol support**: Lending (Aave, Compound, Morpho, Spark), DEXs (Uniswap, Aerodrome, 1inch, CoW), Liquid Staking (Lido, Rocket Pool, EtherFi), and more
- **Multi-chain**: Ethereum and Base
- **Identity integration**: ENS names and Farcaster verification multipliers
- **Configurable weights**: Customize trust weights per event type
- **EigenTrust algorithm**: Calculate global trust scores from local trust relationships

## Installation

1. Clone the repository and navigate to the project directory

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your RPC endpoints:
   ```
   RPC_ETHEREUM=https://eth-mainnet.g.alchemy.com/v2/YOUR_API_KEY
   RPC_BASE=https://base-mainnet.g.alchemy.com/v2/YOUR_API_KEY
   ```

## Usage

### 1. Generate Seed Scores

Generate initial seed scores for trusted contracts:

```bash
python generate_seeds.py --config config.toml
python generate_seeds.py --config config.toml --chain ethereum
```

### 2. Fetch On-Chain Events

Download events from configured protocols:

```bash
python fetch_events.py --config config.toml
python fetch_events.py --config config.toml --year 2025 --month 1
```

### 3. Calculate Trust Scores

Compute trust scores from the fetched events:

```bash
python calculate_trust.py --config config.toml
python calculate_trust.py --config config.toml --input raw/ethereum_aave_20240101_120000.csv
```

### 4. Fetch ENS Names (Optional)

Resolve ENS names for scored addresses:

```bash
python fetch_ens_names.py --config config.toml
```

### 5. Map Identities

Map addresses to ENS/Farcaster identities:

```bash
python map_identities.py --scores scores/ethereum.csv --output scores/ethereum_mapped.csv
```

## Configuration

The `config.toml` file contains:

- **time_periods**: Define the time range for event fetching
- **seed_contracts**: Trusted protocol contracts by chain
- **allowlisted_events**: Event signatures to track per protocol
- **trust_weights**: Scoring weights for different event types
- **output**: Output folder and file format settings
- **indexer**: Rate limiting and batch processing settings

### Trust Weights

Events are weighted based on their trust implications:

| Category | Positive Weight | Negative Weight |
|----------|----------------|-----------------|
| Supply/Deposit | +1.0 | - |
| Repay | +1.0 | - |
| Liquidator | +0.5 | - |
| Borrow | - | -0.5 |
| Withdraw | - | -0.5 |
| Liquidated | - | -1.0 |

### Trust Formula

```
trust = max(0, Σ log(amount) * event_weight)
```

Verification multipliers:
- ENS verified: 2x
- Farcaster verified: configurable
- World ID verified: configurable

## Project Structure

```
chainrank/
├── config.toml           # Main configuration
├── config.defi.toml      # DeFi-specific configuration
├── requirements.txt      # Python dependencies
├── generate_seeds.py     # Generate seed scores
├── fetch_events.py       # Fetch on-chain events
├── calculate_trust.py    # Calculate trust scores
├── fetch_ens_names.py    # Resolve ENS names
├── map_identities.py     # Map addresses to identities
├── find_seed_candidates.py # Find potential seed addresses
├── raw/                  # Raw event data
├── seed/                 # Seed score files
├── scores/               # Calculated trust scores
├── trust/                # Trust edge data
└── output/               # Final output files
```

## Supported Protocols

### Lending & Borrowing
- Aave V3
- Compound V2/V3
- Morpho Blue
- Spark
- Fluid

### Decentralized Exchanges
- Uniswap V3
- Aerodrome
- 1inch
- 0x
- Paraswap
- CoW Protocol

### Liquid Staking & Restaking
- Lido (stETH, wstETH)
- Rocket Pool
- EtherFi
- Frax ETH
- EigenLayer
- Renzo
- Kelp
- Puffer
- Origin (OETH)

### Yield
- Yearn Finance

## License

MIT