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
import gc
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    # Compound V2 cToken Events
    # Mint(address minter, uint256 mintAmount, uint256 mintTokens)
    "Mint": "0x4c209b5fc8ad50758f13e2e1088ba56a560dff690a1c6fef26394f4c03821c4f",
    # Redeem(address redeemer, uint256 redeemAmount, uint256 redeemTokens)
    "Redeem": "0xe5b754fb1abb7f01b499791d0b820ae3b6af3424ac1c59768edb53f4ec31a929",
    # Borrow(address borrower, uint256 borrowAmount, uint256 accountBorrows, uint256 totalBorrows)
    "CompoundBorrow": "0x13ed6866d4e1ee6da46f845c46d7e54120883d75c5ea9a2dacc1c4ca8984ab80",
    # RepayBorrow(address payer, address borrower, uint256 repayAmount, uint256 accountBorrows, uint256 totalBorrows)
    "RepayBorrow": "0x1a2a22cb034d26d1854bdc6666a5b91fe25efbbb5dcad3b0355478d6f5c362a1",
    # LiquidateBorrow(address liquidator, address borrower, uint256 repayAmount, address cTokenCollateral, uint256 seizeTokens)
    "LiquidateBorrow": "0x298637f684da70674f26509b10f07ec2fbc77a335ab1e7d6215a4b2484d8bb52",
    # Compound Comptroller Events
    # DistributedSupplierComp(address indexed cToken, address indexed supplier, uint256 compDelta, uint256 compSupplyIndex)
    "DistributedSupplierComp": "0x2caecd17d02f56fa897705dcc740da2d237c373f70686f4e0d9bd3bf0400ea7a",
    # DistributedBorrowerComp(address indexed cToken, address indexed borrower, uint256 compDelta, uint256 compBorrowIndex)
    "DistributedBorrowerComp": "0x1fc3ecc087d8d2d15e23d0032af5a47571a93e0005cf5a61031e3d5c6d40b185",
    # Morpho Blue Events
    # Supply(bytes32 indexed id, address indexed caller, address indexed onBehalf, uint256 assets, uint256 shares)
    "MorphoSupply": "0xedf8870433c83823eb071d3df1caa8d008f12f6440918c20d75a3602cda30fe0",
    # Withdraw(bytes32 indexed id, address caller, address indexed onBehalf, address indexed receiver, uint256 assets, uint256 shares)
    "MorphoWithdraw": "0xa56fc0ad5702ec05ce63666221f796fb62437c32db1aa1aa075fc6484cf58fbf",
    # Borrow(bytes32 indexed id, address caller, address indexed onBehalf, address indexed receiver, uint256 assets, uint256 shares)
    "MorphoBorrow": "0x570954540bed6b1304a87dfe815a5eda4a648f7097a16240dcd85c9b5fd42a43",
    # Repay(bytes32 indexed id, address indexed caller, address indexed onBehalf, uint256 assets, uint256 shares)
    "MorphoRepay": "0x52acb05cebbd3cd39715469f22afbf5a17496295ef3bc9bb5944056c63ccaa09",
    # Liquidate(bytes32 indexed id, address indexed caller, address indexed borrower, uint256 repaidAssets, uint256 repaidShares, uint256 seizedAssets, uint256 badDebtAssets, uint256 badDebtShares)
    "MorphoLiquidate": "0xa4946ede45d0c6f06a0f5ce92c9ad3b4751452571a9f579dcf878f1c8e1cd892",
    # Fluid Liquidity Events
    # LogOperate(address indexed user, address indexed token, int256 supplyAmount, int256 borrowAmount, address withdrawTo, address borrowTo, uint256 totalAmounts, uint256 exchangePricesAndConfig)
    "FluidOperate": "0x4d93b232a24e82b284ced7461bf4deacffe66759d5c24513e6f29e571ad78d15",
    # DEX Aggregator Events
    # 1inch v5: OrderFilled(address indexed maker, bytes32 orderHash, uint256 remainingAmount)
    "OneInchSwapped": "0xb9ed0243fdf00f0545c63a0af8850c090d86bb46682baec4bf3c496814fe4f02",
    # 0x Exchange: TransformedERC20(address indexed taker, address inputToken, address outputToken, uint256 inputTokenAmount, uint256 outputTokenAmount)
    "ZeroExTransformedERC20": "0x0f6672f78a59ba8e5e5b5d38df3ebc67f3c792e2c9259b8d97d7f00dd78ba1b3",
    # Paraswap v5: SwappedV3(bytes16 uuid, address partner, uint256 feePercent, address initiator, address indexed beneficiary, address indexed srcToken, address indexed destToken, uint256 srcAmount, uint256 receivedAmount, uint256 expectedAmount)
    "ParaswapSwapped": "0xe00361d207b252a464323eb23d45d42583e391f2031acdd2e9fa36efddd43cb0",
    # CoW Protocol: Trade(address indexed owner, address sellToken, address buyToken, uint256 sellAmount, uint256 buyAmount, uint256 feeAmount, bytes orderUid)
    "CowTrade": "0xa07a543ab8a018198e99ca0184c93fe9050a79400a0a723441f84de1d972cc17",
    # Yearn v3 Vault Events
    # Deposit(address indexed sender, address indexed owner, uint256 assets, uint256 shares)
    "YearnDeposit": "0xdcbc1c05240f31ff3ad067ef1ee35ce4997762752e3a095284754544f4c709d7",
    # Withdraw(address indexed sender, address indexed receiver, address indexed owner, uint256 assets, uint256 shares)
    "YearnWithdraw": "0xfbde797d201c681b91056529119e0b02407c7bb96a4a2c75c01fc9667232c8db",
    # ERC-20 Events
    "Transfer": "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef",
    "Approval": "0x8c5be1e5ebec7d5bd14f71427d1e84f3dd0314c0f7b2291e5b200ac8c7c3b925",
    # Uniswap V3 Pool Events
    # Swap(address indexed sender, address indexed recipient, int256 amount0, int256 amount1, uint160 sqrtPriceX96, uint128 liquidity, int24 tick)
    "UniswapV3Swap": "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67",
    # ═══════════════════════════════════════════════════════════════════════════
    # STAKING PROTOCOL EVENTS
    # ═══════════════════════════════════════════════════════════════════════════
    # Lido stETH Events
    # Submitted(address indexed sender, uint256 amount, address referral)
    "LidoSubmitted": "0x96a25c8ce0baabc1fdefd93e9ed25d8e092a3332f3aa9a41722b5697231d1d1a",
    # Lido Withdrawal Queue Events
    # WithdrawalRequested(uint256,address,address,uint256,uint256)
    "LidoWithdrawalRequested": "0xf0cb471f23fb74ea44b8252eb1881a2dca546288d9f6e90d1a0e82fe0ed342ab",
    # WithdrawalClaimed(uint256,address,address,uint256)
    "LidoWithdrawalClaimed": "0x6ad26c5e238e7d002799f9a5db07e81ef14e37386ae03496d7a7ef04713e145b",
    # EigenLayer Strategy Manager Events
    # Deposit(address,address,uint256) - staker, token, shares (3 params, not 4!)
    "EigenDeposit": "0x5548c837ab068cf56a2c2479df0882a4922fd203edb7517321831d95078c5f62",
    # WithdrawalQueued(bytes32,address) - withdrawalRoot, depositor
    "EigenWithdrawalQueued": "0xdab4a053ac3d59b4581b9aff6e51b4f37bf2b19ef0438178968133590aade726",
    # WithdrawalCompleted(bytes32)
    "EigenWithdrawalCompleted": "0xc97098c2f658800b4df29001527f7324bcdffcf6e8751a699ab920a1eced5b1d",
    # EigenLayer Delegation Manager Events
    # OperatorSharesIncreased(address,address,address,uint256) - most common delegation event
    "EigenOperatorSharesIncreased": "0x1ec042c965e2edd7107b51188ee0f383e22e76179041ab3a9d18ff151405166c",
    # OperatorSharesDecreased(address,address,address,uint256)
    "EigenOperatorSharesDecreased": "0x6909600037b75d7b4733aedd815442b5ec018a827751c832aaff64eba5d6d2dd",
    # StakerDelegated(address,address)
    "EigenStakerDelegated": "0xc3ee9f2e5fda98e8066a1f745b2df9285f416fe98cf2559cd21484b3d8743304",
    # StakerUndelegated(address,address)
    "EigenStakerUndelegated": "0xfee30966a256b71e14bc0ebfc94315e28ef4a97a7131a9e2b7a310a73af44676",
    # Rocket Pool Events
    # DepositReceived(address,uint256,uint256)
    "RocketDepositReceived": "0x7aa1a8eb998c779420645fc14513bf058edb347d95c2fc2e6845bdc22f888631",
    # TokensMinted(address indexed to, uint256 amount, uint256 ethAmount, uint256 time)
    "RocketTokensMinted": "0x6155cfd0fd028b0ca77e8495a60cbe563e8bce8611f0aad6fedbdaafc05d44a2",
    # TokensBurned(address indexed from, uint256 amount, uint256 ethAmount, uint256 time)
    "RocketTokensBurned": "0x19783b34589160c168487dc7f9c51ae0bcefe67a47d6708fba90f6ce0366d3d1",
    # ether.fi Events
    # Deposit(address,uint256,uint8,address) - depositor, amount, sourceOfFunds, referral
    "EtherFiDeposit": "0xa241faf62e66ce518d1934ce4c936d806a02289ba483fac23beb8c15755be90d",
    # Withdraw event - b9da3f3df62c28aca604806cc6ee9678189d7591ef511a77bb040fa8361e9e02
    "EtherFiWithdraw": "0xb9da3f3df62c28aca604806cc6ee9678189d7591ef511a77bb040fa8361e9e02",
    # Frax Finance Events
    # ETHSubmitted(address,address,uint256,uint256)
    "FraxETHSubmitted": "0x29b3e86ecfd94a32218997c40b051e650e4fd8c97fc7a4d266be3f7c61c5205b",
    # Frax sfrxETH uses ERC4626 Deposit/Withdraw (same as Yearn)
    "FraxDeposit": "0xdcbc1c05240f31ff3ad067ef1ee35ce4997762752e3a095284754544f4c709d7",
    "FraxWithdraw": "0xfbde797d201c681b91056529119e0b02407c7bb96a4a2c75c01fc9667232c8db",
    # ═══════════════════════════════════════════════════════════════════════════
    # STAKING AGGREGATOR EVENTS
    # ═══════════════════════════════════════════════════════════════════════════
    # Renzo Events
    # Deposit(address indexed depositor, address indexed token, uint256 amount, uint256 ezETHMinted, uint256 timestamp)
    "RenzoDeposit": "0x4e2ca0515ed1aef1395f66b5303bb5d6f1bf9d61a353fa53f73f8ac9973fa9f6",
    # UserWithdrawStarted(bytes32,address,address,uint256,uint256)
    "RenzoWithdraw": "0xc09ed86575d29e28a57c7404cd6a8633bf722b616f11301ff62dc1c059804a7b",
    # Kelp DAO Events
    # AssetDeposit(address,address,uint256,uint256)
    "KelpDeposit": "0xa1fe1983016c44964fc77ca865d02be0f16896800286c569d64467923ca8ffef",
    # AssetWithdraw(address,address,uint256)
    "KelpWithdraw": "0x3d32c53b74e7c44a5e4e1caa6ae931adab7c39bf958730ca47f761ac5ad0ac55",
    # Puffer Finance Events - uses ERC4626 Deposit/Withdraw
    # Deposit(address,address,uint256,uint256)
    "PufferDeposit": "0xdcbc1c05240f31ff3ad067ef1ee35ce4997762752e3a095284754544f4c709d7",
    # Withdraw(address,address,address,uint256,uint256)
    "PufferWithdraw": "0xfbde797d201c681b91056529119e0b02407c7bb96a4a2c75c01fc9667232c8db",
    # Pendle Events
    # Swap(address,address,address,int256,int256)
    "PendleSwap": "0xf0af0459879c8c3eebaefd078b77045d687a15e5a35d91114d6a57067758dc89",
    # MintPY - use Transfer as fallback
    "PendleMint": "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef",
    # RedeemPY - use Transfer as fallback
    "PendleRedeem": "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef",
    # Origin OETH Events
    # Deposit(address indexed _asset, address indexed _account, uint256 _amount)
    "OriginDeposit": "0x5548c837ab068cf56a2c2479df0882a4922fd203edb7517321831d95078c5f62",
    # Withdraw(address,address,uint256)
    "OriginWithdraw": "0x9b1bfa7fa9ee420a16e124f794c35ac9f90472acc99140eb2f6447c714cad8eb",
    # Mint(address indexed _addr, uint256 _value)
    "OriginMint": "0x0f6798a560793a54c3bcfe86a93cde1e73087d944c0ea20544137d4121396885",
    # Redeem(address indexed _addr, uint256 _value)
    "OriginRedeem": "0x222838db2794d11532d940e8dec38ae307ed0b63cd97c233322e221f998767a6",
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

    # Compound V2 cToken Events
    elif event_name == "Mint":
        # Mint(address minter, uint256 mintAmount, uint256 mintTokens)
        data_hex = data.hex() if isinstance(data, bytes) else data
        if len(data_hex) >= 194:  # 0x + 3*64 chars
            decoded["minter"] = "0x" + data_hex[26:66]  # address is padded
            decoded["mint_amount"] = int(data_hex[66:130], 16)
            decoded["mint_tokens"] = int(data_hex[130:194], 16)

    elif event_name == "Redeem":
        # Redeem(address redeemer, uint256 redeemAmount, uint256 redeemTokens)
        data_hex = data.hex() if isinstance(data, bytes) else data
        if len(data_hex) >= 194:
            decoded["redeemer"] = "0x" + data_hex[26:66]
            decoded["redeem_amount"] = int(data_hex[66:130], 16)
            decoded["redeem_tokens"] = int(data_hex[130:194], 16)

    elif event_name == "CompoundBorrow" or event_name == "Borrow":
        # For Compound: Borrow(address borrower, uint256 borrowAmount, uint256 accountBorrows, uint256 totalBorrows)
        # Check if this looks like Compound format (no indexed params, all in data)
        data_hex = data.hex() if isinstance(data, bytes) else data
        if (
            len(data_hex) >= 258 and len(topics) == 1
        ):  # Compound style - no indexed params
            decoded["borrower"] = "0x" + data_hex[26:66]
            decoded["borrow_amount"] = int(data_hex[66:130], 16)
            decoded["account_borrows"] = int(data_hex[130:194], 16)
            decoded["total_borrows"] = int(data_hex[194:258], 16)

    elif event_name == "RepayBorrow":
        # RepayBorrow(address payer, address borrower, uint256 repayAmount, uint256 accountBorrows, uint256 totalBorrows)
        data_hex = data.hex() if isinstance(data, bytes) else data
        if len(data_hex) >= 322:
            decoded["payer"] = "0x" + data_hex[26:66]
            decoded["borrower"] = "0x" + data_hex[90:130]
            decoded["repay_amount"] = int(data_hex[130:194], 16)
            decoded["account_borrows"] = int(data_hex[194:258], 16)
            decoded["total_borrows"] = int(data_hex[258:322], 16)

    elif event_name == "LiquidateBorrow":
        # LiquidateBorrow(address liquidator, address borrower, uint256 repayAmount, address cTokenCollateral, uint256 seizeTokens)
        data_hex = data.hex() if isinstance(data, bytes) else data
        if len(data_hex) >= 322:
            decoded["liquidator"] = "0x" + data_hex[26:66]
            decoded["borrower"] = "0x" + data_hex[90:130]
            decoded["repay_amount"] = int(data_hex[130:194], 16)
            decoded["ctoken_collateral"] = "0x" + data_hex[218:258]
            decoded["seize_tokens"] = int(data_hex[258:322], 16)

    elif event_name == "DistributedSupplierComp" and len(topics) >= 3:
        # DistributedSupplierComp(address indexed cToken, address indexed supplier, uint256 compDelta, uint256 compSupplyIndex)
        decoded["ctoken"] = _extract_address(topics[1])
        decoded["supplier"] = _extract_address(topics[2])
        data_hex = data.hex() if isinstance(data, bytes) else data
        if len(data_hex) >= 130:
            decoded["comp_delta"] = int(data_hex[2:66], 16)
            decoded["comp_supply_index"] = int(data_hex[66:130], 16)

    elif event_name == "DistributedBorrowerComp" and len(topics) >= 3:
        # DistributedBorrowerComp(address indexed cToken, address indexed borrower, uint256 compDelta, uint256 compBorrowIndex)
        decoded["ctoken"] = _extract_address(topics[1])
        decoded["borrower"] = _extract_address(topics[2])
        data_hex = data.hex() if isinstance(data, bytes) else data
        if len(data_hex) >= 130:
            decoded["comp_delta"] = int(data_hex[2:66], 16)
            decoded["comp_borrow_index"] = int(data_hex[66:130], 16)

    # Morpho Blue Events
    elif event_name == "MorphoSupply" and len(topics) >= 4:
        # Supply(bytes32 indexed id, address indexed caller, address indexed onBehalf, uint256 assets, uint256 shares)
        decoded["market_id"] = (
            topics[1].hex() if isinstance(topics[1], bytes) else topics[1]
        )
        decoded["caller"] = _extract_address(topics[2])
        decoded["on_behalf_of"] = _extract_address(topics[3])
        data_hex = data.hex() if isinstance(data, bytes) else data
        if len(data_hex) >= 130:
            decoded["assets"] = int(data_hex[2:66], 16)
            decoded["shares"] = int(data_hex[66:130], 16)

    elif event_name == "MorphoWithdraw" and len(topics) >= 4:
        # Withdraw(bytes32 indexed id, address caller, address indexed onBehalf, address indexed receiver, uint256 assets, uint256 shares)
        decoded["market_id"] = (
            topics[1].hex() if isinstance(topics[1], bytes) else topics[1]
        )
        decoded["on_behalf_of"] = _extract_address(topics[2])
        decoded["receiver"] = _extract_address(topics[3])
        data_hex = data.hex() if isinstance(data, bytes) else data
        if len(data_hex) >= 194:
            decoded["caller"] = "0x" + data_hex[26:66]
            decoded["assets"] = int(data_hex[66:130], 16)
            decoded["shares"] = int(data_hex[130:194], 16)

    elif event_name == "MorphoBorrow" and len(topics) >= 4:
        # Borrow(bytes32 indexed id, address caller, address indexed onBehalf, address indexed receiver, uint256 assets, uint256 shares)
        decoded["market_id"] = (
            topics[1].hex() if isinstance(topics[1], bytes) else topics[1]
        )
        decoded["on_behalf_of"] = _extract_address(topics[2])
        decoded["receiver"] = _extract_address(topics[3])
        data_hex = data.hex() if isinstance(data, bytes) else data
        if len(data_hex) >= 194:
            decoded["caller"] = "0x" + data_hex[26:66]
            decoded["assets"] = int(data_hex[66:130], 16)
            decoded["shares"] = int(data_hex[130:194], 16)

    elif event_name == "MorphoRepay" and len(topics) >= 4:
        # Repay(bytes32 indexed id, address indexed caller, address indexed onBehalf, uint256 assets, uint256 shares)
        decoded["market_id"] = (
            topics[1].hex() if isinstance(topics[1], bytes) else topics[1]
        )
        decoded["caller"] = _extract_address(topics[2])
        decoded["on_behalf_of"] = _extract_address(topics[3])
        data_hex = data.hex() if isinstance(data, bytes) else data
        if len(data_hex) >= 130:
            decoded["assets"] = int(data_hex[2:66], 16)
            decoded["shares"] = int(data_hex[66:130], 16)

    elif event_name == "MorphoLiquidate" and len(topics) >= 4:
        # Liquidate(bytes32 indexed id, address indexed caller, address indexed borrower, uint256 repaidAssets, uint256 repaidShares, uint256 seizedAssets, uint256 badDebtAssets, uint256 badDebtShares)
        decoded["market_id"] = (
            topics[1].hex() if isinstance(topics[1], bytes) else topics[1]
        )
        decoded["liquidator"] = _extract_address(topics[2])
        decoded["borrower"] = _extract_address(topics[3])
        data_hex = data.hex() if isinstance(data, bytes) else data
        if len(data_hex) >= 322:
            decoded["repaid_assets"] = int(data_hex[2:66], 16)
            decoded["repaid_shares"] = int(data_hex[66:130], 16)
            decoded["seized_assets"] = int(data_hex[130:194], 16)
            decoded["bad_debt_assets"] = int(data_hex[194:258], 16)
            decoded["bad_debt_shares"] = int(data_hex[258:322], 16)

    # Fluid Liquidity Events
    elif event_name == "FluidOperate" and len(topics) >= 3:
        # Operate(address indexed user, address indexed token, int256 supplyAmount, int256 borrowAmount, address withdrawTo, address borrowTo, uint256 totalAmounts, uint256 exchangePricesAndConfig)
        decoded["user"] = _extract_address(topics[1])
        decoded["token"] = _extract_address(topics[2])
        data_hex = data.hex() if isinstance(data, bytes) else data
        if len(data_hex) >= 386:
            # supplyAmount and borrowAmount are signed int256
            supply_hex = data_hex[2:66]
            borrow_hex = data_hex[66:130]
            decoded["supply_amount"] = (
                int(supply_hex, 16)
                if int(supply_hex, 16) < 2**255
                else int(supply_hex, 16) - 2**256
            )
            decoded["borrow_amount"] = (
                int(borrow_hex, 16)
                if int(borrow_hex, 16) < 2**255
                else int(borrow_hex, 16) - 2**256
            )
            decoded["withdraw_to"] = "0x" + data_hex[154:194]
            decoded["borrow_to"] = "0x" + data_hex[218:258]
            decoded["total_amounts"] = int(data_hex[258:322], 16)
            decoded["exchange_prices_and_config"] = int(data_hex[322:386], 16)

    # DEX Aggregator Events
    elif event_name == "OneInchSwapped" and len(topics) >= 2:
        # OrderFilled(address indexed maker, bytes32 orderHash, uint256 remainingAmount)
        decoded["maker"] = _extract_address(topics[1])
        data_hex = data.hex() if isinstance(data, bytes) else data
        if len(data_hex) >= 130:
            decoded["order_hash"] = "0x" + data_hex[2:66]
            decoded["remaining_amount"] = int(data_hex[66:130], 16)

    elif event_name == "ZeroExTransformedERC20" and len(topics) >= 2:
        # TransformedERC20(address indexed taker, address inputToken, address outputToken, uint256 inputTokenAmount, uint256 outputTokenAmount)
        decoded["taker"] = _extract_address(topics[1])
        data_hex = data.hex() if isinstance(data, bytes) else data
        if len(data_hex) >= 258:
            decoded["input_token"] = "0x" + data_hex[26:66]
            decoded["output_token"] = "0x" + data_hex[90:130]
            decoded["input_amount"] = int(data_hex[130:194], 16)
            decoded["output_amount"] = int(data_hex[194:258], 16)

    elif event_name == "ParaswapSwapped" and len(topics) >= 4:
        # SwappedV3(bytes16 uuid, address partner, uint256 feePercent, address initiator, address indexed beneficiary, address indexed srcToken, address indexed destToken, uint256 srcAmount, uint256 receivedAmount, uint256 expectedAmount)
        decoded["beneficiary"] = _extract_address(topics[1])
        decoded["src_token"] = _extract_address(topics[2])
        decoded["dest_token"] = _extract_address(topics[3])
        data_hex = data.hex() if isinstance(data, bytes) else data
        if len(data_hex) >= 450:
            decoded["uuid"] = "0x" + data_hex[2:34]
            decoded["partner"] = "0x" + data_hex[58:98]
            decoded["fee_percent"] = int(data_hex[98:162], 16)
            decoded["initiator"] = "0x" + data_hex[186:226]
            decoded["src_amount"] = int(data_hex[226:290], 16)
            decoded["received_amount"] = int(data_hex[290:354], 16)
            decoded["expected_amount"] = int(data_hex[354:418], 16)

    elif event_name == "CowTrade" and len(topics) >= 2:
        # Trade(address indexed owner, address sellToken, address buyToken, uint256 sellAmount, uint256 buyAmount, uint256 feeAmount, bytes orderUid)
        decoded["owner"] = _extract_address(topics[1])
        data_hex = data.hex() if isinstance(data, bytes) else data
        if len(data_hex) >= 322:
            decoded["sell_token"] = "0x" + data_hex[26:66]
            decoded["buy_token"] = "0x" + data_hex[90:130]
            decoded["sell_amount"] = int(data_hex[130:194], 16)
            decoded["buy_amount"] = int(data_hex[194:258], 16)
            decoded["fee_amount"] = int(data_hex[258:322], 16)

    # Yearn v3 Vault Events
    elif event_name == "YearnDeposit" and len(topics) >= 3:
        # Deposit(address indexed sender, address indexed owner, uint256 assets, uint256 shares)
        decoded["sender"] = _extract_address(topics[1])
        decoded["owner"] = _extract_address(topics[2])
        data_hex = data.hex() if isinstance(data, bytes) else data
        if len(data_hex) >= 130:
            decoded["assets"] = int(data_hex[2:66], 16)
            decoded["shares"] = int(data_hex[66:130], 16)

    elif event_name == "YearnWithdraw" and len(topics) >= 4:
        # Withdraw(address indexed sender, address indexed receiver, address indexed owner, uint256 assets, uint256 shares)
        decoded["sender"] = _extract_address(topics[1])
        decoded["receiver"] = _extract_address(topics[2])
        decoded["owner"] = _extract_address(topics[3])
        data_hex = data.hex() if isinstance(data, bytes) else data
        if len(data_hex) >= 130:
            decoded["assets"] = int(data_hex[2:66], 16)
            decoded["shares"] = int(data_hex[66:130], 16)

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

    # ═══════════════════════════════════════════════════════════════════════════
    # STAKING PROTOCOL EVENTS
    # ═══════════════════════════════════════════════════════════════════════════

    # Lido stETH Events
    elif event_name == "LidoSubmitted" and len(topics) >= 2:
        # Submitted(address indexed sender, uint256 amount, address referral)
        decoded["sender"] = _extract_address(topics[1])
        data_hex = data.hex() if isinstance(data, bytes) else data
        if len(data_hex) >= 130:
            decoded["amount"] = int(data_hex[2:66], 16)
            decoded["referral"] = "0x" + data_hex[90:130]

    elif event_name == "LidoWithdrawalRequested" and len(topics) >= 4:
        # WithdrawalRequested(uint256 indexed requestId, address indexed requestor, address indexed owner, uint256 amountOfStETH, uint256 amountOfShares)
        decoded["request_id"] = int(
            topics[1].hex() if isinstance(topics[1], bytes) else topics[1], 16
        )
        decoded["requestor"] = _extract_address(topics[2])
        decoded["owner"] = _extract_address(topics[3])
        data_hex = data.hex() if isinstance(data, bytes) else data
        if len(data_hex) >= 130:
            decoded["amount_steth"] = int(data_hex[2:66], 16)
            decoded["amount_shares"] = int(data_hex[66:130], 16)

    elif event_name == "LidoWithdrawalClaimed" and len(topics) >= 4:
        # WithdrawalClaimed(uint256 indexed requestId, address indexed owner, address indexed receiver, uint256 amountOfETH)
        decoded["request_id"] = int(
            topics[1].hex() if isinstance(topics[1], bytes) else topics[1], 16
        )
        decoded["owner"] = _extract_address(topics[2])
        decoded["receiver"] = _extract_address(topics[3])
        data_hex = data.hex() if isinstance(data, bytes) else data
        if len(data_hex) >= 66:
            decoded["amount_eth"] = int(data_hex[2:66], 16)

    # EigenLayer Strategy Manager Events
    elif event_name == "EigenDeposit" and len(topics) >= 1:
        # Deposit(address,address,uint256) - staker, token, shares (all in data, not indexed)
        data_hex = data.hex() if isinstance(data, bytes) else data
        if len(data_hex) >= 194:
            decoded["staker"] = "0x" + data_hex[26:66]
            decoded["token"] = "0x" + data_hex[90:130]
            decoded["shares"] = int(data_hex[130:194], 16)

    # EigenLayer Delegation Manager Events
    elif event_name == "EigenOperatorSharesIncreased" and len(topics) >= 1:
        # OperatorSharesIncreased(address,address,address,uint256) - all in data
        data_hex = data.hex() if isinstance(data, bytes) else data
        if len(data_hex) >= 258:
            decoded["operator"] = "0x" + data_hex[26:66]
            decoded["staker"] = "0x" + data_hex[90:130]
            decoded["strategy"] = "0x" + data_hex[154:194]
            decoded["shares"] = int(data_hex[194:258], 16)

    elif event_name == "EigenOperatorSharesDecreased" and len(topics) >= 1:
        # OperatorSharesDecreased(address,address,address,uint256) - all in data
        data_hex = data.hex() if isinstance(data, bytes) else data
        if len(data_hex) >= 258:
            decoded["operator"] = "0x" + data_hex[26:66]
            decoded["staker"] = "0x" + data_hex[90:130]
            decoded["strategy"] = "0x" + data_hex[154:194]
            decoded["shares"] = int(data_hex[194:258], 16)

    elif event_name == "EigenStakerDelegated" and len(topics) >= 3:
        # StakerDelegated(address indexed staker, address indexed operator)
        decoded["staker"] = _extract_address(topics[1])
        decoded["operator"] = _extract_address(topics[2])

    elif event_name == "EigenStakerUndelegated" and len(topics) >= 3:
        # StakerUndelegated(address indexed staker, address indexed operator)
        decoded["staker"] = _extract_address(topics[1])
        decoded["operator"] = _extract_address(topics[2])

    # Rocket Pool Events
    elif event_name == "RocketDepositReceived" and len(topics) >= 2:
        # DepositReceived(address indexed from, uint256 amount, uint256 time)
        decoded["from"] = _extract_address(topics[1])
        data_hex = data.hex() if isinstance(data, bytes) else data
        if len(data_hex) >= 130:
            decoded["amount"] = int(data_hex[2:66], 16)
            decoded["time"] = int(data_hex[66:130], 16)

    elif event_name == "RocketTokensMinted" and len(topics) >= 2:
        # TokensMinted(address indexed to, uint256 amount, uint256 ethAmount, uint256 time)
        decoded["to"] = _extract_address(topics[1])
        data_hex = data.hex() if isinstance(data, bytes) else data
        if len(data_hex) >= 194:
            decoded["amount"] = int(data_hex[2:66], 16)
            decoded["eth_amount"] = int(data_hex[66:130], 16)
            decoded["time"] = int(data_hex[130:194], 16)

    elif event_name == "RocketTokensBurned" and len(topics) >= 2:
        # TokensBurned(address indexed from, uint256 amount, uint256 ethAmount, uint256 time)
        decoded["from"] = _extract_address(topics[1])
        data_hex = data.hex() if isinstance(data, bytes) else data
        if len(data_hex) >= 194:
            decoded["amount"] = int(data_hex[2:66], 16)
            decoded["eth_amount"] = int(data_hex[66:130], 16)
            decoded["time"] = int(data_hex[130:194], 16)

    # ether.fi Events
    elif event_name == "EtherFiDeposit" and len(topics) >= 2:
        # Deposit(address indexed sender, uint256 amount, uint8 sourceOfFunds, address referral)
        decoded["sender"] = _extract_address(topics[1])
        data_hex = data.hex() if isinstance(data, bytes) else data
        if len(data_hex) >= 194:
            decoded["amount"] = int(data_hex[2:66], 16)
            decoded["source_of_funds"] = int(data_hex[66:130], 16)
            decoded["referral"] = "0x" + data_hex[154:194]

    elif event_name == "EtherFiWithdraw" and len(topics) >= 2:
        # Withdraw event from ether.fi liquidity pool
        decoded["user"] = _extract_address(topics[1])
        data_hex = data.hex() if isinstance(data, bytes) else data
        if len(data_hex) >= 66:
            decoded["amount"] = int(data_hex[2:66], 16)

    # Frax Finance Events
    elif event_name == "FraxETHSubmitted" and len(topics) >= 3:
        # ETHSubmitted(address indexed sender, address indexed recipient, uint256 sent_amount, uint256 withheld_amt)
        decoded["sender"] = _extract_address(topics[1])
        decoded["recipient"] = _extract_address(topics[2])
        data_hex = data.hex() if isinstance(data, bytes) else data
        if len(data_hex) >= 130:
            decoded["sent_amount"] = int(data_hex[2:66], 16)
            decoded["withheld_amount"] = int(data_hex[66:130], 16)

    elif event_name == "FraxDeposit" and len(topics) >= 3:
        # Deposit(address indexed sender, address indexed owner, uint256 assets, uint256 shares) - ERC4626
        decoded["sender"] = _extract_address(topics[1])
        decoded["owner"] = _extract_address(topics[2])
        data_hex = data.hex() if isinstance(data, bytes) else data
        if len(data_hex) >= 130:
            decoded["assets"] = int(data_hex[2:66], 16)
            decoded["shares"] = int(data_hex[66:130], 16)

    elif event_name == "FraxWithdraw" and len(topics) >= 4:
        # Withdraw(address indexed sender, address indexed receiver, address indexed owner, uint256 assets, uint256 shares) - ERC4626
        decoded["sender"] = _extract_address(topics[1])
        decoded["receiver"] = _extract_address(topics[2])
        decoded["owner"] = _extract_address(topics[3])
        data_hex = data.hex() if isinstance(data, bytes) else data
        if len(data_hex) >= 130:
            decoded["assets"] = int(data_hex[2:66], 16)
            decoded["shares"] = int(data_hex[66:130], 16)

    # ═══════════════════════════════════════════════════════════════════════════
    # STAKING AGGREGATOR EVENTS
    # ═══════════════════════════════════════════════════════════════════════════

    # Renzo Events
    elif event_name == "RenzoDeposit" and len(topics) >= 3:
        # Deposit(address indexed depositor, address indexed token, uint256 amount, uint256 ezETHMinted, uint256 timestamp)
        decoded["depositor"] = _extract_address(topics[1])
        decoded["token"] = _extract_address(topics[2])
        data_hex = data.hex() if isinstance(data, bytes) else data
        if len(data_hex) >= 194:
            decoded["amount"] = int(data_hex[2:66], 16)
            decoded["ezeth_minted"] = int(data_hex[66:130], 16)
            decoded["timestamp"] = int(data_hex[130:194], 16)

    elif event_name == "RenzoWithdraw" and len(topics) >= 3:
        # UserWithdrawStarted(bytes32 indexed withdrawalRoot, address indexed user, address indexed token, uint256 amount, uint256 ezETHBurned)
        decoded["withdrawal_root"] = (
            topics[1].hex() if isinstance(topics[1], bytes) else topics[1]
        )
        decoded["user"] = _extract_address(topics[2])
        if len(topics) >= 4:
            decoded["token"] = _extract_address(topics[3])
        data_hex = data.hex() if isinstance(data, bytes) else data
        if len(data_hex) >= 130:
            decoded["amount"] = int(data_hex[2:66], 16)
            decoded["ezeth_burned"] = int(data_hex[66:130], 16)

    # Kelp DAO Events
    elif event_name == "KelpDeposit" and len(topics) >= 3:
        # AssetDeposit(address indexed depositor, address indexed asset, uint256 depositAmount, uint256 rsethMintAmount)
        decoded["depositor"] = _extract_address(topics[1])
        decoded["asset"] = _extract_address(topics[2])
        data_hex = data.hex() if isinstance(data, bytes) else data
        if len(data_hex) >= 130:
            decoded["amount"] = int(data_hex[2:66], 16)
            decoded["rseth_minted"] = int(data_hex[66:130], 16)

    elif event_name == "KelpWithdraw" and len(topics) >= 3:
        # AssetWithdraw(address indexed user, address indexed asset, uint256 amount)
        decoded["user"] = _extract_address(topics[1])
        decoded["asset"] = _extract_address(topics[2])
        data_hex = data.hex() if isinstance(data, bytes) else data
        if len(data_hex) >= 66:
            decoded["amount"] = int(data_hex[2:66], 16)

    # Puffer Finance Events
    elif event_name == "PufferDeposit" and len(topics) >= 3:
        # Deposited(address indexed depositor, address indexed receiver, uint256 amount, uint256 pufETHAmount)
        decoded["depositor"] = _extract_address(topics[1])
        decoded["receiver"] = _extract_address(topics[2])
        data_hex = data.hex() if isinstance(data, bytes) else data
        if len(data_hex) >= 130:
            decoded["amount"] = int(data_hex[2:66], 16)
            decoded["pufeth_amount"] = int(data_hex[66:130], 16)

    elif event_name == "PufferWithdraw" and len(topics) >= 3:
        # Withdrawn(address indexed user, address indexed receiver, uint256 pufETHAmount, uint256 ethAmount)
        decoded["user"] = _extract_address(topics[1])
        decoded["receiver"] = _extract_address(topics[2])
        data_hex = data.hex() if isinstance(data, bytes) else data
        if len(data_hex) >= 130:
            decoded["pufeth_amount"] = int(data_hex[2:66], 16)
            decoded["eth_amount"] = int(data_hex[66:130], 16)

    # Pendle Events
    elif event_name == "PendleSwap" and len(topics) >= 4:
        # Swap(address indexed caller, address indexed market, address indexed receiver, int256 netPtOut, int256 netSyOut)
        decoded["caller"] = _extract_address(topics[1])
        decoded["market"] = _extract_address(topics[2])
        decoded["receiver"] = _extract_address(topics[3])
        data_hex = data.hex() if isinstance(data, bytes) else data
        if len(data_hex) >= 130:
            # Signed int256
            net_pt_hex = data_hex[2:66]
            net_sy_hex = data_hex[66:130]
            decoded["net_pt_out"] = (
                int(net_pt_hex, 16)
                if int(net_pt_hex, 16) < 2**255
                else int(net_pt_hex, 16) - 2**256
            )
            decoded["net_sy_out"] = (
                int(net_sy_hex, 16)
                if int(net_sy_hex, 16) < 2**255
                else int(net_sy_hex, 16) - 2**256
            )

    elif event_name == "PendleMint" and len(topics) >= 4:
        # MintPY(address indexed caller, address indexed receiver, address indexed YT, uint256 amountSyMinted, uint256 amountPYMinted)
        decoded["caller"] = _extract_address(topics[1])
        decoded["receiver"] = _extract_address(topics[2])
        decoded["yt"] = _extract_address(topics[3])
        data_hex = data.hex() if isinstance(data, bytes) else data
        if len(data_hex) >= 130:
            decoded["amount_sy_minted"] = int(data_hex[2:66], 16)
            decoded["amount_py_minted"] = int(data_hex[66:130], 16)

    elif event_name == "PendleRedeem" and len(topics) >= 4:
        # RedeemPY(address indexed caller, address indexed receiver, address indexed YT, uint256 amountPYRedeemed, uint256 amountSyOut)
        decoded["caller"] = _extract_address(topics[1])
        decoded["receiver"] = _extract_address(topics[2])
        decoded["yt"] = _extract_address(topics[3])
        data_hex = data.hex() if isinstance(data, bytes) else data
        if len(data_hex) >= 130:
            decoded["amount_py_redeemed"] = int(data_hex[2:66], 16)
            decoded["amount_sy_out"] = int(data_hex[66:130], 16)

    # Origin OETH Events
    elif event_name == "OriginDeposit" and len(topics) >= 3:
        # Deposit(address indexed _asset, address indexed _account, uint256 _amount)
        decoded["asset"] = _extract_address(topics[1])
        decoded["account"] = _extract_address(topics[2])
        data_hex = data.hex() if isinstance(data, bytes) else data
        if len(data_hex) >= 66:
            decoded["amount"] = int(data_hex[2:66], 16)

    elif event_name == "OriginWithdraw" and len(topics) >= 3:
        # Withdraw(address indexed _asset, address indexed _account, uint256 _amount)
        decoded["asset"] = _extract_address(topics[1])
        decoded["account"] = _extract_address(topics[2])
        data_hex = data.hex() if isinstance(data, bytes) else data
        if len(data_hex) >= 66:
            decoded["amount"] = int(data_hex[2:66], 16)

    elif event_name == "OriginMint" and len(topics) >= 2:
        # Mint(address indexed _addr, uint256 _value)
        decoded["account"] = _extract_address(topics[1])
        data_hex = data.hex() if isinstance(data, bytes) else data
        if len(data_hex) >= 66:
            decoded["amount"] = int(data_hex[2:66], 16)

    elif event_name == "OriginRedeem" and len(topics) >= 2:
        # Redeem(address indexed _addr, uint256 _value)
        decoded["account"] = _extract_address(topics[1])
        data_hex = data.hex() if isinstance(data, bytes) else data
        if len(data_hex) >= 66:
            decoded["amount"] = int(data_hex[2:66], 16)

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
        self.parallel_batches = self.indexer_config.get("parallel_batches", 10)
        self.save_interval = self.indexer_config.get("save_interval_batches", 10)

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
        """Fetch logs for a batch of blocks with retry logic and automatic splitting."""
        return self._fetch_logs_with_splitting(
            w3, contract_address, topics, from_block, to_block
        )

    def _fetch_logs_with_splitting(
        self,
        w3: Web3,
        contract_address: str,
        topics: list[str],
        from_block: int,
        to_block: int,
        depth: int = 0,
    ) -> list[dict]:
        """Fetch logs, automatically splitting range if response is too big."""
        max_depth = 8  # Max splits: 2^8 = 256x smaller batches

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
                error_str = str(e)
                # Check if response is too big - split the range
                if "too big" in error_str.lower() or "-32008" in error_str:
                    if depth >= max_depth:
                        print(
                            f"\n  Max split depth reached, skipping blocks {from_block}-{to_block}"
                        )
                        return []
                    # Split range in half and fetch recursively
                    mid_block = (from_block + to_block) // 2
                    if mid_block == from_block:
                        print(f"\n  Cannot split further, skipping block {from_block}")
                        return []
                    logs_first = self._fetch_logs_with_splitting(
                        w3, contract_address, topics, from_block, mid_block, depth + 1
                    )
                    logs_second = self._fetch_logs_with_splitting(
                        w3, contract_address, topics, mid_block + 1, to_block, depth + 1
                    )
                    return logs_first + logs_second
                elif attempt < self.max_retries - 1:
                    print(f"  Retry {attempt + 1}/{self.max_retries} after error: {e}")
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    print(f"  Failed after {self.max_retries} attempts: {e}")
                    return []
        return []

    def _process_logs(
        self,
        logs: list[dict],
        chain_name: str,
        protocol_name: str,
        contract_address: str,
        event_map: dict[str, str],
    ) -> list[dict]:
        """Process logs and convert to event records."""
        events = []
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
            events.append(event_record)
        return events

    def _fetch_batch_task(
        self,
        chain_name: str,
        contract_address: str,
        event_topics: list[str],
        batch_start: int,
        batch_end: int,
        batch_num: int,
    ) -> tuple[int, int, list[dict]]:
        """Task to fetch a single batch. Returns (batch_num, batch_end, logs)."""
        w3 = self._get_web3(chain_name)
        logs = self._fetch_logs_batch(
            w3, contract_address, event_topics, batch_start, batch_end
        )
        return (batch_num, batch_end, logs)

    def fetch_events_for_contract(
        self,
        chain_name: str,
        protocol_name: str,
        contract_address: str,
        event_names: list[str],
        from_block: int,
        to_block: int,
    ) -> int:
        """Fetch all specified events for a contract using parallel batch processing.

        Returns the total number of events fetched. Events are saved incrementally to disk.
        """
        pending_events = []
        total_events_saved = 0

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
        print(
            f"  Fetching {len(event_topics)} event types ({self.parallel_batches} parallel, save every {self.save_interval})"
        )
        print(
            f"  Block range: {from_block:,} → {to_block:,} ({total_blocks:,} blocks, {num_batches} batches)"
        )

        # Prepare output file
        year_month = f"{self.year}_{self.month:02d}"
        filename = f"{chain_name}_{protocol_name}_{year_month}.csv"
        filepath = self.raw_folder / filename
        fieldnames = None
        header_written = False

        # Build list of all batches
        batches = []
        current_block = from_block
        batch_num = 0
        while current_block <= to_block:
            batch_end = min(current_block + self.batch_size - 1, to_block)
            batches.append((batch_num, current_block, batch_end))
            batch_num += 1
            current_block = batch_end + 1

        # Process batches in chunks to limit memory usage
        start_time = time.time()
        completed_batches = 0
        total_events = 0
        chunk_size = self.save_interval  # Process this many batches at a time

        # Process in chunks
        for chunk_start in range(0, len(batches), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(batches))
            chunk_batches = batches[chunk_start:chunk_end]

            with ThreadPoolExecutor(max_workers=self.parallel_batches) as executor:
                # Submit only this chunk of batch tasks
                futures = {
                    executor.submit(
                        self._fetch_batch_task,
                        chain_name,
                        contract_address,
                        event_topics,
                        batch_start,
                        batch_end,
                        batch_num,
                    ): batch_num
                    for batch_num, batch_start, batch_end in chunk_batches
                }

                # Process results as they complete
                for future in as_completed(futures):
                    try:
                        batch_num, batch_end, logs = future.result()
                        completed_batches += 1

                        # Process logs into events
                        events = self._process_logs(
                            logs, chain_name, protocol_name, contract_address, event_map
                        )
                        pending_events.extend(events)
                        total_events += len(logs)

                        # Update progress
                        progress = completed_batches / num_batches * 100
                        elapsed = time.time() - start_time
                        batches_per_sec = completed_batches / max(1, elapsed)
                        eta_seconds = (num_batches - completed_batches) / max(
                            1, batches_per_sec
                        )

                        print(
                            f"    Batch {completed_batches}/{num_batches} ({progress:.1f}%) | "
                            f"{total_events:,} events | {batches_per_sec:.1f} batch/s | ETA: {eta_seconds:.0f}s"
                            + " "
                            * 10,
                            end="\r",
                        )
                    except Exception as e:
                        print(f"\n    Error in batch {futures[future]}: {e}")

                # Clear futures dict to free memory
                del futures

            # Save after each chunk and free memory
            if pending_events:
                pending_events.sort(
                    key=lambda x: (x["block_number"], x.get("log_index", 0))
                )
                if fieldnames is None:
                    fieldnames = self._get_csv_fieldnames(pending_events)
                self._append_events_to_csv(
                    filepath,
                    pending_events,
                    fieldnames,
                    write_header=not header_written,
                )
                header_written = True
                total_events_saved += len(pending_events)
                del pending_events
                gc.collect()
                pending_events = []

        elapsed = time.time() - start_time
        print(f"\n  ✓ Completed: {total_events_saved:,} events in {elapsed:.1f}s")
        return total_events_saved

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

    def _get_csv_fieldnames(self, events: list[dict]) -> list[str]:
        """Get ordered fieldnames for CSV output."""
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
        all_fields = set()
        for event in events:
            all_fields.update(event.keys())
        fieldnames = [f for f in priority_fields if f in all_fields]
        fieldnames += sorted([f for f in all_fields if f not in priority_fields])
        return fieldnames

    def _append_events_to_csv(
        self,
        filepath: Path,
        events: list[dict],
        fieldnames: list[str],
        write_header: bool = False,
    ):
        """Append events to CSV file."""
        mode = "w" if write_header else "a"
        with open(filepath, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerows(events)

    def _save_contract_events(
        self, chain_name: str, protocol_name: str, events: list[dict]
    ):
        """Save events for a single contract to CSV immediately."""
        if not events:
            print(f"    No events to save for {protocol_name}")
            return

        year_month = f"{self.year}_{self.month:02d}"
        filename = f"{chain_name}_{protocol_name}_{year_month}.csv"
        filepath = self.raw_folder / filename

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

        # Get all unique fields
        all_fields = set()
        for event in events:
            all_fields.update(event.keys())

        fieldnames = [f for f in priority_fields if f in all_fields]
        fieldnames += sorted([f for f in all_fields if f not in priority_fields])

        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(events)

        print(f"    ✓ Saved {filename}: {len(events)} events")

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

                event_count = self.fetch_events_for_contract(
                    chain_name,
                    protocol_name,
                    contract_address,
                    event_names,
                    from_block,
                    to_block,
                )

                key = f"{chain_name}_{protocol_name}"
                all_events[key] = event_count

        print(f"\nOutput folder: {self.raw_folder.absolute()}")
        return all_events


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
    fetcher.fetch_all_events(chains_filter, contract_filter)

    print("\n✓ Done!")
    return 0


if __name__ == "__main__":
    exit(main())
