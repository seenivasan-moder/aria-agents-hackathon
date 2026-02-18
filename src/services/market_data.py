"""Market data service — CCXT for crypto, simulated Forex/commodities."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import numpy as np

from src.config import config


async def fetch_crypto_data(pairs: list[str] | None = None) -> list[dict[str, Any]]:
    """Fetch crypto market data via CCXT (MEXC)."""
    import ccxt.async_support as ccxt

    pairs = pairs or [p for p in config.watched_pairs if "/" in p and "USD" in p and p.count("/") == 1]
    crypto_pairs = [p for p in pairs if not any(x in p for x in ["EUR/", "GBP/", "JPY/", "XAU", "XAG", "CL/", "NG/"])]

    if not crypto_pairs:
        crypto_pairs = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

    exchange = ccxt.mexc({
        "apiKey": config.mexc_api_key or None,
        "secret": config.mexc_secret_key or None,
        "enableRateLimit": True,
    })

    results = []
    try:
        tickers = await exchange.fetch_tickers(crypto_pairs)
        for symbol, ticker in tickers.items():
            price = ticker.get("last", 0) or 0
            change_24h = ticker.get("percentage", 0) or 0
            volume = ticker.get("quoteVolume", 0) or 0
            high = ticker.get("high", price) or price
            low = ticker.get("low", price) or price

            volatility = ((high - low) / price * 100) if price > 0 else 0
            risk_score = min(100, abs(change_24h) * 5 + volatility * 10)

            if change_24h < -3:
                direction = "bearish"
            elif change_24h > 3:
                direction = "bullish"
            else:
                direction = "neutral"

            if change_24h < -10:
                regime = "oversold"
            elif change_24h > 10:
                regime = "overbought"
            else:
                regime = "neutral"

            results.append({
                "symbol": symbol,
                "asset_class": "crypto",
                "price": price,
                "change_1h": 0,
                "change_24h": round(change_24h, 2),
                "change_7d": 0,
                "volume_24h": round(volume, 2),
                "volatility": round(volatility, 2),
                "funding_rate": 0,
                "risk_score": round(risk_score, 1),
                "direction": direction,
                "regime": regime,
            })
    except Exception as e:
        results.append({"symbol": "ERROR", "error": str(e), "asset_class": "crypto", "price": 0,
                        "risk_score": 0, "direction": "neutral", "regime": "neutral",
                        "change_24h": 0, "volatility": 0, "volume_24h": 0})
    finally:
        await exchange.close()

    return results


def generate_forex_data() -> list[dict[str, Any]]:
    """Generate simulated but realistic Forex data."""
    rng = np.random.default_rng(int(time.time()) // 3600)

    forex_pairs = [
        ("EUR/USD", 1.0850, 0.005),
        ("GBP/USD", 1.2650, 0.007),
        ("USD/JPY", 149.50, 0.8),
        ("USD/CHF", 0.8820, 0.004),
        ("AUD/USD", 0.6530, 0.006),
    ]

    results = []
    for symbol, base_price, std_dev in forex_pairs:
        noise = rng.normal(0, std_dev)
        price = round(base_price + noise, 4)
        change_24h = round(rng.normal(0, 0.3), 2)
        volatility = round(abs(rng.normal(0.5, 0.2)), 2)
        risk_score = round(min(100, abs(change_24h) * 15 + volatility * 20), 1)

        results.append({
            "symbol": symbol,
            "asset_class": "forex",
            "price": price,
            "change_1h": round(rng.normal(0, 0.05), 2),
            "change_24h": change_24h,
            "change_7d": round(rng.normal(0, 0.8), 2),
            "volume_24h": 0,
            "volatility": volatility,
            "funding_rate": 0,
            "risk_score": risk_score,
            "direction": "bearish" if change_24h < -0.2 else ("bullish" if change_24h > 0.2 else "neutral"),
            "regime": "neutral",
        })

    return results


def generate_commodity_data() -> list[dict[str, Any]]:
    """Generate simulated commodity data."""
    rng = np.random.default_rng(int(time.time()) // 3600)

    commodities = [
        ("XAU/USD", 2340.0, 15.0, "Gold"),
        ("XAG/USD", 29.50, 0.8, "Silver"),
        ("CL/USD", 78.50, 2.0, "Crude Oil WTI"),
        ("NG/USD", 2.85, 0.15, "Natural Gas"),
        ("HG/USD", 4.25, 0.1, "Copper"),
    ]

    results = []
    for symbol, base_price, std_dev, name in commodities:
        noise = rng.normal(0, std_dev)
        price = round(base_price + noise, 2)
        change_24h = round(rng.normal(0, 1.0), 2)
        volatility = round(abs(rng.normal(1.2, 0.5)), 2)
        risk_score = round(min(100, abs(change_24h) * 8 + volatility * 12), 1)

        results.append({
            "symbol": symbol,
            "asset_class": "commodity",
            "price": price,
            "change_1h": round(rng.normal(0, 0.2), 2),
            "change_24h": change_24h,
            "change_7d": round(rng.normal(0, 2.5), 2),
            "volume_24h": 0,
            "volatility": volatility,
            "funding_rate": 0,
            "risk_score": risk_score,
            "direction": "bearish" if change_24h < -0.5 else ("bullish" if change_24h > 0.5 else "neutral"),
            "regime": "neutral",
        })

    return results


async def fetch_all_market_data() -> list[dict[str, Any]]:
    """Fetch all market data: crypto (live) + forex + commodities (simulated)."""
    crypto = await fetch_crypto_data()
    forex = generate_forex_data()
    commodities = generate_commodity_data()
    return crypto + forex + commodities
