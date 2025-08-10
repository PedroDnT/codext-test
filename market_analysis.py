"""
market_analysis.py
--------------------

This script fetches real‑time cryptocurrency market data using the `ccxt`
library and then sends that data to a large language model (LLM) for
interpretation of potential trading opportunities.  The program
demonstrates how to combine market data with AI reasoning to generate
trade setups similar to the analysis performed on 10\u00a0Aug\u00a02025.

The script is configurable via environment variables.  Sensitive API
credentials (e.g. API keys for OpenAI, Anthropic, Gemini, Bybit,
Coinglass, etc.) should be supplied through environment variables
rather than hard‑coded into the source.  See the bottom of this file
for a list of expected environment variables.

Example usage::

    export OPENAI_API_KEY=<your openai key>
    python market_analysis.py

The program will fetch recent price data for a set of symbols
(BTC/USDT, ETH/USDT, SOL/USDT, XRP/USDT, DOGE/USDT) from Binance
using ccxt.  It then constructs a prompt summarizing the data and
sends the prompt to the configured LLM.  The LLM’s response (which
should include trade ideas) is printed to the console.

Note:  This script does *not* execute trades or manage real capital.
It is intended for research and informational purposes only.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any

try:
    import ccxt  # type: ignore
except ImportError as exc:
    raise ImportError(
        "ccxt library is required. Install it via 'pip install ccxt'."
    ) from exc

try:
    import openai  # type: ignore
except ImportError:
    # The openai library may not be installed in every environment.  It
    # is optional; if unavailable the script will emit a warning and
    # skip the AI call.
    openai = None


def fetch_symbol_data(exchange: ccxt.Exchange, symbol: str) -> Dict[str, Any]:
    """Fetch ticker data for a given symbol.

    Returns a dictionary with the last price, 24h high, low, weighted
    average price (VWAP), and base volume.
    """
    ticker = exchange.fetch_ticker(symbol)
    return {
        "symbol": symbol,
        "last": ticker.get("last"),
        "high": ticker.get("high"),
        "low": ticker.get("low"),
        "weightedAvg": ticker.get("vwap"),
        "volume": ticker.get("baseVolume"),
    }


def build_prompt(data: List[Dict[str, Any]]) -> str:
    """Construct a prompt summarizing market data for the AI model.

    This function generates a human‑readable description of each symbol,
    including last price, 24h high/low, weighted average, and volume.
    It then requests the AI model to identify trade opportunities and
    provide entry, stop loss, take profit and other details.
    """
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines: List[str] = []
    lines.append(
        f"As of {timestamp}, here is the latest perpetual futures market data:"
    )
    for item in data:
        symbol = item["symbol"]
        last = item["last"]
        high = item["high"]
        low = item["low"]
        weighted = item["weightedAvg"]
        volume = item["volume"]
        lines.append(
            f"- {symbol}: last price {last}, 24h high {high}, 24h low {low}, "
            f"weighted average {weighted}, volume {volume}."
        )
    lines.append(
        "\nPlease analyze this data and suggest high‑conviction intraday or short swing trades (long or short) "
        "for the next few hours.  For each suggestion, specify the symbol, direction, entry price, take profit target(s), "
        "stop loss, trailing stop strategy, and a conviction level from 1 to 5.  Briefly justify each trade using indicators "
        "like funding rates, long/short ratios, liquidation zones, support/resistance, OBV, RSI, top trader positioning, or other relevant data."
    )
    return "\n".join(lines)


def call_openai(prompt: str) -> str:
    """Call the OpenAI chat model to obtain analysis.

    The OPENAI_API_KEY environment variable must be set.  If the
    openai library is unavailable or the key is missing, a placeholder
    message is returned instead of raising an exception.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if openai is None:
        return "[OpenAI library is not installed. Install openai to call the model.]"
    if not api_key:
        return "[OPENAI_API_KEY environment variable is not set. Cannot call OpenAI API.]"
    openai.api_key = api_key
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional crypto market analyst."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )
        choices = response.get("choices", [])
        if choices:
            return choices[0]["message"]["content"].strip()
    except Exception as exc:
        return f"[Error calling OpenAI API: {exc}]"
    return "[No response from OpenAI API.]"


def main() -> None:
    """Main routine for market analysis."""
    # Create a CCXT exchange instance for Binance.  This does not require
    # authentication for public market data.
    exchange = ccxt.binance()
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT"]
    market_data: List[Dict[str, Any]] = []
    for symbol in symbols:
        try:
            data = fetch_symbol_data(exchange, symbol)
            market_data.append(data)
        except Exception as exc:
            print(f"Warning: failed to fetch data for {symbol}: {exc}")
    # Build the AI prompt
    prompt = build_prompt(market_data)
    # Call the model
    analysis = call_openai(prompt)
    # Print results
    print("=== Market Data ===")
    print(json.dumps(market_data, indent=2))
    print("\n=== AI Analysis ===")
    print(analysis)


if __name__ == "__main__":
    main()
