"""
Generate voiceover audio for Airia Sentinel demo video.
Uses Edge TTS (Microsoft) with en-US-GuyNeural voice.
Output: data/voiceover/airia_sentinel_demo.mp3
"""

import asyncio
import os
import edge_tts

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "voiceover")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "airia_sentinel_demo.mp3")

VOICE = "en-US-GuyNeural"  # Professional male voice
RATE = "-5%"  # Slightly slower for clarity

# Sections with pauses between them (SSML break tags)
SECTIONS = [
    # INTRO [0:00 - 0:20]
    (
        "Enterprise treasury teams monitor dozens of financial instruments every single day "
        "— crypto, forex, commodities — while managing millions in corporate exposure. "
        "It's slow, fragmented, and error-prone."
    ),
    (
        "What if sixteen specialized AI agents could do this in under ten seconds?"
    ),

    # WHAT IS AIRIA SENTINEL [0:20 - 0:40]
    (
        "This is Airia Sentinel — a multi-agent orchestration platform for treasury risk management. "
        "It runs a five-phase pipeline of sixteen AI agents, powered by Airia SDK, LM Studio, and Ollama."
    ),
    (
        "Let me show you how it works."
    ),

    # PHASE 1 & 2 [0:40 - 1:10]
    (
        "Phase one: six agents launch in parallel. "
        "Market Intelligence scans ten instruments in real-time via CCXT. "
        "Corporate Context analyzes internal treasury exposure. "
        "Meanwhile, the Anomaly Detector runs z-score analysis looking for volume spikes and contagion patterns. "
        "The Sentiment Scorer classifies market sentiment. "
        "The Liquidity Analyzer checks order book depth. "
        "And the News Scanner detects market-moving events."
    ),
    (
        "Phase two: three agents analyze the signals together. "
        "The Risk Aggregator fuses everything with weighted scoring and cascade detection. "
        "The Correlation Matrix identifies cross-asset dependencies. "
        "And the Volatility Forecaster computes Value-at-Risk estimates."
    ),

    # PHASE 3 & 4 [1:10 - 1:40]
    (
        "Phase three: Consensus Strategy queries four different AI models in parallel "
        "— LM Studio M1, M2, Ollama, and Airia cloud — "
        "then merges their hedging recommendations using weighted voting. "
        "The Position Sizer calculates optimal allocation using Kelly Criterion and Risk Parity."
    ),
    (
        "Phase four: validation. "
        "The Strategy Backtester runs a Monte Carlo simulation — fifty trajectories over thirty days — "
        "computing Sharpe ratio, max drawdown, and win rate. "
        "The Portfolio Optimizer computes the efficient frontier using Markowitz mean-variance optimization."
    ),

    # PHASE 5 + HITL [1:40 - 2:10]
    (
        "Phase five: four agents run simultaneously. "
        "Compliance generates a PDF report enriched by Airia's executive summary. "
        "The Report Synthesizer assembles a six-section executive report. "
        "The Alert Agent dispatches warnings based on risk thresholds. "
        "And the Execution Strategist plans TWAP and VWAP order slicing with slippage estimation."
    ),
    (
        "Now the Human-in-the-Loop gateway kicks in. "
        "No strategy executes without human approval. "
        "The CFO reviews the recommended hedge, sees the risk metrics, the backtest results, and clicks Approve."
    ),

    # DASHBOARD [2:10 - 2:35]
    (
        "Everything runs on a real-time web dashboard — "
        "you can see all sixteen agents completing live, market signals updating, "
        "risk gauges moving, the full audit trail, and cluster health for all five AI nodes."
    ),
    (
        "The architecture is fully hybrid: every agent tries Airia cloud first, "
        "then falls back to local GPU inference. "
        "This means the platform works even offline. "
        "All prompts are bilingual — English and French — with a single toggle."
    ),

    # CLOSING [2:35 - 3:00]
    (
        "To summarize: Airia Sentinel orchestrates sixteen AI agents across five phases, "
        "from market scanning to executive reporting, in under ten seconds. "
        "It combines Airia SDK with a local GPU cluster of nine GPUs and forty-six gigabytes of VRAM. "
        "Every decision is audited, every strategy is backtested, and every execution requires human approval."
    ),
    (
        "This is Airia Sentinel — multi-agent treasury intelligence, powered by Airia."
    ),
]

PAUSE_BETWEEN_SECTIONS_MS = 1200


async def generate():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Build full SSML text with pauses
    ssml_parts = []
    for i, section in enumerate(SECTIONS):
        ssml_parts.append(section)
        if i < len(SECTIONS) - 1:
            ssml_parts.append(f'<break time="{PAUSE_BETWEEN_SECTIONS_MS}ms"/>')

    full_text = " ".join(ssml_parts)

    # Use SSML for better control
    ssml = (
        f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">'
        f'<voice name="{VOICE}">'
        f'<prosody rate="{RATE}">'
        f'{full_text}'
        f'</prosody>'
        f'</voice>'
        f'</speak>'
    )

    print(f"Voice: {VOICE}")
    print(f"Rate: {RATE}")
    print(f"Sections: {len(SECTIONS)}")
    print(f"Generating audio...")

    communicate = edge_tts.Communicate(ssml, voice=VOICE)
    await communicate.save(OUTPUT_FILE)

    file_size = os.path.getsize(OUTPUT_FILE)
    print(f"\nDone!")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Size: {file_size / 1024:.0f} KB")
    print(f"\nPlay with: start {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(generate())
