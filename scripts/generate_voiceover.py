"""
Generate voiceover audio for Airia Sentinel demo video (3 minutes).
Each section is generated separately via Edge TTS, then concatenated
with real silence gaps using ffmpeg.
Output: data/voiceover/airia_sentinel_demo.mp3
"""

import asyncio
import os
import subprocess
import tempfile
import edge_tts

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "voiceover")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "airia_sentinel_demo.mp3")

VOICE = "en-US-GuyNeural"
RATE = "-5%"

# (text, pause_after_in_seconds)
SCRIPT = [
    # INTRO
    (
        "Enterprise treasury teams monitor dozens of financial instruments every single day, "
        "crypto, forex, commodities, while managing millions in corporate exposure, "
        "it is slow, fragmented, and error prone",
        1.0,
    ),
    (
        "What if sixteen specialized AI agents could do this in under ten seconds",
        1.0,
    ),

    # WHAT IS AIRIA SENTINEL
    (
        "This is Airia Sentinel, a multi-agent orchestration platform for treasury risk management, "
        "it runs a five phase pipeline of sixteen AI agents, powered by Airia SDK, LM Studio, and Ollama",
        1.0,
    ),
    (
        "Let me show you how it works",
        1.0,
    ),

    # PHASE 1
    (
        "Phase one, six agents launch in parallel, "
        "Market Intelligence scans ten instruments in real time, "
        "Corporate Context analyzes treasury exposure, "
        "the Anomaly Detector looks for volume spikes, "
        "the Sentiment Scorer classifies market mood, "
        "the Liquidity Analyzer checks order book depth, "
        "and the News Scanner detects market moving events",
        1.0,
    ),

    # PHASE 2
    (
        "Phase two, three agents analyze the signals together, "
        "the Risk Aggregator fuses everything with weighted scoring and cascade detection, "
        "the Correlation Matrix identifies cross asset dependencies, "
        "and the Volatility Forecaster computes Value at Risk estimates",
        1.0,
    ),

    # PHASE 3
    (
        "Phase three, Consensus Strategy queries four AI models in parallel, "
        "then merges their hedging recommendations using weighted voting, "
        "the Position Sizer calculates optimal allocation using Kelly Criterion and Risk Parity",
        1.0,
    ),

    # PHASE 4
    (
        "Phase four, validation, "
        "the Strategy Backtester runs a Monte Carlo simulation, fifty trajectories over thirty days, "
        "computing Sharpe ratio, max drawdown, and win rate, "
        "the Portfolio Optimizer computes the efficient frontier using Markowitz mean variance optimization",
        1.0,
    ),

    # PHASE 5
    (
        "Phase five, four agents run simultaneously, "
        "Compliance generates a PDF report, "
        "the Report Synthesizer assembles an executive report, "
        "the Alert Agent dispatches warnings, "
        "and the Execution Strategist plans order slicing",
        1.0,
    ),

    # HITL
    (
        "Now the Human in the Loop gateway kicks in, "
        "no strategy executes without human approval, "
        "the CFO reviews the recommended hedge, sees the risk metrics, and clicks approve",
        1.0,
    ),

    # DASHBOARD
    (
        "Everything runs on a real time web dashboard, "
        "you can see all sixteen agents completing live, market signals updating, "
        "risk gauges moving, the full audit trail, and cluster health for all five AI nodes",
        1.0,
    ),
    (
        "The architecture is fully hybrid, every agent tries Airia cloud first, "
        "then falls back to local GPU inference, "
        "this means the platform works even offline, "
        "all prompts are bilingual, English and French, with a single toggle",
        0.5,
    ),

    # CLOSING
    (
        "To summarize, Airia Sentinel orchestrates sixteen AI agents across five phases, "
        "from market scanning to executive reporting, in under ten seconds, "
        "it combines Airia SDK with a local GPU cluster, "
        "every decision is audited, every strategy is backtested, and every execution requires human approval",
        1.0,
    ),
    (
        "This is Airia Sentinel, multi agent treasury intelligence, powered by Airia",
        0,
    ),
]


async def generate_section(text: str, index: int, tmp_dir: str) -> str:
    """Generate one section as a temporary MP3 file."""
    path = os.path.join(tmp_dir, f"section_{index:02d}.mp3")
    communicate = edge_tts.Communicate(text, voice=VOICE, rate=RATE)
    await communicate.save(path)
    return path


def get_duration(path: str) -> float:
    """Get audio duration in seconds using ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def generate_silence(duration_s: float, tmp_dir: str, index: int) -> str:
    """Generate a silent MP3 file of given duration."""
    path = os.path.join(tmp_dir, f"silence_{index:02d}.mp3")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i",
         f"anullsrc=r=24000:cl=mono", "-t", str(duration_s),
         "-c:a", "libmp3lame", "-b:a", "192k", "-q:a", "2", path],
        capture_output=True,
    )
    return path


async def generate():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Voice: {VOICE}")
    print(f"Rate:  {RATE}")
    print(f"Sections: {len(SCRIPT)}")
    print()

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Step 1: Generate each section as separate MP3
        all_parts = []
        total_speech = 0

        for i, (text, pause_s) in enumerate(SCRIPT):
            path = await generate_section(text, i, tmp_dir)
            dur = get_duration(path)
            total_speech += dur
            print(f"  Section {i + 1:2d}: {dur:5.1f}s  {text[:65]}")
            all_parts.append(path)

            # Add silence after section if needed
            if pause_s > 0 and i < len(SCRIPT) - 1:
                silence_path = generate_silence(pause_s, tmp_dir, i)
                all_parts.append(silence_path)
                total_speech += pause_s

        # Step 2: Create ffmpeg concat list
        concat_list = os.path.join(tmp_dir, "concat.txt")
        with open(concat_list, "w") as f:
            for part in all_parts:
                f.write(f"file '{part.replace(os.sep, '/')}'\n")

        # Step 3: Concatenate all parts
        print(f"\nConcatenating {len(all_parts)} audio parts...")
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", concat_list, "-c:a", "libmp3lame", "-b:a", "192k",
             "-q:a", "2", OUTPUT_FILE],
            capture_output=True,
        )

    # Final stats
    duration = get_duration(OUTPUT_FILE)
    file_size = os.path.getsize(OUTPUT_FILE)
    minutes = int(duration // 60)
    seconds = int(duration % 60)

    print(f"\nDone!")
    print(f"Output:   {os.path.abspath(OUTPUT_FILE)}")
    print(f"Size:     {file_size / 1024:.0f} KB")
    print(f"Duration: {minutes}:{seconds:02d}")

    if duration < 160:
        print(f"\nWARNING: Audio is {duration:.0f}s, target is 170-180s")
    elif duration > 185:
        print(f"\nWARNING: Audio is {duration:.0f}s, may exceed 3 min limit")
    else:
        print(f"\nPerfect for a 3-minute demo video")


if __name__ == "__main__":
    asyncio.run(generate())
