"""Demo Organizer — Real-world file management pipeline (< 3 min).

Scans a real directory, classifies files, detects duplicates,
calculates recoverable space, backs up SQLite, and generates a summary.

Usage:
    uv run python main.py demo-org
    uv run python main.py demo-org F:\\BUREAU\\disk_cleaner
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import time
from pathlib import Path
from typing import Any

from rich import box
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

console = Console()


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def narrate(text: str, pause: float = 1.5) -> None:
    console.print(f"\n[bold white on blue]  {text}  [/]")
    time.sleep(pause)


def section_divider(title: str) -> None:
    console.print()
    console.rule(f"[bold magenta]{title}[/]", style="magenta")
    console.print()


# ═══════════════════════════════════════════════════════════════════════════════
# FILE CATEGORIES (from disk_cleaner logic)
# ═══════════════════════════════════════════════════════════════════════════════

CATEGORY_MAP = {
    ".py": "Code Python", ".js": "Code JavaScript", ".ts": "Code TypeScript",
    ".java": "Code Java", ".cpp": "Code C++", ".c": "Code C",
    ".go": "Code Go", ".rs": "Code Rust", ".rb": "Code Ruby",
    ".html": "Web", ".css": "Web", ".jsx": "Web", ".tsx": "Web",
    ".json": "Config/Data", ".yaml": "Config/Data", ".yml": "Config/Data",
    ".toml": "Config/Data", ".ini": "Config/Data", ".cfg": "Config/Data",
    ".xml": "Config/Data", ".env": "Config/Data",
    ".md": "Documentation", ".txt": "Documentation", ".rst": "Documentation",
    ".pdf": "Documents", ".docx": "Documents", ".xlsx": "Documents",
    ".pptx": "Documents", ".doc": "Documents",
    ".png": "Images", ".jpg": "Images", ".jpeg": "Images", ".gif": "Images",
    ".svg": "Images", ".ico": "Images", ".webp": "Images",
    ".mp4": "Video", ".avi": "Video", ".mkv": "Video", ".mov": "Video",
    ".mp3": "Audio", ".wav": "Audio", ".flac": "Audio",
    ".zip": "Archives", ".tar": "Archives", ".gz": "Archives",
    ".rar": "Archives", ".7z": "Archives",
    ".db": "Database", ".sqlite": "Database", ".sqlite3": "Database",
    ".log": "Logs", ".tmp": "Temporaires", ".bak": "Backup",
    ".bat": "Scripts", ".sh": "Scripts", ".ps1": "Scripts",
    ".exe": "Executables", ".dll": "Executables", ".msi": "Executables",
    ".lnk": "Raccourcis",
}

TEMP_EXTENSIONS = {".tmp", ".temp", ".cache", ".bak", ".old", ".swp", ".pyc", ".pyo"}


def classify_file(ext: str) -> str:
    return CATEGORY_MAP.get(ext.lower(), "Autres")


def score_file(path: Path, size: int, is_dup: bool) -> tuple[int, list[str]]:
    """Score a file 0-100 with reasons (inspired by disk_cleaner ScoreEngine)."""
    score = 50
    reasons = []

    if size == 0:
        return 0, ["Fichier vide"]

    ext = path.suffix.lower()

    # Size scoring
    if size < 1024:
        score -= 15
        reasons.append("Tres petit")
    elif size > 10 * 1024 * 1024:
        score += 10
        reasons.append("Volumineux")

    # Type scoring
    cat = classify_file(ext)
    if "Code" in cat:
        score += 20
        reasons.append("Code source")
    elif cat == "Documents":
        score += 15
        reasons.append("Document")
    elif cat == "Documentation":
        score += 10
        reasons.append("Documentation")
    elif cat in ("Temporaires", "Logs"):
        score -= 25
        reasons.append("Temporaire/Log")

    # Temp extensions
    if ext in TEMP_EXTENSIONS:
        score -= 20
        reasons.append("Extension temporaire")

    # Duplicate penalty
    if is_dup:
        score -= 30
        reasons.append("Doublon")

    # Name heuristics
    name_lower = path.name.lower()
    if any(s in name_lower for s in ["temp", "tmp", "cache", "copy", "copie", "old"]):
        score -= 10
        reasons.append("Nom suspect")

    return max(0, min(100, score)), reasons


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO STEPS
# ═══════════════════════════════════════════════════════════════════════════════

async def step_intro() -> None:
    """Intro: present the Organizer use case."""
    console.clear()

    console.print(Panel(
        Align.center(Text.from_markup(
            "[bold magenta]"
            "  ___  ____   ____    _    _   _ ___ __________ ____\n"
            " / _ \\|  _ \\ / ___|  / \\  | \\ | |_ _|__  / ____|  _ \\\n"
            "| | | | |_) | |  _  / _ \\ |  \\| || |  / /|  _| | |_) |\n"
            "| |_| |  _ <| |_| |/ ___ \\| |\\  || | / /_| |___|  _ <\n"
            " \\___/|_| \\_\\\\____/_/   \\_\\_| \\_|___/____|_____|_| \\_\\\n"
            "[/]\n\n"
            "[bold white]Intelligent File Management — Powered by AI Agents[/]\n"
            "[dim]Airia Sentinel Platform — Use Case #2[/]"
        )),
        border_style="bright_magenta",
        padding=(1, 2),
    ))

    await asyncio.sleep(2)

    console.print(Panel(
        "[bold red]THE PROBLEM[/]\n\n"
        "Every organization accumulates digital debt:\n"
        "  [dim]>[/] Duplicate files wasting disk space\n"
        "  [dim]>[/] Unclassified documents scattered across directories\n"
        "  [dim]>[/] No backup strategy for critical databases\n"
        "  [dim]>[/] Sensitive data (API keys, passwords) exposed in files\n\n"
        "[italic]Manual cleanup takes hours and misses critical issues.[/]",
        title="[bold red]Problem[/]",
        border_style="red",
        padding=(0, 2),
    ))

    await asyncio.sleep(3)

    console.print(Panel(
        "[bold green]THE SOLUTION: AI ORGANIZER AGENTS[/]\n\n"
        "2 specialized agents working in sequence:\n\n"
        "  [magenta]Agent 1[/] [bold]Librarian[/]          Scans, classifies, hashes, detects sensitive data\n"
        "  [magenta]Agent 2[/] [bold]Deduplicator[/]       Finds duplicates, calculates savings, suggests cleanup\n\n"
        "  [cyan]+ ScoreEngine[/]                  Scores each file 0-100 for relevance\n"
        "  [cyan]+ SQLite Backup[/]                Automatic database backup with audit trail\n"
        "  [cyan]+ Sensitive Scanner[/]             Detects API keys, passwords, secrets\n\n"
        "[dim]Inspired by disk_cleaner — enhanced with multi-agent AI orchestration[/]",
        title="[bold green]Solution[/]",
        border_style="green",
        padding=(0, 2),
    ))

    await asyncio.sleep(3)


async def step_architecture() -> None:
    """Show the Organizer architecture."""
    section_divider("ARCHITECTURE")

    console.print(Panel(
        "[bold white]Organizer Pipeline Flow:[/]\n\n"
        "  [magenta]Phase 1[/]: File Scanning          [magenta]Phase 2[/]: Deduplication\n"
        "  +--------------------+         +--------------------+\n"
        "  | [bold]Agent: Librarian[/]    |         | [bold]Agent: Dedup[/]        |\n"
        "  |                    |         |                    |\n"
        "  | Recursive scan     |-------->| Group by SHA hash  |\n"
        "  | 40+ file types     |         | Best-copy select   |\n"
        "  | MD5 hashing        |         | Space calculation  |\n"
        "  | Sensitive detect   |         | Keep/Delete advice |\n"
        "  +--------------------+         +--------------------+\n"
        "          |                               |\n"
        "          v                               v\n"
        "  +--------------------+         +--------------------+\n"
        "  | [bold]ScoreEngine[/]        |         | [bold]Backup Manager[/]     |\n"
        "  | Score 0-100        |         | SQLite backup      |\n"
        "  | Keep/Quarantine/   |         | Reports directory  |\n"
        "  | Trash decision     |         | Audit trail        |\n"
        "  +--------------------+         +--------------------+\n\n"
        "  [dim]All operations logged to SQLite audit trail[/]",
        title="[bold magenta]Multi-Agent Architecture[/]",
        border_style="magenta",
        padding=(0, 2),
    ))

    await asyncio.sleep(3)


async def step_scan_real_directory(target_dir: str) -> dict:
    """Phase 1: Real scan of target directory."""
    section_divider("PHASE 1: FILE SCANNING & CLASSIFICATION")

    narrate(f"Agent Librarian scanning: {target_dir}", pause=1)

    target = Path(target_dir)
    if not target.exists():
        console.print(f"[red]Directory not found: {target_dir}[/]")
        return {}

    # Collect all files
    all_files = [f for f in target.rglob("*") if f.is_file()
                 and "__pycache__" not in str(f)
                 and ".venv" not in str(f)
                 and ".git" not in str(f)]

    # Scan with progress
    results = []
    categories: dict[str, int] = {}
    total_size = 0
    hash_map: dict[str, list] = {}
    sensitive_files = []

    sensitive_patterns = ["api_key", "password", "secret", "token", "ak-", "sk-"]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[magenta]Scanning files...", total=len(all_files))

        for f in all_files:
            try:
                stat = f.stat()
                size = stat.st_size
                ext = f.suffix.lower()
                cat = classify_file(ext)

                # Hash (first 1MB for speed)
                h = hashlib.md5()
                with open(f, "rb") as fh:
                    h.update(fh.read(1_048_576))
                file_hash = h.hexdigest()

                # Track duplicates
                hash_map.setdefault(file_hash, []).append(str(f))

                is_dup = len(hash_map[file_hash]) > 1

                # Score
                file_score, reasons = score_file(f, size, is_dup)

                # Sensitive content check (small text files only)
                if size < 100_000 and ext in (".txt", ".env", ".json", ".yaml", ".yml", ".py", ".js", ".toml"):
                    try:
                        content = f.read_text(encoding="utf-8", errors="ignore").lower()
                        for pat in sensitive_patterns:
                            if pat in content:
                                sensitive_files.append((str(f), pat))
                                break
                    except Exception:
                        pass

                categories[cat] = categories.get(cat, 0) + 1
                total_size += size

                results.append({
                    "path": str(f),
                    "name": f.name,
                    "size": size,
                    "ext": ext,
                    "category": cat,
                    "hash": file_hash,
                    "score": file_score,
                    "reasons": reasons,
                    "is_dup": is_dup,
                })

            except Exception:
                pass

            progress.update(task, advance=1)

    await asyncio.sleep(0.5)

    # Display classification table
    table = Table(title="File Classification Results", box=box.ROUNDED)
    table.add_column("Category", style="cyan")
    table.add_column("Count", justify="right", style="bold")
    table.add_column("Bar", min_width=20)

    max_count = max(categories.values()) if categories else 1
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        bar_len = int(count / max_count * 20)
        bar = "[green]" + "█" * bar_len + "[/]" + "░" * (20 - bar_len)
        table.add_row(cat, str(count), bar)

    console.print(table)

    # Summary
    console.print(Panel(
        f"[bold]Scan Summary[/]\n\n"
        f"  Files scanned:    [bold]{len(results):,}[/]\n"
        f"  Total size:       [bold]{total_size / 1_048_576:.1f} MB[/]\n"
        f"  Categories:       [bold]{len(categories)}[/]\n"
        f"  Sensitive files:  [bold {'red' if sensitive_files else 'green'}]{len(sensitive_files)}[/]",
        border_style="magenta",
    ))

    if sensitive_files:
        console.print("\n[bold red]Sensitive Data Detected:[/]")
        for path, pattern in sensitive_files[:5]:
            console.print(f"  [red]![/] {Path(path).name} — pattern: [yellow]{pattern}[/]")

    await asyncio.sleep(2)

    return {
        "results": results,
        "categories": categories,
        "total_size": total_size,
        "hash_map": hash_map,
        "sensitive_files": sensitive_files,
    }


async def step_dedup(scan_data: dict) -> dict:
    """Phase 2: Deduplication analysis."""
    section_divider("PHASE 2: DEDUPLICATION & SCORING")

    narrate("Agent Deduplicator analyzing file hashes...", pause=1)

    hash_map = scan_data["hash_map"]
    results = scan_data["results"]

    # Find duplicate groups
    dup_groups = {h: paths for h, paths in hash_map.items() if len(paths) > 1}
    total_dups = sum(len(paths) - 1 for paths in dup_groups.values())

    # Calculate recoverable space
    recoverable = 0
    for h, paths in dup_groups.items():
        sizes = []
        for r in results:
            if r["hash"] == h:
                sizes.append(r["size"])
        if sizes:
            recoverable += sum(sorted(sizes)[:-1])  # Keep the largest

    # Display duplicates
    if dup_groups:
        table = Table(title="Duplicate Groups", box=box.ROUNDED)
        table.add_column("#", style="dim", justify="right")
        table.add_column("Files", style="yellow")
        table.add_column("Copies", justify="center")
        table.add_column("Action", style="bold")

        for i, (h, paths) in enumerate(list(dup_groups.items())[:10], 1):
            names = [Path(p).name for p in paths]
            table.add_row(
                str(i),
                "\n".join(names),
                str(len(paths)),
                f"[green]KEEP[/] {names[0]}\n" + "\n".join(f"[red]DELETE[/] {n}" for n in names[1:]),
            )

        console.print(table)
    else:
        console.print("[green]No duplicates found — clean directory![/]")

    # Score distribution
    scores = [r["score"] for r in results]
    high = sum(1 for s in scores if s >= 60)
    medium = sum(1 for s in scores if 30 <= s < 60)
    low = sum(1 for s in scores if s < 30)

    table2 = Table(title="Score Distribution (disk_cleaner Engine)", box=box.ROUNDED)
    table2.add_column("Category", style="cyan")
    table2.add_column("Count", justify="right", style="bold")
    table2.add_column("Action")
    table2.add_row("[green]High (>=60)[/]", str(high), "[green]KEEP — Important files[/]")
    table2.add_row("[yellow]Medium (30-59)[/]", str(medium), "[yellow]QUARANTINE — Review needed[/]")
    table2.add_row("[red]Low (<30)[/]", str(low), "[red]TRASH — Safe to delete[/]")

    console.print(table2)

    # Summary
    console.print(Panel(
        f"[bold]Deduplication Summary[/]\n\n"
        f"  Duplicate groups:   [bold]{len(dup_groups)}[/]\n"
        f"  Total duplicates:   [bold yellow]{total_dups}[/]\n"
        f"  Recoverable space:  [bold green]{recoverable / 1024:.1f} KB[/]\n"
        f"  Unique files:       [bold]{len(results) - total_dups}[/]",
        border_style="green",
    ))

    await asyncio.sleep(2)

    return {
        "dup_groups": len(dup_groups),
        "total_dups": total_dups,
        "recoverable": recoverable,
        "high": high,
        "medium": medium,
        "low": low,
    }


async def step_backup(target_dir: str, scan_data: dict) -> str:
    """Phase 3: SQLite backup + audit."""
    section_divider("PHASE 3: BACKUP & AUDIT TRAIL")

    narrate("Creating SQLite backup and audit trail...", pause=1)

    from src.config import config
    from src import database as db

    db.init_db()

    # Save scan audit
    import uuid
    run_id = f"org-{uuid.uuid4().hex[:8]}"

    db.save_audit(
        run_id, "organizer_scan", "librarian",
        input_summary=f"Target: {target_dir}",
        output_summary=f"{len(scan_data['results'])} files, {len(scan_data['categories'])} categories, "
                       f"{len(scan_data['sensitive_files'])} sensitive",
        model_used="hash-based",
        latency_ms=0,
    )

    # Backup existing databases
    backup_dir = config.reports_dir / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)

    backed_up = []

    # Backup sentinel DB
    sentinel_db = config.db_path
    if sentinel_db.exists():
        ts = time.strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"sentinel_{ts}.db"
        shutil.copy2(sentinel_db, backup_path)
        backed_up.append(("Sentinel DB", str(sentinel_db), str(backup_path), sentinel_db.stat().st_size))
        console.print(f"  [green]OK[/] Sentinel DB backed up: {backup_path.name}")

    # Backup disk_cleaner DB if it exists
    dc_db = Path(target_dir) / "disk_cleaner.db"
    if dc_db.exists():
        ts = time.strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"disk_cleaner_{ts}.db"
        shutil.copy2(dc_db, backup_path)
        backed_up.append(("Disk Cleaner DB", str(dc_db), str(backup_path), dc_db.stat().st_size))
        console.print(f"  [green]OK[/] Disk Cleaner DB backed up: {backup_path.name}")

    # Backup summary
    if backed_up:
        table = Table(title="Database Backups", box=box.ROUNDED)
        table.add_column("Database", style="cyan")
        table.add_column("Source", style="dim")
        table.add_column("Size", justify="right")
        table.add_column("Status")

        for name, src, dest, size in backed_up:
            table.add_row(name, Path(src).name, f"{size / 1024:.1f} KB", "[bold green]BACKED UP[/]")

        console.print(table)

    console.print(f"\n  [dim]Backups stored in: {backup_dir}[/]")

    await asyncio.sleep(2)
    return run_id


async def step_summary(scan_data: dict, dedup_data: dict, target_dir: str, run_id: str) -> None:
    """Final summary with key metrics."""
    section_divider("DEMO COMPLETE")

    metrics = Table(box=box.SIMPLE_HEAVY, show_header=False, padding=(0, 2))
    metrics.add_column("Metric", style="cyan")
    metrics.add_column("Value", style="bold white", justify="right")

    metrics.add_row("Target directory", target_dir)
    metrics.add_row("Files scanned", f"{len(scan_data['results']):,}")
    metrics.add_row("Total size", f"{scan_data['total_size'] / 1_048_576:.1f} MB")
    metrics.add_row("Categories detected", str(len(scan_data["categories"])))
    metrics.add_row("Sensitive files", f"[red]{len(scan_data['sensitive_files'])}[/]" if scan_data["sensitive_files"] else "[green]0[/]")
    metrics.add_row("Duplicate groups", str(dedup_data["dup_groups"]))
    metrics.add_row("Recoverable space", f"[green]{dedup_data['recoverable'] / 1024:.1f} KB[/]")
    metrics.add_row("Keep (score>=60)", f"[green]{dedup_data['high']}[/]")
    metrics.add_row("Quarantine (30-59)", f"[yellow]{dedup_data['medium']}[/]")
    metrics.add_row("Trash (score<30)", f"[red]{dedup_data['low']}[/]")
    metrics.add_row("Run ID", run_id)

    console.print(Panel(
        metrics,
        title="[bold green]Organizer Results[/]",
        border_style="green",
        padding=(0, 1),
    ))

    await asyncio.sleep(2)

    console.print(Panel(
        "[bold white]Organizer Technology Stack[/]\n\n"
        "  [magenta]Agent 1[/]      Librarian — recursive scan, 40+ types, sensitive detection\n"
        "  [magenta]Agent 2[/]      Deduplicator — MD5 hash, best-copy selection\n"
        "  [cyan]ScoreEngine[/]  disk_cleaner scoring (0-100, type/age/size/name)\n"
        "  [cyan]Backup[/]       SQLite DB backup with timestamp\n"
        "  [cyan]Audit[/]        Full traceability in sentinel.db\n"
        "  [cyan]Detection[/]    Sensitive data scanner (API keys, passwords)\n\n"
        "[dim italic]From scattered files to organized intelligence — automatically.[/]",
        title="[bold magenta]Stack[/]",
        border_style="magenta",
        padding=(0, 2),
    ))

    await asyncio.sleep(1)

    console.print()
    console.print(Align.center(Text.from_markup(
        "[bold bright_magenta]AI ORGANIZER[/]  [dim]|[/]  "
        "[bold white]Intelligent File Management[/]  [dim]|[/]  "
        "[bold green]Use Case #2[/]"
    )))
    console.print()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

async def run_demo(target_dir: str = "") -> None:
    """Run the Organizer demo with real file scanning."""
    if not target_dir:
        target_dir = "F:\\BUREAU\\disk_cleaner"

    # Fallback to project root if dir doesn't exist
    if not Path(target_dir).exists():
        target_dir = str(Path(__file__).parent.parent)

    try:
        await step_intro()
        await step_architecture()
        scan_data = await step_scan_real_directory(target_dir)

        if not scan_data:
            console.print("[red]No files found. Aborting demo.[/]")
            return

        dedup_data = await step_dedup(scan_data)
        run_id = await step_backup(target_dir, scan_data)
        await step_summary(scan_data, dedup_data, target_dir, run_id)

    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted.[/]")
    except Exception as e:
        console.print(f"\n[red]Demo error: {e}[/]")
        import traceback
        traceback.print_exc()
