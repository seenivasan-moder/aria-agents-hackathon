"""Agent Deduplicator — File deduplication and space recovery.

Identifies duplicate files by content hash, suggests keep/delete decisions,
and calculates recoverable disk space.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.table import Table

from src.agents.organizer.librarian import FileInfo
from src import database as db

console = Console()


@dataclass
class DuplicateGroup:
    content_hash: str
    files: list[FileInfo]
    total_size: int = 0
    recoverable_size: int = 0
    keep_path: str = ""  # recommended file to keep
    reason: str = ""

    def __post_init__(self):
        self.total_size = sum(f.size_bytes for f in self.files)
        self.recoverable_size = self.total_size - (self.files[0].size_bytes if self.files else 0)


@dataclass
class DeduplicationReport:
    total_files_scanned: int
    duplicate_groups: list[DuplicateGroup]
    total_duplicates: int
    recoverable_bytes: int
    unique_files: int


def _select_best_copy(files: list[FileInfo]) -> tuple[str, str]:
    """Select the best copy to keep based on heuristics."""
    # Prefer: most recent > shortest path > largest file
    sorted_files = sorted(files, key=lambda f: (
        -f.modified_time,  # most recent first
        len(f.path),       # shortest path
        -f.size_bytes,     # largest first (might have more metadata)
    ))
    best = sorted_files[0]
    reason = "Most recent version with shortest path"
    return best.path, reason


def find_duplicates(files: list[FileInfo], min_size: int = 1024) -> DeduplicationReport:
    """Find duplicate files by content hash.

    Args:
        files: List of FileInfo from librarian scan
        min_size: Minimum file size to consider (skip tiny files)

    Returns:
        DeduplicationReport with duplicate groups and stats
    """
    # Group by hash
    hash_groups: dict[str, list[FileInfo]] = defaultdict(list)
    for f in files:
        if f.content_hash and f.size_bytes >= min_size:
            hash_groups[f.content_hash].append(f)

    # Find groups with > 1 file
    duplicate_groups = []
    for content_hash, group_files in hash_groups.items():
        if len(group_files) > 1:
            keep_path, reason = _select_best_copy(group_files)
            dg = DuplicateGroup(
                content_hash=content_hash,
                files=group_files,
                keep_path=keep_path,
                reason=reason,
            )
            duplicate_groups.append(dg)

    # Sort by recoverable space (largest savings first)
    duplicate_groups.sort(key=lambda g: g.recoverable_size, reverse=True)

    total_duplicates = sum(len(g.files) - 1 for g in duplicate_groups)
    recoverable = sum(g.recoverable_size for g in duplicate_groups)
    unique = len(files) - total_duplicates

    return DeduplicationReport(
        total_files_scanned=len(files),
        duplicate_groups=duplicate_groups,
        total_duplicates=total_duplicates,
        recoverable_bytes=recoverable,
        unique_files=unique,
    )


async def run(run_id: str, files: list[FileInfo]) -> DeduplicationReport:
    """Execute Agent Deduplicator: find duplicates and calculate savings."""
    t0 = time.monotonic()
    console.print("[bold magenta]Agent Deduplicator[/] analyzing file hashes...")

    report = find_duplicates(files)
    latency = (time.monotonic() - t0) * 1000

    db.save_audit(
        run_id, "deduplication", "dedup_agent",
        input_summary=f"{len(files)} files to analyze",
        output_summary=f"{report.total_duplicates} duplicates in {len(report.duplicate_groups)} groups, "
                       f"recoverable: {report.recoverable_bytes / 1_048_576:.1f} MB",
        model_used="hash-based",
        latency_ms=latency,
    )

    _print_report(report)
    console.print(f"[green]Deduplicator done[/] — {report.recoverable_bytes / 1_048_576:.1f} MB recoverable in {int(latency)}ms")
    return report


def _print_report(report: DeduplicationReport) -> None:
    table = Table(title="Deduplication Report", show_lines=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Files Scanned", f"{report.total_files_scanned:,}")
    table.add_row("Unique Files", f"{report.unique_files:,}")
    table.add_row("Duplicate Files", f"[yellow]{report.total_duplicates:,}[/]")
    table.add_row("Duplicate Groups", str(len(report.duplicate_groups)))
    table.add_row("Recoverable Space", f"[bold green]{report.recoverable_bytes / 1_048_576:.1f} MB[/]")
    console.print(table)

    if report.duplicate_groups:
        console.print("\n[bold]Top Duplicate Groups:[/]")
        for i, g in enumerate(report.duplicate_groups[:5]):
            console.print(
                f"  {i+1}. [yellow]{len(g.files)} copies[/] — "
                f"{g.recoverable_size / 1024:.0f} KB recoverable — "
                f"keep: [green]{g.keep_path}[/]"
            )
            for f in g.files:
                marker = "[green]KEEP[/]" if f.path == g.keep_path else "[red]DELETE[/]"
                console.print(f"     {marker} {f.path} ({f.size_bytes / 1024:.0f} KB)")
