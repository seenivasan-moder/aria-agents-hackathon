"""Agent Librarian — Intelligent file scanning, classification and metadata extraction.

Scans target directories, extracts metadata, classifies files using AI,
and detects sensitive content (API keys, credentials, personal data).
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.progress import Progress

from src.config import config
from src import database as db

console = Console()


class FileCategory(str, Enum):
    CODE = "code"
    DOCUMENT = "document"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    DATA = "data"
    ARCHIVE = "archive"
    CONFIG = "config"
    CACHE = "cache"
    UNKNOWN = "unknown"


class SensitivityLevel(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"  # API keys, passwords, private keys


CATEGORY_MAP: dict[str, FileCategory] = {
    # Code
    ".py": FileCategory.CODE, ".js": FileCategory.CODE, ".ts": FileCategory.CODE,
    ".java": FileCategory.CODE, ".cpp": FileCategory.CODE, ".c": FileCategory.CODE,
    ".rs": FileCategory.CODE, ".go": FileCategory.CODE, ".rb": FileCategory.CODE,
    ".php": FileCategory.CODE, ".cs": FileCategory.CODE, ".swift": FileCategory.CODE,
    ".sh": FileCategory.CODE, ".bat": FileCategory.CODE, ".ps1": FileCategory.CODE,
    # Documents
    ".pdf": FileCategory.DOCUMENT, ".doc": FileCategory.DOCUMENT, ".docx": FileCategory.DOCUMENT,
    ".txt": FileCategory.DOCUMENT, ".md": FileCategory.DOCUMENT, ".rtf": FileCategory.DOCUMENT,
    ".xlsx": FileCategory.DOCUMENT, ".csv": FileCategory.DOCUMENT,
    # Images
    ".png": FileCategory.IMAGE, ".jpg": FileCategory.IMAGE, ".jpeg": FileCategory.IMAGE,
    ".gif": FileCategory.IMAGE, ".svg": FileCategory.IMAGE, ".webp": FileCategory.IMAGE,
    # Video
    ".mp4": FileCategory.VIDEO, ".avi": FileCategory.VIDEO, ".mkv": FileCategory.VIDEO,
    ".mov": FileCategory.VIDEO, ".webm": FileCategory.VIDEO,
    # Audio
    ".mp3": FileCategory.AUDIO, ".wav": FileCategory.AUDIO, ".flac": FileCategory.AUDIO,
    # Data
    ".json": FileCategory.DATA, ".xml": FileCategory.DATA, ".yaml": FileCategory.DATA,
    ".yml": FileCategory.DATA, ".toml": FileCategory.DATA, ".sql": FileCategory.DATA,
    ".db": FileCategory.DATA, ".sqlite": FileCategory.DATA,
    # Archives
    ".zip": FileCategory.ARCHIVE, ".tar": FileCategory.ARCHIVE, ".gz": FileCategory.ARCHIVE,
    ".7z": FileCategory.ARCHIVE, ".rar": FileCategory.ARCHIVE,
    # Config
    ".env": FileCategory.CONFIG, ".ini": FileCategory.CONFIG, ".cfg": FileCategory.CONFIG,
    # Cache
    ".pyc": FileCategory.CACHE, ".o": FileCategory.CACHE, ".class": FileCategory.CACHE,
}

SENSITIVE_PATTERNS = [
    "api_key", "apikey", "api-key", "secret_key", "secretkey",
    "password", "passwd", "private_key", "privatekey",
    "token", "bearer", "authorization",
    "-----BEGIN", "ssh-rsa", "ssh-ed25519",
    "AKIA",  # AWS access key prefix
]


@dataclass
class FileInfo:
    path: str
    name: str
    extension: str
    size_bytes: int
    modified_time: float
    category: FileCategory = FileCategory.UNKNOWN
    sensitivity: SensitivityLevel = SensitivityLevel.PUBLIC
    content_hash: str = ""
    project_group: str = ""
    flags: list[str] = field(default_factory=list)


def _hash_file(path: str, chunk_size: int = 8192) -> str:
    """Compute MD5 hash of file content (first 1MB for speed)."""
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            total = 0
            while chunk := f.read(chunk_size):
                h.update(chunk)
                total += len(chunk)
                if total > 1_048_576:  # 1MB limit for speed
                    break
        return h.hexdigest()
    except (OSError, PermissionError):
        return ""


def _detect_sensitivity(path: str, ext: str) -> tuple[SensitivityLevel, list[str]]:
    """Detect if a file contains sensitive data."""
    flags = []
    name_lower = os.path.basename(path).lower()

    # Filename-based detection
    if name_lower in (".env", ".env.local", "credentials.json", "secrets.yaml", "id_rsa", "id_ed25519"):
        return SensitivityLevel.SECRET, ["Sensitive filename detected"]

    # Content-based detection for text files
    if ext in (".py", ".js", ".ts", ".json", ".yaml", ".yml", ".toml", ".env", ".txt", ".cfg", ".ini", ".md"):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read(4096)  # first 4KB
            content_lower = content.lower()
            for pattern in SENSITIVE_PATTERNS:
                if pattern in content_lower:
                    flags.append(f"Contains '{pattern}'")
            if flags:
                return SensitivityLevel.SECRET, flags
        except (OSError, PermissionError):
            pass

    return SensitivityLevel.PUBLIC, []


def _classify_file(path: str) -> FileInfo:
    """Classify a single file: category, sensitivity, metadata."""
    p = Path(path)
    ext = p.suffix.lower()
    stat = p.stat()

    category = CATEGORY_MAP.get(ext, FileCategory.UNKNOWN)
    sensitivity, flags = _detect_sensitivity(path, ext)

    # Detect project group from parent directory
    parts = p.parts
    project_group = ""
    for keyword in ["src", "lib", "app", "test", "docs", "data", "build", "dist"]:
        if keyword in parts:
            idx = parts.index(keyword)
            if idx > 0:
                project_group = parts[idx - 1]
                break
    if not project_group and len(parts) > 2:
        project_group = parts[-3] if len(parts) > 3 else parts[-2]

    return FileInfo(
        path=str(p),
        name=p.name,
        extension=ext,
        size_bytes=stat.st_size,
        modified_time=stat.st_mtime,
        category=category,
        sensitivity=sensitivity,
        content_hash=_hash_file(path) if stat.st_size < 100_000_000 else "",  # skip >100MB
        project_group=project_group,
        flags=flags,
    )


def scan_directory(
    target_dir: str,
    max_files: int = 10000,
    skip_hidden: bool = True,
    skip_patterns: list[str] | None = None,
) -> list[FileInfo]:
    """Scan a directory and classify all files."""
    skip_patterns = skip_patterns or [".git", "node_modules", "__pycache__", ".venv", ".tox"]
    results = []
    count = 0

    for root, dirs, files in os.walk(target_dir):
        # Skip hidden and excluded directories
        if skip_hidden:
            dirs[:] = [d for d in dirs if not d.startswith(".")]
        dirs[:] = [d for d in dirs if d not in skip_patterns]

        for fname in files:
            if count >= max_files:
                break
            if skip_hidden and fname.startswith("."):
                continue

            fpath = os.path.join(root, fname)
            try:
                info = _classify_file(fpath)
                results.append(info)
                count += 1
            except (OSError, PermissionError):
                continue

    return results


async def run(run_id: str, target_dir: str, max_files: int = 5000) -> list[FileInfo]:
    """Execute Agent Librarian: scan and classify files."""
    t0 = time.monotonic()
    console.print(f"[bold magenta]Agent Librarian[/] scanning {target_dir}...")

    files = scan_directory(target_dir, max_files=max_files)
    latency = (time.monotonic() - t0) * 1000

    # Stats
    total_size = sum(f.size_bytes for f in files)
    categories = {}
    for f in files:
        categories[f.category.value] = categories.get(f.category.value, 0) + 1
    sensitive = [f for f in files if f.sensitivity in (SensitivityLevel.CONFIDENTIAL, SensitivityLevel.SECRET)]

    db.save_audit(
        run_id, "file_scan", "librarian",
        input_summary=f"Scanned {target_dir}",
        output_summary=f"{len(files)} files, {total_size / 1_048_576:.0f} MB, "
                       f"{len(sensitive)} sensitive, categories: {categories}",
        model_used="rule-based",
        latency_ms=latency,
    )

    _print_summary(files, total_size, categories, sensitive)
    console.print(f"[green]Librarian done[/] — {len(files)} files in {int(latency)}ms")
    return files


def _print_summary(files: list[FileInfo], total_size: int, categories: dict, sensitive: list) -> None:
    table = Table(title="File Scan Summary", show_lines=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Total Files", f"{len(files):,}")
    table.add_row("Total Size", f"{total_size / 1_048_576:.1f} MB")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        table.add_row(f"  {cat}", str(count))
    table.add_row("Sensitive Files", f"[red]{len(sensitive)}[/]" if sensitive else "[green]0[/]")
    console.print(table)

    if sensitive:
        console.print("\n[bold red]Sensitive Files Detected:[/]")
        for f in sensitive[:10]:
            console.print(f"  [red]![/] {f.path} — {', '.join(f.flags)}")
