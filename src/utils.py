"""Утилиты проекта."""

from __future__ import annotations

from pathlib import Path


def ensure_dir(path: str | Path, create: bool = False) -> Path:
    path = Path(path)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def log_info(msg: str) -> None:
    print(f"[INFO] {msg}")


def log_ok(msg: str) -> None:
    print(f"[OK]   {msg}")


def log_warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def log_error(msg: str) -> None:
    print(f"[ERROR] {msg}")
