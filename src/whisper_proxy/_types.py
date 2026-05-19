"""Shared data types with no heavy dependencies."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Word:
    token: str
    start_sec: float
    end_sec: float
