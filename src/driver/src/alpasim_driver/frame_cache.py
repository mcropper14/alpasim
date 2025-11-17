"""Session-specific helpers for async VAM driver."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import wraps
from threading import RLock
from typing import List, Optional

import numpy as np
import torch


@dataclass
class FrameEntry:
    """Represents a single camera frame and its cached tokenization."""

    timestamp_us: int
    image: Optional[np.ndarray]
    tokens: Optional[torch.Tensor]


def synchronized(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


@dataclass
class FrameCache:
    """Keeps a bounded, time-ordered buffer of frames and tokens for a session."""

    context_length: int
    entries: List[FrameEntry] = field(default_factory=list)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    @synchronized
    def add_image(self, timestamp_us: int, image: np.ndarray) -> None:
        """Insert or replace an image while keeping entries ordered by timestamp."""
        inserted = False
        # Iterate from newest to oldest since most inserts append.
        for offset, entry in enumerate(reversed(self.entries)):
            if entry.timestamp_us == timestamp_us:
                raise ValueError(f"Frame {timestamp_us} already exists in cache")
            if entry.timestamp_us < timestamp_us:
                insert_at = len(self.entries) - offset
                self.entries.insert(insert_at, FrameEntry(timestamp_us, image, None))
                inserted = True
                break
        if not inserted:
            self.entries.insert(0, FrameEntry(timestamp_us, image, None))

        self.prune()

    @synchronized
    def pending_frames(self) -> List[FrameEntry]:
        """Return frames that still need tokenization in timestamp order."""
        return [
            entry
            for entry in self.entries
            if entry.image is not None and entry.tokens is None
        ]

    @synchronized
    def frame_count(self) -> int:
        """Total number of frames currently cached."""
        return len(self.entries)

    @synchronized
    def token_count(self) -> int:
        """Number of frames that already have cached tokens."""
        return sum(1 for entry in self.entries if entry.tokens is not None)

    @synchronized
    def latest_token_window(self) -> List[torch.Tensor]:
        """Return the newest context window of tokens (oldest first)."""
        tokens = [entry.tokens for entry in self.entries]
        if any(token is None for token in tokens):
            raise ValueError("Insufficient tokens: some tokens are None")
        if len(tokens) < self.context_length:
            raise ValueError(
                f"Insufficient tokens: have {len(tokens)}, need {self.context_length}"
            )
        assert len(tokens) == self.context_length
        return tokens

    @synchronized
    def prune(self) -> None:
        """Bound the cache to the configured temporal context length."""
        excess = len(self.entries) - self.context_length
        if excess <= 0:
            return
        del self.entries[:excess]

    @synchronized
    def reset(self) -> None:
        """Clear all cached frames (e.g., when resetting the session)."""
        self.entries.clear()
