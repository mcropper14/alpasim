import numpy as np
import pytest
import torch

from ..frame_cache import FrameCache


def _make_image(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(2, 2, 3), dtype=np.uint8)


def _make_tokens(seed: int) -> torch.Tensor:
    rng = torch.Generator().manual_seed(seed)
    return torch.randint(0, 10, (3, 3), generator=rng, dtype=torch.int32)


def test_add_image_orders_and_prunes() -> None:
    cache = FrameCache(context_length=3)
    cache.add_image(30, _make_image(30))
    cache.add_image(10, _make_image(10))
    cache.add_image(20, _make_image(20))

    assert [entry.timestamp_us for entry in cache.entries] == [10, 20, 30]
    assert cache.frame_count() == 3
    assert cache.token_count() == 0

    cache.add_image(25, _make_image(25))
    assert [entry.timestamp_us for entry in cache.entries] == [20, 25, 30]
    assert cache.frame_count() == 3


def test_pending_frames_update_when_tokens_assigned() -> None:
    cache = FrameCache(context_length=3)
    cache.add_image(1, _make_image(1))
    cache.add_image(2, _make_image(2))

    pending = cache.pending_frames()
    assert len(pending) == 2

    pending[0].tokens = _make_tokens(1)

    pending = cache.pending_frames()
    assert len(pending) == 1
    assert pending[0].timestamp_us == 2

    pending[0].tokens = _make_tokens(2)

    pending = cache.pending_frames()
    assert len(pending) == 0
    assert all(entry.image is not None for entry in cache.entries)
    assert all(entry.tokens is not None for entry in cache.entries)
    assert cache.token_count() == 2


def test_latest_token_window_requires_full_context() -> None:
    cache = FrameCache(context_length=2)
    cache.add_image(1, _make_image(1))
    cache.add_image(2, _make_image(2))

    first_pending = cache.pending_frames()[0]
    first_pending.tokens = _make_tokens(1)

    with pytest.raises(ValueError):
        cache.latest_token_window()

    cache.pending_frames()[0].tokens = _make_tokens(2)
    window = cache.latest_token_window()
    assert len(window) == 2


def test_add_image_rejects_duplicate_timestamp() -> None:
    cache = FrameCache(context_length=2)
    cache.add_image(5, _make_image(5))

    with pytest.raises(ValueError):
        cache.add_image(5, _make_image(5))


def test_reset_clears_all_entries() -> None:
    cache = FrameCache(context_length=3)
    cache.add_image(1, _make_image(1))
    cache.add_image(2, _make_image(2))
    pending = cache.pending_frames()
    pending[0].tokens = _make_tokens(1)

    assert cache.frame_count() == 2
    assert cache.token_count() == 1

    cache.reset()

    assert cache.frame_count() == 0
    assert cache.token_count() == 0
    assert cache.entries == []


def test_pending_frames_order_after_prune() -> None:
    cache = FrameCache(context_length=3)
    cache.add_image(10, _make_image(10))
    cache.add_image(20, _make_image(20))
    cache.add_image(30, _make_image(30))

    # Tokenize frames that have already been processed to leave only the newest pending.
    oldest_entry = cache.pending_frames()[0]
    oldest_entry.tokens = _make_tokens(10)
    next_entry = cache.pending_frames()[0]
    next_entry.tokens = _make_tokens(20)

    cache.add_image(40, _make_image(40))

    pending = cache.pending_frames()
    assert [entry.timestamp_us for entry in pending] == [30, 40]
