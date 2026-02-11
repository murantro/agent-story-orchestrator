# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the in-memory store."""

import numpy as np
import pytest

from app.memory.base import MemoryEntry
from app.memory.in_memory_store import InMemoryStore


def _make_entry(
    npc_id: str = "npc1",
    text: str = "something happened",
    embedding: np.ndarray | None = None,
    timestamp: float = 0.0,
) -> MemoryEntry:
    if embedding is None:
        embedding = np.random.rand(384).astype(np.float32)
    return MemoryEntry(
        npc_id=npc_id,
        event_text=text,
        embedding=embedding,
        game_timestamp=timestamp,
    )


@pytest.mark.asyncio
async def test_store_and_search():
    """Stored memories should be retrievable via search."""
    store = InMemoryStore()
    target_embedding = np.random.rand(384).astype(np.float32)
    target_embedding /= np.linalg.norm(target_embedding)

    # Store one matching memory and one unrelated
    await store.store(
        _make_entry(npc_id="npc1", text="relevant", embedding=target_embedding)
    )
    random_embedding = np.random.rand(384).astype(np.float32)
    await store.store(
        _make_entry(npc_id="npc1", text="unrelated", embedding=random_embedding)
    )

    results = await store.search("npc1", target_embedding, limit=1)
    assert len(results) == 1
    assert results[0].event_text == "relevant"


@pytest.mark.asyncio
async def test_search_filters_by_npc_id():
    """Search should only return memories for the specified NPC."""
    store = InMemoryStore()
    embedding = np.ones(384, dtype=np.float32)
    await store.store(
        _make_entry(npc_id="npc1", text="npc1 memory", embedding=embedding)
    )
    await store.store(
        _make_entry(npc_id="npc2", text="npc2 memory", embedding=embedding)
    )

    results = await store.search("npc1", embedding, limit=10)
    assert all(r.npc_id == "npc1" for r in results)
    assert len(results) == 1


@pytest.mark.asyncio
async def test_get_recent_returns_newest_first():
    """get_recent should return memories ordered by timestamp, newest first."""
    store = InMemoryStore()
    await store.store(_make_entry(npc_id="npc1", text="old", timestamp=1.0))
    await store.store(_make_entry(npc_id="npc1", text="newer", timestamp=5.0))
    await store.store(_make_entry(npc_id="npc1", text="newest", timestamp=10.0))

    results = await store.get_recent("npc1", limit=2)
    assert len(results) == 2
    assert results[0].event_text == "newest"
    assert results[1].event_text == "newer"


@pytest.mark.asyncio
async def test_search_empty_store():
    """Search on an empty store should return empty list."""
    store = InMemoryStore()
    results = await store.search("npc1", np.zeros(384, dtype=np.float32))
    assert results == []


@pytest.mark.asyncio
async def test_get_recent_empty_store():
    store = InMemoryStore()
    results = await store.get_recent("npc1")
    assert results == []
