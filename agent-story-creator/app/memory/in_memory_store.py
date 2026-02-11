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

"""In-memory implementation of MemoryStore for testing and prototyping.

Uses brute-force cosine similarity. Not suitable for production with
large memory counts — use QdrantStore or PgVectorStore instead.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .base import MemoryEntry, MemoryStore


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class InMemoryStore(MemoryStore):
    """Dict-based memory store with brute-force cosine similarity search."""

    def __init__(self) -> None:
        self._memories: dict[str, list[MemoryEntry]] = defaultdict(list)

    async def store(self, entry: MemoryEntry) -> None:
        self._memories[entry.npc_id].append(entry)

    async def search(
        self,
        npc_id: str,
        query_embedding: np.ndarray,
        limit: int = 5,
    ) -> list[MemoryEntry]:
        entries = self._memories.get(npc_id, [])
        if not entries:
            return []

        scored = [
            (entry, _cosine_similarity(query_embedding, entry.embedding))
            for entry in entries
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [entry for entry, _ in scored[:limit]]

    async def get_recent(
        self,
        npc_id: str,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        entries = self._memories.get(npc_id, [])
        sorted_entries = sorted(entries, key=lambda e: e.game_timestamp, reverse=True)
        return sorted_entries[:limit]
