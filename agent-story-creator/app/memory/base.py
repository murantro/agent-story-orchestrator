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

"""Abstract memory store interface.

Implementations:
  - InMemoryStore: simple dict-based store (testing, prototyping)
  - QdrantStore: Qdrant-backed vector search (singleplayer, future)
  - PgVectorStore: PostgreSQL pgvector (multiplayer, future)
"""

from __future__ import annotations

import abc
import uuid
from dataclasses import dataclass, field

import numpy as np


@dataclass
class MemoryEntry:
    """A single memory stored in the vector DB.

    Attributes:
        memory_id: Unique identifier.
        npc_id: Owning NPC.
        event_text: Human-readable description.
        embedding: Dense vector from a sentence embedding model.
        importance: How significant this memory is (0-1).
        emotional_valence: How positive/negative this memory feels (-1 to 1).
        game_timestamp: In-game time when this memory was formed.
        location_id: Where this memory was formed.
    """

    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    npc_id: str = ""
    event_text: str = ""
    embedding: np.ndarray = field(
        default_factory=lambda: np.zeros(384, dtype=np.float32)
    )
    importance: float = 0.5
    emotional_valence: float = 0.0
    game_timestamp: float = 0.0
    location_id: str = "default"


class MemoryStore(abc.ABC):
    """Abstract interface for NPC memory storage and retrieval."""

    @abc.abstractmethod
    async def store(self, entry: MemoryEntry) -> None:
        """Store a memory entry.

        Args:
            entry: The memory to store.
        """

    @abc.abstractmethod
    async def search(
        self,
        npc_id: str,
        query_embedding: np.ndarray,
        limit: int = 5,
    ) -> list[MemoryEntry]:
        """Search for memories relevant to a query.

        Args:
            npc_id: Only search this NPC's memories.
            query_embedding: The query vector to match against.
            limit: Maximum number of results.

        Returns:
            List of matching memories, ordered by relevance.
        """

    @abc.abstractmethod
    async def get_recent(
        self,
        npc_id: str,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        """Get the most recent memories for an NPC.

        Args:
            npc_id: The NPC whose memories to retrieve.
            limit: Maximum number of results.

        Returns:
            List of memories ordered by recency (newest first).
        """
