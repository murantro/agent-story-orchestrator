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

"""Persistent in-process world state manager.

Maintains NPC registry, event queue, game clock, and memory store
across requests. For singleplayer, this lives in-process. For
multiplayer, it would be backed by PostgreSQL + Redis.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.config import (
    DEFAULT_TICK_DELTA_HOURS,
    EMOTION_DECAY_RATE,
    EVENT_IMPACT_SCALE,
    INITIAL_GAME_TIME,
    MAX_NPCS,
    MAX_RECENT_MEMORIES,
)
from app.events.event_queue import EventQueue
from app.events.propagation import EventPropagator
from app.memory.base import MemoryEntry
from app.memory.in_memory_store import InMemoryStore
from app.models.events import WorldEvent
from app.models.locations import LocationGraph
from app.models.npc_status import NPCVectorialStatus
from app.simulation.emotion_engine import EmotionEngine
from app.simulation.environment_engine import EnvironmentEngine
from app.simulation.intention_engine import IntentionEngine
from app.simulation.interaction_engine import InteractionEngine
from app.simulation.movement_engine import MovementEngine
from app.simulation.relationship_engine import RelationshipEngine
from app.simulation.social_engine import SocialInfluenceEngine
from app.simulation.vitality_engine import VitalityEngine


@dataclass
class TickResult:
    """Summary of a simulation tick.

    Attributes:
        game_time: In-game time after the tick.
        npcs_updated: Number of NPCs whose vectors were updated.
        events_delivered: Number of events delivered this tick.
        events_pending: Number of events still in the queue.
    """

    game_time: float = 0.0
    npcs_updated: int = 0
    events_delivered: int = 0
    events_pending: int = 0
    interactions_resolved: int = 0
    npcs_moved: int = 0


class WorldStateManager:
    """Persistent in-process world state.

    Maintains NPC registry, event queue, game clock, and memory store.
    Uses an asyncio lock for safe concurrent access from FastAPI handlers.

    Args:
        game_time: Initial in-game time (hours since epoch).
        max_npcs: Maximum number of NPCs allowed.
        decay_rate: Emotion decay rate per tick.
        impact_scale: Global event impact multiplier.
    """

    def __init__(
        self,
        game_time: float = INITIAL_GAME_TIME,
        max_npcs: int = MAX_NPCS,
        decay_rate: float = EMOTION_DECAY_RATE,
        impact_scale: float = EVENT_IMPACT_SCALE,
    ):
        self._npcs: dict[str, NPCVectorialStatus] = {}
        self._event_queue = EventQueue()
        self._propagator = EventPropagator(self._event_queue)
        self._memory_store = InMemoryStore()
        self._emotion_engine = EmotionEngine(
            decay_rate=decay_rate, impact_scale=impact_scale
        )
        self._intention_engine = IntentionEngine()
        self._interaction_engine = InteractionEngine()
        self._relationship_engine = RelationshipEngine()
        self._environment_engine = EnvironmentEngine()
        self._movement_engine = MovementEngine()
        self._vitality_engine = VitalityEngine()
        self._social_engine = SocialInfluenceEngine()
        self._location_graph = LocationGraph()
        self._game_time = game_time
        self._max_npcs = max_npcs
        self._lock = asyncio.Lock()

    @property
    def game_time(self) -> float:
        return self._game_time

    @property
    def npc_count(self) -> int:
        return len(self._npcs)

    @property
    def memory_store(self) -> InMemoryStore:
        return self._memory_store

    @property
    def location_graph(self) -> LocationGraph:
        return self._location_graph

    # --- NPC CRUD ---

    def add_npc(self, npc: NPCVectorialStatus) -> None:
        """Register an NPC in the world.

        Args:
            npc: The NPC to add.

        Raises:
            ValueError: If max NPCs exceeded or NPC ID already exists.
        """
        if len(self._npcs) >= self._max_npcs:
            raise ValueError(f"Maximum NPC count ({self._max_npcs}) reached.")
        if npc.npc_id in self._npcs:
            raise ValueError(f"NPC with ID {npc.npc_id!r} already exists.")
        self._npcs[npc.npc_id] = npc

    def get_npc(self, npc_id: str) -> NPCVectorialStatus | None:
        """Get an NPC by ID, or None if not found."""
        return self._npcs.get(npc_id)

    def list_npcs(self, location_id: str | None = None) -> list[NPCVectorialStatus]:
        """List all NPCs, optionally filtered by location.

        Args:
            location_id: If provided, only return NPCs at this location.

        Returns:
            List of NPCs.
        """
        npcs = list(self._npcs.values())
        if location_id is not None:
            npcs = [n for n in npcs if n.location_id == location_id]
        return npcs

    def remove_npc(self, npc_id: str) -> bool:
        """Remove an NPC from the world.

        Args:
            npc_id: ID of the NPC to remove.

        Returns:
            True if removed, False if not found.
        """
        if npc_id in self._npcs:
            del self._npcs[npc_id]
            return True
        return False

    # --- Events ---

    def submit_event(self, event: WorldEvent) -> int:
        """Submit a world event and schedule its propagation cascade.

        Args:
            event: The event to submit. Its timestamp is set to current game_time
                   if not already set.

        Returns:
            Number of scheduled deliveries (including original).
        """
        if event.timestamp == 0.0:
            event.timestamp = self._game_time
        return self._propagator.submit(event)

    def get_due_events(self) -> list[WorldEvent]:
        """Pop all events due at the current game time.

        Returns:
            List of due events.
        """
        return self._event_queue.pop_due(self._game_time)

    # --- Memory ---

    async def form_memory_from_event(
        self, event: WorldEvent, npc: NPCVectorialStatus
    ) -> None:
        """Create a memory entry when an NPC witnesses an event.

        Stores event description and emotional valence in the memory store.
        Uses a zero embedding (will be replaced by sentence-transformers later).

        Args:
            event: The world event witnessed.
            npc: The NPC forming the memory.
        """
        if not event.description:
            return
        entry = MemoryEntry(
            npc_id=npc.npc_id,
            event_text=event.description,
            importance=event.intensity,
            emotional_valence=float(np.mean(event.emotion_impact)),
            game_timestamp=self._game_time,
            location_id=event.location_id,
        )
        await self._memory_store.store(entry)
        npc.recent_memories.append(event.description)
        if len(npc.recent_memories) > MAX_RECENT_MEMORIES:
            npc.recent_memories = npc.recent_memories[-MAX_RECENT_MEMORIES:]

    # --- Simulation ---

    async def tick(self, delta_hours: float = DEFAULT_TICK_DELTA_HOURS) -> TickResult:
        """Advance the simulation by delta_hours.

        Runs the full pipeline:
          1. Advance clock
          2. Deliver due events
          3. Apply event impacts to emotions, health/energy, social + memories
          4. Decay emotions toward baseline
          5. Recompute intentions (biased by energy/health)
          6. Resolve autonomous NPC interactions
          7. Apply relationship updates, vitality costs + submit interaction events
          8. Decay weak relationships
          9. Process NPC movement (arrivals + new departures)
          10. Update NPC environment vectors from locations
          11. Apply per-tick vitality dynamics (energy drain/regen, health)
          12. Apply per-tick social influence (peer pressure, decay)

        Args:
            delta_hours: In-game hours to advance.

        Returns:
            TickResult with summary.
        """
        async with self._lock:
            self._game_time += delta_hours

            # 1. Deliver due events
            due_events = self._event_queue.pop_due(self._game_time)

            # 2. Get all NPCs as a list for batch processing
            npcs = list(self._npcs.values())

            interactions_resolved = 0
            npcs_moved = 0

            if npcs:
                # 3. Apply event impacts to emotions, health/energy, social
                #    + form memories
                for event in due_events:
                    self._emotion_engine.apply_event_batch(npcs, event)
                    self._vitality_engine.apply_event_batch(npcs, event)
                    self._social_engine.apply_event_batch(npcs, event)
                    for npc in npcs:
                        await self.form_memory_from_event(event, npc)

                # 4. Decay emotions toward baseline
                self._emotion_engine.tick(npcs)

                # 5. Recompute intentions (now biased by energy/health)
                self._intention_engine.tick(npcs)

                # 6. Resolve autonomous NPC interactions
                outcomes = self._interaction_engine.tick(npcs, self._game_time)
                interactions_resolved = len(outcomes)

                # 7. Apply relationship deltas, vitality costs + submit events
                for outcome in outcomes:
                    npc_a = self._npcs.get(outcome.npc_a_id)
                    npc_b = self._npcs.get(outcome.npc_b_id)
                    if npc_a and npc_b:
                        self._relationship_engine.apply_delta(
                            npc_a, npc_b, outcome.relationship_delta
                        )
                        # Apply energy and health costs from interaction
                        self._vitality_engine.apply_interaction_costs(
                            npc_a,
                            npc_b,
                            outcome.energy_cost,
                            outcome.health_delta_a,
                            outcome.health_delta_b,
                        )
                        # Form memories for both participants
                        await self.form_memory_from_event(outcome.event, npc_a)
                        await self.form_memory_from_event(outcome.event, npc_b)
                    # Submit event for propagation
                    self.submit_event(outcome.event)

                # 8. Decay weak relationships
                self._relationship_engine.decay(npcs)

                # 9. Process NPC movement (arrivals + new departures)
                new_journeys = self._movement_engine.tick(
                    npcs, self._location_graph, self._game_time
                )
                npcs_moved = len(new_journeys)

                # 10. Update NPC environment vectors from locations
                self._environment_engine.tick(npcs, self._location_graph)

                # 11. Apply per-tick vitality dynamics (drain, regen, health)
                self._vitality_engine.tick(npcs)

                # 12. Apply per-tick social influence (peer pressure, decay)
                self._social_engine.tick(npcs)

            return TickResult(
                game_time=self._game_time,
                npcs_updated=len(npcs),
                events_delivered=len(due_events),
                events_pending=len(self._event_queue),
                interactions_resolved=interactions_resolved,
                npcs_moved=npcs_moved,
            )

    # --- Serialization ---

    def snapshot(self) -> dict[str, Any]:
        """Create a JSON-serializable snapshot of the entire world state.

        Returns:
            Dict containing all world state data.
        """
        from app.agents.serialization import serialize_npc

        return {
            "game_time": self._game_time,
            "npcs": {npc_id: serialize_npc(npc) for npc_id, npc in self._npcs.items()},
            "locations": self._location_graph.to_dict(),
        }

    def restore(self, data: dict[str, Any]) -> None:
        """Restore world state from a snapshot.

        Args:
            data: Dict previously produced by snapshot().
        """
        from app.agents.serialization import deserialize_npc

        self._game_time = data["game_time"]
        self._npcs.clear()
        for npc_id, npc_data in data["npcs"].items():
            self._npcs[npc_id] = deserialize_npc(npc_data)
        self._event_queue = EventQueue()
        self._propagator = EventPropagator(self._event_queue)
        if "locations" in data:
            self._location_graph = LocationGraph.from_dict(data["locations"])
