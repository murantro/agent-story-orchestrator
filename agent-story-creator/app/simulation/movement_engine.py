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

"""Movement engine - NPC autonomous movement decisions.

Each tick, NPCs may decide to move to an adjacent location based on
their intention vectors, energy, and the location graph. Movement
is not instant: NPCs enter a "traveling" state and arrive after
travel_hours have elapsed.

Intention-to-movement mapping:
  - explore: strong pull toward unvisited/distant locations
  - survive: flee from low-safety locations
  - escape: flee from current location
  - socialize: move toward crowded locations
  - achieve/dominate: move toward resource-rich locations
  - nurture: stay near NPCs they have positive relationships with

All math is pure numpy - no LLM calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.models.locations import LocationGraph
from app.models.npc_status import INTENTION_LABELS, NPCVectorialStatus

# Intention indices
_IDX_SURVIVE = INTENTION_LABELS.index("survive")
_IDX_SOCIALIZE = INTENTION_LABELS.index("socialize")
_IDX_ACHIEVE = INTENTION_LABELS.index("achieve")
_IDX_EXPLORE = INTENTION_LABELS.index("explore")
_IDX_DOMINATE = INTENTION_LABELS.index("dominate")
_IDX_ESCAPE = INTENTION_LABELS.index("escape")

# Environment indices (matching ENVIRONMENT_LABELS order)
_ENV_SAFETY = 0
_ENV_RESOURCES = 1
_ENV_CROWDING = 3

# NPCs below this energy won't move.
_MIN_ENERGY_TO_MOVE = 0.15

# Base probability multiplier for movement decisions.
_BASE_MOVE_PROBABILITY = 0.2

# Energy cost per hour of travel.
_ENERGY_COST_PER_HOUR = 0.02


@dataclass
class TravelState:
    """Tracks an NPC currently traveling between locations.

    Attributes:
        npc_id: The traveling NPC.
        origin_id: Where they left from.
        destination_id: Where they're going.
        departure_time: Game-time when they left.
        arrival_time: Game-time when they arrive.
    """

    npc_id: str
    origin_id: str
    destination_id: str
    departure_time: float
    arrival_time: float


@dataclass
class MovementEngine:
    """Decides NPC movement and manages travel state.

    Args:
        move_probability: Base probability an NPC considers moving.
    """

    move_probability: float = _BASE_MOVE_PROBABILITY
    _travelers: dict[str, TravelState] = field(default_factory=dict)
    _rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())

    @property
    def travelers(self) -> dict[str, TravelState]:
        """Current traveling NPCs (read-only view)."""
        return dict(self._travelers)

    def is_traveling(self, npc_id: str) -> bool:
        """Check if an NPC is currently in transit."""
        return npc_id in self._travelers

    def score_destination(
        self,
        npc: NPCVectorialStatus,
        dest_env: np.ndarray,
        edge_danger: float,
        travel_hours: float,
    ) -> float:
        """Score how attractive a destination is for an NPC.

        Higher scores = NPC more likely to travel there.

        Args:
            npc: The NPC considering the move.
            dest_env: Environment vector of the destination.
            edge_danger: Danger level of the path (0-1).
            travel_hours: Travel time in game-hours.

        Returns:
            Attractiveness score (can be negative).
        """
        intention = npc.intention
        curr_env = npc.environment

        score = 0.0

        # Explore: prefer destinations different from current
        env_diff = float(np.linalg.norm(dest_env - curr_env))
        score += intention[_IDX_EXPLORE] * env_diff * 2.0

        # Survive: prefer safer destinations, avoid danger
        safety_gain = dest_env[_ENV_SAFETY] - curr_env[_ENV_SAFETY]
        score += intention[_IDX_SURVIVE] * safety_gain * 3.0
        score -= intention[_IDX_SURVIVE] * edge_danger * 2.0

        # Escape: strong pull to leave, any destination
        score += intention[_IDX_ESCAPE] * 1.5

        # Socialize: prefer crowded destinations
        crowding_gain = dest_env[_ENV_CROWDING] - curr_env[_ENV_CROWDING]
        score += intention[_IDX_SOCIALIZE] * crowding_gain * 2.0

        # Achieve/Dominate: prefer resource-rich destinations
        resource_gain = dest_env[_ENV_RESOURCES] - curr_env[_ENV_RESOURCES]
        score += (
            (intention[_IDX_ACHIEVE] + intention[_IDX_DOMINATE]) * resource_gain * 2.0
        )

        # Penalize long travel times
        score -= travel_hours * 0.1

        # Penalize danger for non-brave NPCs
        score -= edge_danger * (1.0 - intention[_IDX_DOMINATE]) * 1.5

        return score

    def decide_movement(
        self,
        npc: NPCVectorialStatus,
        graph: LocationGraph,
    ) -> str | None:
        """Decide whether an NPC should move and where.

        Args:
            npc: The NPC to consider.
            graph: The world location graph.

        Returns:
            Destination location_id, or None if NPC stays.
        """
        if npc.energy < _MIN_ENERGY_TO_MOVE:
            return None

        neighbors = graph.get_neighbors(npc.location_id)
        if not neighbors:
            return None

        # Score each neighbor
        scores: list[tuple[str, float]] = []
        for edge in neighbors:
            dest = graph.get_location(edge.target_id)
            if dest is None:
                continue
            # Skip full locations
            if dest.capacity > 0:
                # We don't know exact count here; MovementEngine.tick handles this
                pass
            s = self.score_destination(
                npc, dest.environment, edge.danger, edge.travel_hours
            )
            scores.append((edge.target_id, s))

        if not scores:
            return None

        # Only consider destinations with positive scores
        positive = [(dest_id, s) for dest_id, s in scores if s > 0.0]
        if not positive:
            return None

        # Pick the best destination
        best_id, best_score = max(positive, key=lambda x: x[1])

        # Probability to actually move (scaled by best score)
        prob = self.move_probability * float(np.clip(best_score, 0.0, 1.0))
        if self._rng.random() < prob:
            return best_id

        return None

    def tick(
        self,
        npcs: list[NPCVectorialStatus],
        graph: LocationGraph,
        game_time: float,
    ) -> list[TravelState]:
        """Process movement for all NPCs in one tick.

        1. Arrive travelers who have reached their destination.
        2. Decide new movements for stationary NPCs.
        3. Deduct energy from travelers.

        Args:
            npcs: All NPCs in the world (mutated in-place).
            graph: The world location graph.
            game_time: Current in-game time.

        Returns:
            List of newly started journeys this tick.
        """
        # Count NPCs per location (for capacity checks)
        counts: dict[str, int] = {}
        for npc in npcs:
            counts[npc.location_id] = counts.get(npc.location_id, 0) + 1

        # 1. Arrive travelers who have reached their destination
        arrived: list[str] = []
        for npc_id, travel in self._travelers.items():
            if game_time >= travel.arrival_time:
                arrived.append(npc_id)

        for npc_id in arrived:
            travel = self._travelers.pop(npc_id)
            # Find the NPC and update their location
            for npc in npcs:
                if npc.npc_id == npc_id:
                    npc.location_id = travel.destination_id
                    break

        # 2. Decide new movements for stationary NPCs
        new_journeys: list[TravelState] = []
        for npc in npcs:
            if self.is_traveling(npc.npc_id):
                continue
            dest_id = self.decide_movement(npc, graph)
            if dest_id is None:
                continue

            # Check capacity at destination
            dest = graph.get_location(dest_id)
            if dest and dest.capacity > 0:
                dest_count = counts.get(dest_id, 0)
                if dest_count >= dest.capacity:
                    continue

            # Get travel time
            edge = graph.get_edge(npc.location_id, dest_id)
            if edge is None:
                continue

            travel = TravelState(
                npc_id=npc.npc_id,
                origin_id=npc.location_id,
                destination_id=dest_id,
                departure_time=game_time,
                arrival_time=game_time + edge.travel_hours,
            )
            self._travelers[npc.npc_id] = travel
            new_journeys.append(travel)

            # Deduct energy for the journey
            cost = edge.travel_hours * _ENERGY_COST_PER_HOUR
            npc.energy = max(0.0, npc.energy - cost)

        return new_journeys
