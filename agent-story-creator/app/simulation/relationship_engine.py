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

"""Relationship engine - updates NPC-to-NPC affinities.

Relationships are sparse floats in [-1, 1]:
  -1 = sworn enemy, 0 = stranger, +1 = best friend.

Updates come from two sources:
  1. Interaction deltas (from InteractionEngine each tick).
  2. Passive decay (weak ties fade toward 0 over time).

Formula:
  new = old + delta * (1 - abs(old))

The (1 - abs(old)) term makes extreme affinities harder to shift,
so a best friend doesn't become an enemy from one bad interaction.
"""

from __future__ import annotations

import numpy as np

from app.config import RELATIONSHIP_DECAY_RATE, RELATIONSHIP_DELTA_SCALE
from app.models.npc_status import NPCVectorialStatus

# Affinities below this absolute value are pruned on decay.
_PRUNE_THRESHOLD = 0.01


class RelationshipEngine:
    """Manages NPC relationship affinities.

    Args:
        decay_rate: Per-tick decay rate for weak ties.
        delta_scale: Global multiplier applied to all deltas.
    """

    def __init__(
        self,
        decay_rate: float = RELATIONSHIP_DECAY_RATE,
        delta_scale: float = RELATIONSHIP_DELTA_SCALE,
    ):
        self.decay_rate = decay_rate
        self.delta_scale = delta_scale

    def apply_delta(
        self,
        npc_a: NPCVectorialStatus,
        npc_b: NPCVectorialStatus,
        delta: float,
    ) -> None:
        """Apply a symmetric relationship change between two NPCs.

        Both NPCs' relationship dicts are updated so the relationship
        is always consistent from both sides.

        Args:
            npc_a: First NPC (mutated).
            npc_b: Second NPC (mutated).
            delta: Raw affinity change (positive = friendlier).
        """
        scaled = delta * self.delta_scale
        old_a = npc_a.relationships.get(npc_b.npc_id, 0.0)
        old_b = npc_b.relationships.get(npc_a.npc_id, 0.0)
        new_a = float(np.clip(old_a + scaled * (1.0 - abs(old_a)), -1.0, 1.0))
        new_b = float(np.clip(old_b + scaled * (1.0 - abs(old_b)), -1.0, 1.0))
        npc_a.relationships[npc_b.npc_id] = new_a
        npc_b.relationships[npc_a.npc_id] = new_b

    def decay(self, npcs: list[NPCVectorialStatus]) -> None:
        """Decay all weak relationships toward zero and prune negligible ones.

        Args:
            npcs: NPCs to mutate.
        """
        for npc in npcs:
            to_prune: list[str] = []
            for other_id, affinity in npc.relationships.items():
                new_val = affinity * (1.0 - self.decay_rate)
                if abs(new_val) < _PRUNE_THRESHOLD:
                    to_prune.append(other_id)
                else:
                    npc.relationships[other_id] = new_val
            for other_id in to_prune:
                del npc.relationships[other_id]

    def personality_compatibility(
        self,
        npc_a: NPCVectorialStatus,
        npc_b: NPCVectorialStatus,
    ) -> float:
        """Compute personality compatibility between two NPCs.

        Returns a value in [-1, 1] where positive means compatible
        personalities (similar openness, agreeableness, etc.).

        Args:
            npc_a: First NPC.
            npc_b: Second NPC.

        Returns:
            Compatibility score.
        """
        diff = npc_a.personality - npc_b.personality
        distance = float(np.linalg.norm(diff))
        # Max distance for 5-dim unit vectors is sqrt(5) ≈ 2.24
        max_dist = float(np.sqrt(len(npc_a.personality)))
        return 1.0 - 2.0 * (distance / max_dist)
