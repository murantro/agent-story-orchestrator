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

"""Social influence engine - peer pressure and social contagion.

Updates NPC social_influence vectors based on:
  1. Peer pressure from co-located NPCs (proximity contagion).
  2. Archetype radiation profiles (a priest radiates religious_devotion).
  3. Relationship weighting (friends influence more; enemies may cause
     reactance, i.e. pushing in the opposite direction).
  4. Personality susceptibility (high agreeableness = more susceptible).
  5. Event-driven social shifts (events with non-zero social_impact).
  6. Passive decay toward zero (without external pressure, influence fades).

The social_influence vector (6-dim) represents external social pressures
an NPC perceives from their social environment:
  [cultural_conformity, economic_pressure, fashion_awareness,
   status_seeking, religious_devotion, political_alignment]

All math is pure numpy - no LLM calls.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from app.config import SOCIAL_BLEND_RATE, SOCIAL_DECAY_RATE, SOCIAL_EVENT_SCALE
from app.models.events import WorldEvent
from app.models.npc_status import (
    PERSONALITY_LABELS,
    SOCIAL_INFLUENCE_DIM,
    NPCVectorialStatus,
)

# Personality index for susceptibility calculation.
_IDX_AGREEABLENESS = PERSONALITY_LABELS.index("agreeableness")
_IDX_NEUROTICISM = PERSONALITY_LABELS.index("neuroticism")

# --- Archetype radiation profiles ---
# Each archetype radiates a specific social signal to peers.
# Values represent the strength of each social dimension this archetype
# projects to others nearby. NPCs without a profile radiate nothing.
#
# Dims: cultural_conformity, economic_pressure, fashion_awareness,
#        status_seeking, religious_devotion, political_alignment
_ARCHETYPE_PROFILES: dict[str, np.ndarray] = {
    "merchant": np.array([0.0, 0.4, 0.2, 0.1, 0.0, 0.0], dtype=np.float32),
    "priest": np.array([0.15, 0.0, 0.0, 0.0, 0.5, 0.1], dtype=np.float32),
    "noble": np.array([0.1, 0.0, 0.15, 0.4, 0.0, 0.3], dtype=np.float32),
    "guard": np.array([0.25, 0.0, 0.0, 0.1, 0.0, 0.2], dtype=np.float32),
    "soldier": np.array([0.2, 0.0, 0.0, 0.1, 0.0, 0.25], dtype=np.float32),
    "artist": np.array([0.0, 0.0, 0.4, 0.1, 0.0, 0.0], dtype=np.float32),
    "bard": np.array([0.1, 0.0, 0.35, 0.0, 0.0, 0.0], dtype=np.float32),
    "farmer": np.array([0.2, 0.1, 0.0, 0.0, 0.1, 0.0], dtype=np.float32),
    "scholar": np.array([0.1, 0.0, 0.0, 0.15, 0.0, 0.1], dtype=np.float32),
    "criminal": np.array([0.0, 0.2, 0.0, 0.15, 0.0, -0.1], dtype=np.float32),
}

_ZERO_PROFILE = np.zeros(SOCIAL_INFLUENCE_DIM, dtype=np.float32)


def get_archetype_profile(archetype: str) -> np.ndarray:
    """Return the social radiation profile for an archetype.

    Args:
        archetype: NPC archetype string.

    Returns:
        6-dim social radiation vector (read-only reference).
    """
    return _ARCHETYPE_PROFILES.get(archetype, _ZERO_PROFILE)


@dataclass
class SocialInfluenceEngine:
    """Updates NPC social_influence vectors via peer pressure and events.

    Args:
        blend_rate: How quickly NPC social vector blends toward peer signal.
        decay_rate: Per-tick decay rate toward zero (no external pressure).
        event_scale: Multiplier for event social_impact.
    """

    blend_rate: float = SOCIAL_BLEND_RATE
    decay_rate: float = SOCIAL_DECAY_RATE
    event_scale: float = SOCIAL_EVENT_SCALE

    def compute_susceptibility(self, npc: NPCVectorialStatus) -> float:
        """Compute how susceptible an NPC is to social influence.

        High agreeableness = more susceptible, high neuroticism = slightly
        less susceptible (contrarian tendencies).

        Args:
            npc: The NPC.

        Returns:
            Susceptibility factor in [0.2, 1.0].
        """
        agreeableness = float(npc.personality[_IDX_AGREEABLENESS])
        neuroticism = float(npc.personality[_IDX_NEUROTICISM])
        # Base susceptibility from agreeableness, slight reduction from neuroticism
        raw = 0.4 + 0.5 * agreeableness - 0.15 * neuroticism
        return float(np.clip(raw, 0.2, 1.0))

    def compute_peer_signal(
        self,
        npc: NPCVectorialStatus,
        co_located: list[NPCVectorialStatus],
    ) -> np.ndarray:
        """Compute the weighted social signal from co-located peers.

        Each peer radiates:
          - Their own social_influence (contagion - influence propagates)
          - Their archetype radiation profile (innate role-based signal)

        Weighted by relationship affinity (friends influence more,
        enemies contribute less). The signal is averaged by peer count
        so that relationship weight directly scales signal strength.

        Args:
            npc: The NPC receiving influence (not mutated).
            co_located: All NPCs at the same location (may include self).

        Returns:
            6-dim weighted social signal from peers.
        """
        peer_count = 0
        weighted_sum = np.zeros(SOCIAL_INFLUENCE_DIM, dtype=np.float32)

        for other in co_located:
            if other.npc_id == npc.npc_id:
                continue

            peer_count += 1

            # Relationship weight: friends influence more, enemies less
            rel = npc.relationships.get(other.npc_id, 0.0)
            # Weight: 0.5 (stranger) + rel*0.5, so range [0, 1] for friends
            # Enemies get low positive weight (0.1 at rel=-0.8)
            weight = 0.5 + rel * 0.5

            # The signal this peer radiates
            signal = other.social_influence + get_archetype_profile(other.archetype)

            weighted_sum += weight * signal

        if peer_count == 0:
            return np.zeros(SOCIAL_INFLUENCE_DIM, dtype=np.float32)

        return weighted_sum / peer_count

    def apply_event(self, npc: NPCVectorialStatus, event: WorldEvent) -> None:
        """Apply an event's social impact to an NPC.

        Impact is scaled by event intensity and the global event_scale.

        Args:
            npc: The NPC (mutated in-place).
            event: The world event to apply.
        """
        if np.all(event.social_impact == 0.0):
            return
        impact = event.social_impact * event.intensity * self.event_scale
        npc.social_influence = np.clip(npc.social_influence + impact, 0.0, 1.0).astype(
            np.float32
        )

    def apply_event_batch(
        self, npcs: list[NPCVectorialStatus], event: WorldEvent
    ) -> None:
        """Apply an event's social impact in-place to multiple NPCs.

        Args:
            npcs: NPCs to mutate.
            event: The event to apply.
        """
        if np.all(event.social_impact == 0.0):
            return
        for npc in npcs:
            self.apply_event(npc, event)

    def tick(self, npcs: list[NPCVectorialStatus]) -> None:
        """Apply one tick of social influence dynamics to all NPCs.

        Order per NPC:
          1. Compute peer signal from co-located NPCs.
          2. Blend social_influence toward peer signal (scaled by susceptibility).
          3. Apply passive decay toward zero.
          4. Clamp to [0, 1].

        Args:
            npcs: NPCs to mutate in-place.
        """
        # Group NPCs by location for peer pressure computation
        by_location: dict[str, list[NPCVectorialStatus]] = defaultdict(list)
        for npc in npcs:
            by_location[npc.location_id].append(npc)

        for npc in npcs:
            co_located = by_location.get(npc.location_id, [])

            # 1. Compute peer signal
            peer_signal = self.compute_peer_signal(npc, co_located)

            # 2. Blend toward peer signal (scaled by susceptibility)
            susceptibility = self.compute_susceptibility(npc)
            effective_rate = self.blend_rate * susceptibility
            npc.social_influence = npc.social_influence + effective_rate * (
                peer_signal - npc.social_influence
            )

            # 3. Passive decay toward zero
            npc.social_influence *= 1.0 - self.decay_rate

            # 4. Clamp to [0, 1]
            npc.social_influence = np.clip(npc.social_influence, 0.0, 1.0).astype(
                np.float32
            )
