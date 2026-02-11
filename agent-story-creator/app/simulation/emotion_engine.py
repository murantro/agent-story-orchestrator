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

"""Emotion engine — decay toward personality baseline + event impact.

Emotions naturally decay toward a baseline determined by the NPC's
personality. Events push emotions in specific directions.
"""

from __future__ import annotations

import numpy as np

from app.models.events import WorldEvent
from app.models.npc_status import (
    EMOTION_DIM,
    PERSONALITY_DIM,
    NPCVectorialStatus,
)

# Mapping from Big Five personality to emotion baseline (5 → 8 matrix)
# High openness → more anticipation and surprise
# High conscientiousness → more trust
# High extraversion → more joy
# High agreeableness → more trust, less anger
# High neuroticism → more sadness, fear, anger
PERSONALITY_TO_EMOTION_BASELINE = np.array(
    [
        # joy   sad   anger  fear  surpr  disg  trust  antic
        [0.1, 0.0, 0.0, 0.0, 0.2, 0.0, 0.1, 0.3],  # openness
        [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.1],  # conscientiousness
        [0.3, -0.1, 0.0, -0.1, 0.1, 0.0, 0.1, 0.1],  # extraversion
        [0.2, 0.0, -0.2, 0.0, 0.0, -0.1, 0.3, 0.0],  # agreeableness
        [-0.2, 0.3, 0.2, 0.3, 0.0, 0.1, -0.2, 0.0],  # neuroticism
    ],
    dtype=np.float32,
)

assert PERSONALITY_TO_EMOTION_BASELINE.shape == (PERSONALITY_DIM, EMOTION_DIM)


class EmotionEngine:
    """Manages emotion decay and event-driven emotion shifts.

    Args:
        decay_rate: How quickly emotions return to baseline per tick (0-1).
            0 = no decay, 1 = instant snap to baseline.
        impact_scale: Global multiplier for event emotion impacts.
    """

    def __init__(self, decay_rate: float = 0.05, impact_scale: float = 1.0):
        self.decay_rate = decay_rate
        self.impact_scale = impact_scale

    def compute_baseline(self, personality: np.ndarray) -> np.ndarray:
        """Compute the emotional baseline from a personality vector.

        Args:
            personality: 5-dim Big Five vector.

        Returns:
            8-dim emotion baseline vector.
        """
        baseline = personality @ PERSONALITY_TO_EMOTION_BASELINE
        return np.clip(baseline, 0.0, 1.0)

    def decay(self, npc: NPCVectorialStatus) -> np.ndarray:
        """Decay current emotion toward personality-derived baseline.

        Args:
            npc: The NPC (not mutated).

        Returns:
            New emotion vector after decay.
        """
        baseline = self.compute_baseline(npc.personality)
        new_emotion = npc.emotion + self.decay_rate * (baseline - npc.emotion)
        return np.clip(new_emotion, 0.0, 1.0).astype(np.float32)

    def apply_event(self, npc: NPCVectorialStatus, event: WorldEvent) -> np.ndarray:
        """Apply an event's emotion impact to an NPC.

        The impact is scaled by event intensity and the global impact_scale.

        Args:
            npc: The NPC (not mutated).
            event: The world event to apply.

        Returns:
            New emotion vector after impact.
        """
        impact = event.emotion_impact * event.intensity * self.impact_scale
        new_emotion = npc.emotion + impact
        return np.clip(new_emotion, 0.0, 1.0).astype(np.float32)

    def tick(self, npcs: list[NPCVectorialStatus]) -> None:
        """Apply emotion decay in-place for all NPCs.

        Args:
            npcs: NPCs to mutate.
        """
        for npc in npcs:
            npc.emotion = self.decay(npc)

    def apply_event_batch(
        self,
        npcs: list[NPCVectorialStatus],
        event: WorldEvent,
    ) -> None:
        """Apply an event's emotion impact in-place to multiple NPCs.

        Args:
            npcs: NPCs to mutate.
            event: The event to apply.
        """
        for npc in npcs:
            npc.emotion = self.apply_event(npc, event)
