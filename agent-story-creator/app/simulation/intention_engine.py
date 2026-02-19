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

"""Intention computation engine.

Recalculates NPC intention vectors from their composite state.
Pure linear algebra — no LLM calls. Designed for batch processing
of hundreds of NPCs per tick.

Formula:
    intention_new = normalize(
        w_personality * M_personality @ personality
        + w_emotion   * M_emotion   @ emotion
        + w_social    * M_social    @ social_influence
        + w_env       * M_env       @ environment
        + w_momentum  * intention_old
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.models.npc_status import (
    EMOTION_DIM,
    ENVIRONMENT_DIM,
    INTENTION_DIM,
    INTENTION_LABELS,
    PERSONALITY_DIM,
    SOCIAL_INFLUENCE_DIM,
    NPCVectorialStatus,
)

# Intention indices for vitality bias.
_IDX_SURVIVE = INTENTION_LABELS.index("survive")
_IDX_ESCAPE = INTENTION_LABELS.index("escape")

# Thresholds below which vitality biases kick in.
_LOW_ENERGY_THRESHOLD = 0.3
_LOW_HEALTH_THRESHOLD = 0.4

# Strength of the vitality bias on intention computation.
_ENERGY_SURVIVE_BIAS = 0.5
_HEALTH_SURVIVE_BIAS = 0.8
_HEALTH_ESCAPE_BIAS = 0.3


def _random_mapping(input_dim: int, output_dim: int) -> np.ndarray:
    """Create a small random transformation matrix (for initialization)."""
    rng = np.random.default_rng(42)
    m = rng.normal(0, 0.3, (output_dim, input_dim)).astype(np.float32)
    return m


@dataclass
class ArchetypeWeights:
    """Tunable weights and transformation matrices for an NPC archetype.

    Game designers can adjust these per archetype to create different
    behavioral profiles (e.g., a guard prioritizes "survive" and "dominate",
    a merchant prioritizes "achieve" and "socialize").
    """

    w_personality: float = 0.25
    w_emotion: float = 0.25
    w_social: float = 0.15
    w_environment: float = 0.15
    w_momentum: float = 0.20

    # Transformation matrices (input_dim → INTENTION_DIM)
    m_personality: np.ndarray = field(
        default_factory=lambda: _random_mapping(PERSONALITY_DIM, INTENTION_DIM)
    )
    m_emotion: np.ndarray = field(
        default_factory=lambda: _random_mapping(EMOTION_DIM, INTENTION_DIM)
    )
    m_social: np.ndarray = field(
        default_factory=lambda: _random_mapping(SOCIAL_INFLUENCE_DIM, INTENTION_DIM)
    )
    m_environment: np.ndarray = field(
        default_factory=lambda: _random_mapping(ENVIRONMENT_DIM, INTENTION_DIM)
    )


def _normalize(v: np.ndarray) -> np.ndarray:
    """L2-normalize a vector; return uniform if zero."""
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return np.ones_like(v) / len(v)
    return v / norm


class IntentionEngine:
    """Computes updated intention vectors for NPCs.

    Supports both single-NPC updates and batch processing.
    """

    def __init__(self, archetype_weights: dict[str, ArchetypeWeights] | None = None):
        self._weights: dict[str, ArchetypeWeights] = archetype_weights or {}
        self._default = ArchetypeWeights()

    def get_weights(self, archetype: str) -> ArchetypeWeights:
        return self._weights.get(archetype, self._default)

    def register_archetype(self, name: str, weights: ArchetypeWeights) -> None:
        self._weights[name] = weights

    def compute(self, npc: NPCVectorialStatus) -> np.ndarray:
        """Compute a new intention vector for a single NPC.

        Includes vitality bias: low energy boosts "survive", low health
        strongly boosts "survive" and "escape".

        Args:
            npc: The NPC whose intention to recompute.

        Returns:
            A new normalized intention vector (does NOT mutate the NPC).
        """
        w = self.get_weights(npc.archetype)

        raw = (
            w.w_personality * (w.m_personality @ npc.personality)
            + w.w_emotion * (w.m_emotion @ npc.emotion)
            + w.w_social * (w.m_social @ npc.social_influence)
            + w.w_environment * (w.m_environment @ npc.environment)
            + w.w_momentum * npc.intention
        )

        # Vitality bias: low energy/health shifts intentions toward survival
        if npc.energy < _LOW_ENERGY_THRESHOLD:
            deficit = (_LOW_ENERGY_THRESHOLD - npc.energy) / _LOW_ENERGY_THRESHOLD
            raw[_IDX_SURVIVE] += _ENERGY_SURVIVE_BIAS * deficit

        if npc.health < _LOW_HEALTH_THRESHOLD:
            deficit = (_LOW_HEALTH_THRESHOLD - npc.health) / _LOW_HEALTH_THRESHOLD
            raw[_IDX_SURVIVE] += _HEALTH_SURVIVE_BIAS * deficit
            raw[_IDX_ESCAPE] += _HEALTH_ESCAPE_BIAS * deficit

        return _normalize(raw)

    def compute_batch(self, npcs: list[NPCVectorialStatus]) -> list[np.ndarray]:
        """Compute new intentions for a batch of NPCs.

        Args:
            npcs: List of NPCs to update.

        Returns:
            List of new intention vectors (same order as input).
        """
        return [self.compute(npc) for npc in npcs]

    def tick(self, npcs: list[NPCVectorialStatus]) -> None:
        """Update intention vectors in-place for all NPCs.

        Args:
            npcs: NPCs to mutate.
        """
        for npc in npcs:
            npc.intention = self.compute(npc)
