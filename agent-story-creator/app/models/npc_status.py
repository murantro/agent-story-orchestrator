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

"""NPC Vectorial Status model.

Each NPC's state is a composite of hand-designed semantic vectors.
These are NOT LLM embeddings — they are interpretable, debuggable,
and deterministic feature vectors with fixed dimensions.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import numpy as np

# --- Dimension constants ---
INTENTION_DIM = (
    8  # survive, socialize, achieve, explore, create, dominate, nurture, escape
)
EMOTION_DIM = (
    8  # Plutchik: joy, sadness, anger, fear, surprise, disgust, trust, anticipation
)
PERSONALITY_DIM = (
    5  # Big Five: openness, conscientiousness, extraversion, agreeableness, neuroticism
)
SOCIAL_INFLUENCE_DIM = 6  # cultural_conformity, economic_pressure, fashion, status_seeking, religious, political
ENVIRONMENT_DIM = 4  # safety, resource_abundance, weather_comfort, crowding

# Named indices for readability
INTENTION_LABELS = [
    "survive",
    "socialize",
    "achieve",
    "explore",
    "create",
    "dominate",
    "nurture",
    "escape",
]
EMOTION_LABELS = [
    "joy",
    "sadness",
    "anger",
    "fear",
    "surprise",
    "disgust",
    "trust",
    "anticipation",
]
PERSONALITY_LABELS = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]
SOCIAL_LABELS = [
    "cultural_conformity",
    "economic_pressure",
    "fashion_awareness",
    "status_seeking",
    "religious_devotion",
    "political_alignment",
]
ENVIRONMENT_LABELS = [
    "safety",
    "resource_abundance",
    "weather_comfort",
    "crowding",
]


def _zero_vec(dim: int) -> np.ndarray:
    return np.zeros(dim, dtype=np.float32)


def _uniform_vec(dim: int) -> np.ndarray:
    v = np.ones(dim, dtype=np.float32) / dim
    return v


@dataclass
class NPCVectorialStatus:
    """Full vectorial state of a single NPC.

    Attributes:
        npc_id: Unique identifier.
        name: Display name.
        archetype: Category for tuning weights (e.g. "merchant", "guard", "noble").
        intention: 8-dim vector — what the NPC wants to do right now.
        emotion: 8-dim Plutchik emotion wheel.
        personality: 5-dim Big Five (read-only baseline, set at creation).
        social_influence: 6-dim external social pressures.
        environment: 4-dim environmental conditions sensed by the NPC.
        energy: Scalar 0-1.
        health: Scalar 0-1.
        importance: How plot-relevant this NPC is (0-1). Determines dialogue tier.
        relationships: Sparse dict of npc_id to affinity (-1 to 1).
        recent_memories: Last N significant event descriptions (plain text).
        location_id: Reference to a location entity.
    """

    npc_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    archetype: str = "generic"

    # Core vectors
    intention: np.ndarray = field(default_factory=lambda: _uniform_vec(INTENTION_DIM))
    emotion: np.ndarray = field(default_factory=lambda: _zero_vec(EMOTION_DIM))
    personality: np.ndarray = field(
        default_factory=lambda: _uniform_vec(PERSONALITY_DIM)
    )
    social_influence: np.ndarray = field(
        default_factory=lambda: _zero_vec(SOCIAL_INFLUENCE_DIM)
    )
    environment: np.ndarray = field(default_factory=lambda: _zero_vec(ENVIRONMENT_DIM))

    # Scalars
    energy: float = 1.0
    health: float = 1.0
    importance: float = 0.5

    # Relationships (sparse)
    relationships: dict[str, float] = field(default_factory=dict)

    # Memory
    recent_memories: list[str] = field(default_factory=list)

    # Location
    location_id: str = "default"

    def dominant_intention(self) -> str:
        """Return the label of the strongest intention dimension."""
        idx = int(np.argmax(self.intention))
        return INTENTION_LABELS[idx]

    def dominant_emotion(self) -> str:
        """Return the label of the strongest emotion dimension."""
        idx = int(np.argmax(self.emotion))
        return EMOTION_LABELS[idx]

    def to_character_sheet(self) -> str:
        """Serialize vectorial status into a human-readable prompt fragment.

        Used as context when generating LLM dialogue.
        """
        top_intentions = sorted(
            zip(INTENTION_LABELS, self.intention.tolist(), strict=True),
            key=lambda x: x[1],
            reverse=True,
        )[:3]
        top_emotions = sorted(
            zip(EMOTION_LABELS, self.emotion.tolist(), strict=True),
            key=lambda x: x[1],
            reverse=True,
        )[:3]
        top_personality = sorted(
            zip(PERSONALITY_LABELS, self.personality.tolist(), strict=True),
            key=lambda x: x[1],
            reverse=True,
        )[:3]

        intent_str = ", ".join(f"{label} ({val:.2f})" for label, val in top_intentions)
        emot_str = ", ".join(f"{label} ({val:.2f})" for label, val in top_emotions)
        pers_str = ", ".join(f"{label} ({val:.2f})" for label, val in top_personality)

        memories_str = (
            "; ".join(self.recent_memories[-5:])
            if self.recent_memories
            else "nothing notable recently"
        )

        return (
            f"Name: {self.name} ({self.archetype})\n"
            f"Drives: {intent_str}\n"
            f"Mood: {emot_str}\n"
            f"Personality: {pers_str}\n"
            f"Energy: {self.energy:.1f}, Health: {self.health:.1f}\n"
            f"Recent events: {memories_str}"
        )
