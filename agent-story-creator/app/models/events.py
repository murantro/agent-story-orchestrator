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

"""World event model and locality scale definitions.

Events propagate through concentric locality scales with delay and attenuation.
"""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field

import numpy as np

from .npc_status import EMOTION_DIM, SOCIAL_INFLUENCE_DIM


class LocalityScale(enum.IntEnum):
    """Concentric locality scales, ordered from narrowest to broadest."""

    PERSONAL = 0
    FAMILY = 1
    CITY = 2
    REGIONAL = 3
    NATIONAL = 4
    GLOBAL = 5


# Propagation configuration: (delay_in_game_hours, attenuation_factor)
PROPAGATION_RULES: dict[tuple[LocalityScale, LocalityScale], tuple[float, float]] = {
    (LocalityScale.PERSONAL, LocalityScale.FAMILY): (1.0, 0.8),
    (LocalityScale.FAMILY, LocalityScale.CITY): (4.0, 0.5),
    (LocalityScale.CITY, LocalityScale.REGIONAL): (24.0, 0.3),
    (LocalityScale.REGIONAL, LocalityScale.NATIONAL): (72.0, 0.15),
    (LocalityScale.NATIONAL, LocalityScale.GLOBAL): (168.0, 0.05),
}

INTENSITY_THRESHOLD = 0.02  # Events below this stop propagating


@dataclass
class WorldEvent:
    """A world event that affects NPC vectorial statuses.

    Attributes:
        event_id: Unique identifier.
        source_npc_id: NPC that caused this event (None for environmental events).
        event_type: Category string (e.g. "murder", "marriage", "trade_deal").
        description: Human-readable description.
        origin_scale: The locality scale where the event originated.
        current_scale: How far the event has propagated so far.
        location_id: Geographic reference.
        timestamp: In-game time (hours since epoch).
        intensity: 0.0-1.0, attenuates as the event propagates outward.
        emotion_impact: How this event shifts emotion vectors of NPCs who hear it.
        social_impact: How this event shifts social influence vectors.
    """

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_npc_id: str | None = None
    event_type: str = ""
    description: str = ""
    origin_scale: LocalityScale = LocalityScale.PERSONAL
    current_scale: LocalityScale = LocalityScale.PERSONAL
    location_id: str = "default"
    timestamp: float = 0.0
    intensity: float = 1.0
    emotion_impact: np.ndarray = field(
        default_factory=lambda: np.zeros(EMOTION_DIM, dtype=np.float32)
    )
    social_impact: np.ndarray = field(
        default_factory=lambda: np.zeros(SOCIAL_INFLUENCE_DIM, dtype=np.float32)
    )

    def can_propagate(self) -> bool:
        """Check if this event has enough intensity to propagate further."""
        if self.intensity < INTENSITY_THRESHOLD:
            return False
        return self.current_scale < LocalityScale.GLOBAL

    def next_propagation(self) -> tuple[LocalityScale, float, float] | None:
        """Return (next_scale, delay_hours, new_intensity) or None if done."""
        if not self.can_propagate():
            return None
        next_scale = LocalityScale(self.current_scale + 1)
        key = (self.current_scale, next_scale)
        if key not in PROPAGATION_RULES:
            return None
        delay, attenuation = PROPAGATION_RULES[key]
        new_intensity = self.intensity * attenuation
        if new_intensity < INTENSITY_THRESHOLD:
            return None
        return next_scale, delay, new_intensity
