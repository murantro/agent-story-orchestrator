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

"""Environment engine - updates NPC environment vectors from locations.

Each tick, every NPC's environment vector is blended toward the
environment of their current location. This feeds into the intention
computation (unsafe locations push NPCs toward survive/escape, etc.).

The crowding dimension is computed dynamically from the number of
NPCs currently at that location relative to its capacity.
"""

from __future__ import annotations

import numpy as np

from app.models.locations import LocationGraph
from app.models.npc_status import ENVIRONMENT_LABELS, NPCVectorialStatus

# Index of the crowding dimension in the environment vector.
_CROWDING_IDX = ENVIRONMENT_LABELS.index("crowding")


class EnvironmentEngine:
    """Updates NPC environment vectors from the location graph.

    Args:
        blend_rate: How quickly NPC environment adapts to location (0-1).
            0 = no change, 1 = instant snap to location environment.
    """

    def __init__(self, blend_rate: float = 0.5):
        self.blend_rate = blend_rate

    def compute_crowding(
        self,
        location_capacity: int,
        npc_count_at_location: int,
    ) -> float:
        """Compute dynamic crowding value for a location.

        Args:
            location_capacity: Max NPCs (0 = unlimited).
            npc_count_at_location: Current NPC count at the location.

        Returns:
            Crowding value in [0, 1].
        """
        if location_capacity <= 0:
            # Unlimited capacity: use a soft scale (10 NPCs = 0.5 crowding)
            return float(np.clip(npc_count_at_location / 20.0, 0.0, 1.0))
        return float(np.clip(npc_count_at_location / location_capacity, 0.0, 1.0))

    def tick(
        self,
        npcs: list[NPCVectorialStatus],
        graph: LocationGraph,
    ) -> None:
        """Update environment vectors for all NPCs in-place.

        For each NPC, blends their environment vector toward their
        current location's environment. The crowding dimension is
        computed dynamically from NPC count at that location.

        Args:
            npcs: NPCs to update (mutated in-place).
            graph: The world location graph.
        """
        # Count NPCs per location for dynamic crowding
        counts: dict[str, int] = {}
        for npc in npcs:
            counts[npc.location_id] = counts.get(npc.location_id, 0) + 1

        for npc in npcs:
            loc = graph.get_location(npc.location_id)
            if loc is None:
                continue

            # Start from location's base environment
            target_env = loc.environment.copy()

            # Override crowding with dynamic value
            target_env[_CROWDING_IDX] = self.compute_crowding(
                loc.capacity, counts.get(npc.location_id, 0)
            )

            # Blend NPC environment toward location target
            npc.environment = np.clip(
                npc.environment + self.blend_rate * (target_env - npc.environment),
                0.0,
                1.0,
            ).astype(np.float32)
