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

"""Schedule engine - NPC daily routines and activity management.

Each tick, the schedule engine determines what activity each NPC
should be doing based on:
  - The current hour of the in-game day (game_time % 24).
  - The NPC's archetype (different archetypes keep different hours).
  - Energy level (exhausted NPCs may sleep outside schedule).

Activities control what NPCs can do:
  - sleeping:  No interactions, no movement, boosted energy regen.
  - resting:   Reduced interactions, no movement, moderate energy regen.
  - working:   Normal interactions, no movement (stay at workplace).
  - leisure:   Normal interactions, normal movement.

All math is pure Python - no LLM calls.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.models.npc_status import NPCVectorialStatus

# --- Activity constants ---
SLEEPING = "sleeping"
RESTING = "resting"
WORKING = "working"
LEISURE = "leisure"

# Energy threshold below which NPCs collapse into sleep regardless of schedule.
_EXHAUSTION_THRESHOLD = 0.05

# --- Schedule type ---
# Each schedule is a list of (start_hour, end_hour, activity) tuples.
# Hours are in [0, 24). The schedule must cover the full 24 hours.
ScheduleSlot = tuple[float, float, str]

# --- Archetype schedule templates ---
# Each template defines a 24-hour cycle. Hours wrap around midnight.

_DEFAULT_SCHEDULE: list[ScheduleSlot] = [
    (0, 6, SLEEPING),
    (6, 7, RESTING),
    (7, 12, WORKING),
    (12, 13, LEISURE),
    (13, 18, WORKING),
    (18, 22, LEISURE),
    (22, 24, SLEEPING),
]

_GUARD_SCHEDULE: list[ScheduleSlot] = [
    (0, 6, WORKING),  # Night watch
    (6, 8, RESTING),
    (8, 14, SLEEPING),
    (14, 16, RESTING),
    (16, 24, WORKING),  # Evening/night shift
]

_MERCHANT_SCHEDULE: list[ScheduleSlot] = [
    (0, 6, SLEEPING),
    (6, 7, RESTING),
    (7, 18, WORKING),  # Long market hours
    (18, 20, LEISURE),
    (20, 22, RESTING),
    (22, 24, SLEEPING),
]

_PRIEST_SCHEDULE: list[ScheduleSlot] = [
    (0, 5, SLEEPING),
    (5, 6, RESTING),  # Early rise
    (6, 8, WORKING),  # Morning prayers
    (8, 9, LEISURE),
    (9, 12, WORKING),  # Services
    (12, 13, LEISURE),
    (13, 17, WORKING),  # Pastoral duties
    (17, 19, LEISURE),
    (19, 22, RESTING),
    (22, 24, SLEEPING),
]

_FARMER_SCHEDULE: list[ScheduleSlot] = [
    (0, 5, SLEEPING),
    (5, 6, RESTING),  # Early rise
    (6, 12, WORKING),
    (12, 13, LEISURE),  # Lunch
    (13, 18, WORKING),
    (18, 21, LEISURE),
    (21, 24, SLEEPING),  # Early to bed
]

_NOBLE_SCHEDULE: list[ScheduleSlot] = [
    (0, 8, SLEEPING),  # Sleeps in
    (8, 9, RESTING),
    (9, 12, WORKING),  # Court
    (12, 14, LEISURE),  # Long lunch
    (14, 17, WORKING),
    (17, 23, LEISURE),  # Entertaining
    (23, 24, RESTING),
]

_CRIMINAL_SCHEDULE: list[ScheduleSlot] = [
    (0, 6, WORKING),  # Night activities
    (6, 14, SLEEPING),  # Nocturnal
    (14, 16, RESTING),
    (16, 22, LEISURE),
    (22, 24, WORKING),  # Night activities
]

_ARTIST_SCHEDULE: list[ScheduleSlot] = [
    (0, 2, WORKING),  # Late-night inspiration
    (2, 9, SLEEPING),
    (9, 10, RESTING),
    (10, 13, WORKING),
    (13, 15, LEISURE),
    (15, 19, WORKING),
    (19, 24, LEISURE),  # Evening performances
]

_SCHOLAR_SCHEDULE: list[ScheduleSlot] = [
    (0, 6, SLEEPING),
    (6, 7, RESTING),
    (7, 12, WORKING),  # Morning study
    (12, 13, LEISURE),
    (13, 18, WORKING),  # Afternoon research
    (18, 20, LEISURE),
    (20, 22, WORKING),  # Evening reading
    (22, 24, SLEEPING),
]

# Map archetype names to schedule templates.
_ARCHETYPE_SCHEDULES: dict[str, list[ScheduleSlot]] = {
    "generic": _DEFAULT_SCHEDULE,
    "guard": _GUARD_SCHEDULE,
    "soldier": _GUARD_SCHEDULE,
    "merchant": _MERCHANT_SCHEDULE,
    "priest": _PRIEST_SCHEDULE,
    "farmer": _FARMER_SCHEDULE,
    "noble": _NOBLE_SCHEDULE,
    "criminal": _CRIMINAL_SCHEDULE,
    "artist": _ARTIST_SCHEDULE,
    "bard": _ARTIST_SCHEDULE,
    "scholar": _SCHOLAR_SCHEDULE,
}


def get_schedule(archetype: str) -> list[ScheduleSlot]:
    """Return the schedule template for an archetype.

    Args:
        archetype: NPC archetype string.

    Returns:
        List of (start_hour, end_hour, activity) tuples covering 24 hours.
    """
    return _ARCHETYPE_SCHEDULES.get(archetype, _DEFAULT_SCHEDULE)


def resolve_activity(schedule: list[ScheduleSlot], hour_of_day: float) -> str:
    """Determine what activity an NPC should be doing at a given hour.

    Args:
        schedule: The NPC's schedule template.
        hour_of_day: Current hour in [0, 24).

    Returns:
        Activity string (sleeping, resting, working, leisure).
    """
    for start, end, activity in schedule:
        if start <= hour_of_day < end:
            return activity
    # Fallback (should not happen with a complete schedule)
    return LEISURE


@dataclass
class ScheduleEngine:
    """Assigns NPC activities based on time-of-day and archetype.

    Also handles exhaustion override: NPCs with critically low energy
    are forced to sleep regardless of their schedule.

    Args:
        exhaustion_threshold: Energy below this forces sleep.
    """

    exhaustion_threshold: float = _EXHAUSTION_THRESHOLD

    def compute_activity(
        self,
        npc: NPCVectorialStatus,
        game_time: float,
    ) -> str:
        """Compute what activity an NPC should be doing.

        Args:
            npc: The NPC (not mutated).
            game_time: Current in-game time (hours since epoch).

        Returns:
            Activity string.
        """
        # Exhaustion override: collapse into sleep
        if npc.energy < self.exhaustion_threshold:
            return SLEEPING

        hour_of_day = game_time % 24.0
        schedule = get_schedule(npc.archetype)
        return resolve_activity(schedule, hour_of_day)

    def tick(
        self,
        npcs: list[NPCVectorialStatus],
        game_time: float,
    ) -> None:
        """Assign activities to all NPCs in-place.

        Args:
            npcs: NPCs to mutate.
            game_time: Current in-game time.
        """
        for npc in npcs:
            npc.activity = self.compute_activity(npc, game_time)
