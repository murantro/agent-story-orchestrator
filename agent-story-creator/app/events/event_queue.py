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

"""In-process event queue ordered by game-time.

For singleplayer: events are scheduled with a future game-time and
delivered to NPCs when the simulation clock reaches that time.

For multiplayer: this would be replaced by Google Cloud Pub/Sub.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

from app.models.events import WorldEvent


@dataclass(order=True)
class _ScheduledEvent:
    """Wrapper for heap ordering by delivery_time."""

    delivery_time: float
    event: WorldEvent = field(compare=False)


class EventQueue:
    """Priority queue of world events ordered by delivery time.

    Events are pushed with a scheduled delivery time and popped
    when the simulation clock advances past that time.
    """

    def __init__(self) -> None:
        self._heap: list[_ScheduledEvent] = []

    def push(self, event: WorldEvent, delivery_time: float) -> None:
        """Schedule an event for delivery at a future game-time.

        Args:
            event: The world event to schedule.
            delivery_time: In-game time (hours) when this event should fire.
        """
        heapq.heappush(self._heap, _ScheduledEvent(delivery_time, event))

    def pop_due(self, current_time: float) -> list[WorldEvent]:
        """Pop all events whose delivery time has arrived.

        Args:
            current_time: Current in-game time (hours).

        Returns:
            List of events that are due, ordered by delivery time.
        """
        due: list[WorldEvent] = []
        while self._heap and self._heap[0].delivery_time <= current_time:
            due.append(heapq.heappop(self._heap).event)
        return due

    def peek_next_time(self) -> float | None:
        """Return the delivery time of the next event, or None if empty."""
        if not self._heap:
            return None
        return self._heap[0].delivery_time

    def __len__(self) -> int:
        return len(self._heap)

    @property
    def empty(self) -> bool:
        return len(self._heap) == 0
