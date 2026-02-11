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

"""Event propagation logic.

Handles cascading events through locality scales with delay and attenuation.
When an event arrives at a locality, a propagated copy is scheduled
for delivery at the next broader locality after the configured delay.
"""

from __future__ import annotations

import copy

from app.models.events import WorldEvent

from .event_queue import EventQueue


class EventPropagator:
    """Propagates events outward through locality scales.

    When an event is submitted, it is immediately queued at its origin scale.
    Then, for each subsequent scale, an attenuated copy is scheduled
    with the appropriate delay.
    """

    def __init__(self, queue: EventQueue):
        self._queue = queue

    def submit(self, event: WorldEvent) -> int:
        """Submit a new event and schedule its full propagation cascade.

        The event is queued at its origin scale immediately, then
        attenuated copies are scheduled for each broader scale.

        Args:
            event: The event to propagate.

        Returns:
            Number of scheduled deliveries (including the original).
        """
        self._queue.push(event, event.timestamp)
        count = 1

        current = copy.deepcopy(event)
        while True:
            result = current.next_propagation()
            if result is None:
                break
            next_scale, delay, new_intensity = result

            propagated = copy.deepcopy(current)
            propagated.current_scale = next_scale
            propagated.intensity = new_intensity
            propagated.emotion_impact = current.emotion_impact * (
                new_intensity / max(current.intensity, 1e-8)
            )
            propagated.social_impact = current.social_impact * (
                new_intensity / max(current.intensity, 1e-8)
            )

            delivery_time = current.timestamp + delay
            propagated.timestamp = delivery_time
            self._queue.push(propagated, delivery_time)
            count += 1

            current = propagated

        return count

    def get_pending_count(self) -> int:
        """Return the number of pending events in the queue."""
        return len(self._queue)
