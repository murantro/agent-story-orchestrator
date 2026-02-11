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

"""Tests for event propagation and queue."""

import numpy as np

from app.events.event_queue import EventQueue
from app.events.propagation import EventPropagator
from app.models.events import (
    INTENSITY_THRESHOLD,
    LocalityScale,
    WorldEvent,
)
from app.models.npc_status import EMOTION_DIM, SOCIAL_INFLUENCE_DIM


def _make_event(**kwargs) -> WorldEvent:
    defaults = {
        "event_type": "test",
        "description": "A test event",
        "origin_scale": LocalityScale.PERSONAL,
        "current_scale": LocalityScale.PERSONAL,
        "timestamp": 0.0,
        "intensity": 1.0,
        "emotion_impact": np.ones(EMOTION_DIM, dtype=np.float32) * 0.5,
        "social_impact": np.ones(SOCIAL_INFLUENCE_DIM, dtype=np.float32) * 0.3,
    }
    defaults.update(kwargs)
    return WorldEvent(**defaults)


class TestEventQueue:
    def test_push_and_pop_ordered(self):
        """Events should be popped in delivery-time order."""
        q = EventQueue()
        e1 = _make_event(event_type="first")
        e2 = _make_event(event_type="second")
        e3 = _make_event(event_type="third")
        q.push(e3, 30.0)
        q.push(e1, 10.0)
        q.push(e2, 20.0)

        due = q.pop_due(15.0)
        assert len(due) == 1
        assert due[0].event_type == "first"

        due = q.pop_due(25.0)
        assert len(due) == 1
        assert due[0].event_type == "second"

    def test_pop_due_returns_empty_when_none_ready(self):
        q = EventQueue()
        q.push(_make_event(), 100.0)
        assert q.pop_due(50.0) == []

    def test_pop_due_returns_all_ready(self):
        q = EventQueue()
        q.push(_make_event(), 10.0)
        q.push(_make_event(), 20.0)
        q.push(_make_event(), 30.0)
        due = q.pop_due(25.0)
        assert len(due) == 2

    def test_empty_property(self):
        q = EventQueue()
        assert q.empty
        q.push(_make_event(), 10.0)
        assert not q.empty

    def test_peek_next_time(self):
        q = EventQueue()
        assert q.peek_next_time() is None
        q.push(_make_event(), 42.0)
        assert q.peek_next_time() == 42.0


class TestEventPropagation:
    def test_personal_event_propagates_to_multiple_scales(self):
        """A personal event should cascade through all locality scales."""
        q = EventQueue()
        prop = EventPropagator(q)
        event = _make_event(intensity=1.0)

        count = prop.submit(event)
        # personal + family + city + regional + national (global intensity too low)
        assert count >= 4, f"Expected at least 4 deliveries, got {count}"

    def test_intensity_attenuates_with_propagation(self):
        """Propagated events should have decreasing intensity."""
        q = EventQueue()
        prop = EventPropagator(q)
        event = _make_event(intensity=1.0)
        prop.submit(event)

        # Collect all events by draining queue at a far future time
        all_events = q.pop_due(1_000_000.0)
        intensities = [e.intensity for e in all_events]
        # Should be monotonically decreasing (first is original at 1.0)
        for i in range(1, len(intensities)):
            assert intensities[i] <= intensities[i - 1]

    def test_low_intensity_stops_propagation(self):
        """Events below threshold should not propagate further."""
        q = EventQueue()
        prop = EventPropagator(q)
        event = _make_event(intensity=0.01)  # Below threshold
        count = prop.submit(event)
        assert count == 1, "Low-intensity event should only deliver at origin"

    def test_propagation_delays_increase(self):
        """Each scale should have a larger delivery time than the previous."""
        q = EventQueue()
        prop = EventPropagator(q)
        event = _make_event(intensity=1.0, timestamp=0.0)
        prop.submit(event)

        all_events = q.pop_due(1_000_000.0)
        timestamps = [e.timestamp for e in all_events]
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1]

    def test_event_can_propagate_check(self):
        """can_propagate() should respect intensity threshold and max scale."""
        event = _make_event(intensity=1.0)
        assert event.can_propagate()

        event_low = _make_event(intensity=INTENSITY_THRESHOLD / 2)
        assert not event_low.can_propagate()

        event_global = _make_event(
            intensity=1.0,
            current_scale=LocalityScale.GLOBAL,
        )
        assert not event_global.can_propagate()
