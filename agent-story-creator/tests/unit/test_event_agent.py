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

"""Tests for EventAgent."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.agents.event_agent import EventAgent
from app.agents.serialization import serialize_event
from app.models.events import LocalityScale, WorldEvent


def _make_ctx(state: dict) -> MagicMock:
    ctx = MagicMock()
    ctx.session.state = dict(state)
    return ctx


@pytest.mark.asyncio
async def test_event_submission_and_delivery():
    event = WorldEvent(
        event_id="evt-1",
        event_type="murder",
        origin_scale=LocalityScale.PERSONAL,
        current_scale=LocalityScale.PERSONAL,
        timestamp=10.0,
        intensity=0.9,
    )

    ctx = _make_ctx(
        {
            "pending_events": [serialize_event(event)],
            "current_time": 10.0,
        }
    )
    agent = EventAgent(name="event_agent")

    events = []
    async for ev in agent._run_async_impl(ctx):
        events.append(ev)

    assert len(events) == 1
    assert "1 submitted" in events[0].content.parts[0].text
    assert "1 delivered" in events[0].content.parts[0].text

    # The original event should be delivered (timestamp <= current_time)
    delivered = ctx.session.state["events"]
    assert len(delivered) >= 1

    # Pending should be cleared
    assert ctx.session.state["pending_events"] == []


@pytest.mark.asyncio
async def test_event_future_not_delivered():
    event = WorldEvent(
        event_id="evt-future",
        event_type="trade",
        timestamp=100.0,
        intensity=0.5,
    )

    ctx = _make_ctx(
        {
            "pending_events": [serialize_event(event)],
            "current_time": 50.0,  # Before the event timestamp
        }
    )
    agent = EventAgent(name="event_agent")

    events = []
    async for ev in agent._run_async_impl(ctx):
        events.append(ev)

    assert "0 delivered" in events[0].content.parts[0].text


@pytest.mark.asyncio
async def test_event_agent_no_pending():
    ctx = _make_ctx(
        {
            "pending_events": [],
            "current_time": 0.0,
        }
    )
    agent = EventAgent(name="event_agent")

    events = []
    async for ev in agent._run_async_impl(ctx):
        events.append(ev)

    assert "0 submitted" in events[0].content.parts[0].text
    assert "0 delivered" in events[0].content.parts[0].text
