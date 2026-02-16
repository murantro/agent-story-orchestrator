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

"""Tests for EmotionAgent."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from app.agents.emotion_agent import EmotionAgent
from app.agents.serialization import serialize_event, serialize_npc
from app.models.events import WorldEvent
from app.models.npc_status import NPCVectorialStatus


def _make_ctx(state: dict) -> MagicMock:
    """Create a mock InvocationContext with the given state."""
    ctx = MagicMock()
    ctx.session.state = dict(state)
    return ctx


@pytest.mark.asyncio
async def test_emotion_decay_updates_npcs():
    npc = NPCVectorialStatus(npc_id="npc-1", name="Test")
    npc.emotion = np.array([0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    ctx = _make_ctx({"npcs": [serialize_npc(npc)], "events": []})
    agent = EmotionAgent(name="emotion_agent")

    events = []
    async for event in agent._run_async_impl(ctx):
        events.append(event)

    assert len(events) == 1
    assert "1 NPCs decayed" in events[0].content.parts[0].text

    updated_npcs = ctx.session.state["npcs"]
    assert len(updated_npcs) == 1
    # Joy should have decayed slightly toward baseline
    assert updated_npcs[0]["emotion"][0] != pytest.approx(0.8, abs=1e-6)


@pytest.mark.asyncio
async def test_emotion_event_impact():
    npc = NPCVectorialStatus(npc_id="npc-1", name="Test")
    npc.emotion = np.zeros(8, dtype=np.float32)

    event = WorldEvent(
        event_id="evt-1",
        event_type="celebration",
        intensity=1.0,
    )
    event.emotion_impact = np.array(
        [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
    )

    ctx = _make_ctx(
        {
            "npcs": [serialize_npc(npc)],
            "events": [serialize_event(event)],
        }
    )
    agent = EmotionAgent(name="emotion_agent")

    events = []
    async for ev in agent._run_async_impl(ctx):
        events.append(ev)

    assert "1 events applied" in events[0].content.parts[0].text
    updated = ctx.session.state["npcs"][0]
    # Joy should have increased from the event
    assert updated["emotion"][0] > 0.0


@pytest.mark.asyncio
async def test_emotion_agent_empty_npcs():
    ctx = _make_ctx({"npcs": [], "events": []})
    agent = EmotionAgent(name="emotion_agent")

    events = []
    async for event in agent._run_async_impl(ctx):
        events.append(event)

    assert len(events) == 1
    assert "No NPCs" in events[0].content.parts[0].text
