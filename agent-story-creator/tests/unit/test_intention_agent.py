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

"""Tests for IntentionAgent."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from app.agents.intention_agent import IntentionAgent
from app.agents.serialization import serialize_npc
from app.models.npc_status import NPCVectorialStatus


def _make_ctx(state: dict) -> MagicMock:
    ctx = MagicMock()
    ctx.session.state = dict(state)
    return ctx


@pytest.mark.asyncio
async def test_intention_recompute():
    npc = NPCVectorialStatus(npc_id="npc-1", name="Guard", archetype="guard")
    npc.emotion = np.array([0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    ctx = _make_ctx({"npcs": [serialize_npc(npc)]})
    agent = IntentionAgent(name="intention_agent")

    events = []
    async for event in agent._run_async_impl(ctx):
        events.append(event)

    assert len(events) == 1
    assert "1 NPCs recomputed" in events[0].content.parts[0].text

    updated = ctx.session.state["npcs"][0]
    new_intention = np.array(updated["intention"], dtype=np.float32)
    # Intention should be normalized (unit vector)
    assert np.linalg.norm(new_intention) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.asyncio
async def test_intention_agent_empty_npcs():
    ctx = _make_ctx({"npcs": []})
    agent = IntentionAgent(name="intention_agent")

    events = []
    async for event in agent._run_async_impl(ctx):
        events.append(event)

    assert "No NPCs" in events[0].content.parts[0].text


@pytest.mark.asyncio
async def test_intention_batch_update():
    npcs = [NPCVectorialStatus(npc_id=f"npc-{i}", name=f"NPC {i}") for i in range(5)]

    ctx = _make_ctx({"npcs": [serialize_npc(npc) for npc in npcs]})
    agent = IntentionAgent(name="intention_agent")

    events = []
    async for event in agent._run_async_impl(ctx):
        events.append(event)

    assert "5 NPCs recomputed" in events[0].content.parts[0].text
    assert len(ctx.session.state["npcs"]) == 5
