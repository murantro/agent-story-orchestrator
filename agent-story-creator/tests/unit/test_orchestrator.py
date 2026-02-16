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

"""Tests for OrchestratorAgent."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from app.agents.orchestrator import OrchestratorAgent, build_simulation_pipeline
from app.agents.serialization import serialize_event, serialize_npc
from app.models.events import WorldEvent
from app.models.npc_status import NPCVectorialStatus


def _make_ctx(state: dict) -> MagicMock:
    """Create a mock InvocationContext compatible with ADK's SequentialAgent.

    SequentialAgent.run_async accesses ctx.model_copy(), plugin_manager,
    invocation_id, branch, agent_states, and end_of_agents.
    """
    ctx = MagicMock()
    ctx.session.state = dict(state)
    ctx.invocation_id = "test-invocation"
    ctx.branch = "test-branch"
    ctx.agent_states = {}
    ctx.end_of_agents = {}
    ctx.end_invocation = False

    # plugin_manager.run_before_agent_callback and run_after_agent_callback
    # are awaited by BaseAgent.run_async
    ctx.plugin_manager.run_before_agent_callback = AsyncMock(return_value=None)
    ctx.plugin_manager.run_after_agent_callback = AsyncMock(return_value=None)

    # model_copy() is called by SequentialAgent to create child contexts
    # It must return a ctx-like object with the same state dict reference
    def _model_copy(update=None, **kwargs):
        child = _make_ctx(state)
        # Share the same state dict so writes propagate
        child.session.state = ctx.session.state
        if update:
            for key, val in update.items():
                setattr(child, key, val)
        return child

    ctx.model_copy = _model_copy
    return ctx


def _make_full_state() -> dict:
    """Create a complete state for a full orchestration run."""
    npc = NPCVectorialStatus(
        npc_id="npc-1",
        name="Merchant",
        archetype="merchant",
    )
    npc.emotion = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    event = WorldEvent(
        event_id="evt-1",
        event_type="trade_deal",
        timestamp=10.0,
        intensity=0.7,
    )
    event.emotion_impact = np.array(
        [0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0], dtype=np.float32
    )

    return {
        "request_type": "full",
        "npcs": [serialize_npc(npc)],
        "pending_events": [serialize_event(event)],
        "current_time": 10.0,
        "dialogue_requests": [
            {"npc_id": "npc-1", "player_initiated": False},
        ],
    }


@pytest.mark.asyncio
async def test_full_orchestration():
    state = _make_full_state()
    ctx = _make_ctx(state)
    agent = OrchestratorAgent(name="orchestrator")

    events = []
    async for ev in agent._run_async_impl(ctx):
        events.append(ev)

    # Should have events from event_agent, emotion_agent, intention_agent,
    # dialogue_agent, and the orchestrator summary
    assert len(events) >= 3
    last_event = events[-1]
    assert "simulation" in last_event.content.parts[0].text
    assert "dialogue" in last_event.content.parts[0].text
    assert "orchestrator_status" in ctx.session.state


@pytest.mark.asyncio
async def test_tick_only():
    state = _make_full_state()
    state["request_type"] = "tick"
    ctx = _make_ctx(state)
    agent = OrchestratorAgent(name="orchestrator")

    events = []
    async for ev in agent._run_async_impl(ctx):
        events.append(ev)

    last_text = events[-1].content.parts[0].text
    assert "simulation" in last_text
    assert "dialogue" not in last_text


@pytest.mark.asyncio
async def test_dialogue_only():
    npc = NPCVectorialStatus(npc_id="npc-1", name="Merchant")
    state = {
        "request_type": "dialogue",
        "npcs": [serialize_npc(npc)],
        "pending_events": [],
        "current_time": 0.0,
        "dialogue_requests": [
            {"npc_id": "npc-1", "player_initiated": False},
        ],
    }
    ctx = _make_ctx(state)
    agent = OrchestratorAgent(name="orchestrator")

    events = []
    async for ev in agent._run_async_impl(ctx):
        events.append(ev)

    last_text = events[-1].content.parts[0].text
    assert "dialogue" in last_text
    assert "simulation" not in last_text


@pytest.mark.asyncio
async def test_build_simulation_pipeline():
    pipeline = build_simulation_pipeline()
    assert pipeline.name == "simulation_pipeline"
    assert len(pipeline.sub_agents) == 3
    agent_names = [a.name for a in pipeline.sub_agents]
    assert agent_names == ["event_agent", "emotion_agent", "intention_agent"]
