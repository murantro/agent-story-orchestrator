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

"""Tests for DialogueAgent."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from app.agents.dialogue_agent import DialogueAgent
from app.agents.serialization import serialize_npc
from app.models.npc_status import NPCVectorialStatus


def _make_ctx(state: dict) -> MagicMock:
    ctx = MagicMock()
    ctx.session.state = dict(state)
    return ctx


def _make_npc(npc_id: str, importance: float = 0.5) -> NPCVectorialStatus:
    npc = NPCVectorialStatus(npc_id=npc_id, name=f"NPC-{npc_id}", importance=importance)
    # Set a clear dominant intention + emotion for predictable template selection
    npc.intention = np.array([0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    npc.emotion = np.array([0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return npc


@pytest.mark.asyncio
async def test_template_tier_for_ambient():
    """Non-player-initiated requests should use template tier."""
    npc = _make_npc("npc-1")
    ctx = _make_ctx(
        {
            "npcs": [serialize_npc(npc)],
            "dialogue_requests": [
                {"npc_id": "npc-1", "player_initiated": False},
            ],
        }
    )
    agent = DialogueAgent(name="dialogue_agent")

    events = []
    async for ev in agent._run_async_impl(ctx):
        events.append(ev)

    assert "1 template" in events[0].content.parts[0].text
    responses = ctx.session.state["dialogue_responses"]
    assert len(responses) == 1
    assert responses[0]["tier"] == "template"
    assert responses[0]["text"]  # Should have some text


@pytest.mark.asyncio
async def test_cloud_tier_for_important_npc():
    """Player-initiated + important NPC should route to cloud LLM."""
    npc = _make_npc("npc-2", importance=0.9)
    ctx = _make_ctx(
        {
            "npcs": [serialize_npc(npc)],
            "dialogue_requests": [
                {"npc_id": "npc-2", "player_initiated": True},
            ],
        }
    )
    agent = DialogueAgent(name="dialogue_agent")

    events = []
    async for ev in agent._run_async_impl(ctx):
        events.append(ev)

    assert "1 queued for LLM" in events[0].content.parts[0].text
    llm_reqs = ctx.session.state.get("llm_dialogue_requests", [])
    assert len(llm_reqs) == 1
    assert llm_reqs[0]["npc_id"] == "npc-2"
    assert "character_sheet" in llm_reqs[0]


@pytest.mark.asyncio
async def test_missing_npc_returns_error():
    ctx = _make_ctx(
        {
            "npcs": [],
            "dialogue_requests": [
                {"npc_id": "missing-npc", "player_initiated": False},
            ],
        }
    )
    agent = DialogueAgent(name="dialogue_agent")

    events = []
    async for ev in agent._run_async_impl(ctx):
        events.append(ev)

    responses = ctx.session.state["dialogue_responses"]
    assert len(responses) == 1
    assert responses[0]["tier"] == "error"


@pytest.mark.asyncio
async def test_no_dialogue_requests():
    ctx = _make_ctx(
        {
            "npcs": [],
            "dialogue_requests": [],
        }
    )
    agent = DialogueAgent(name="dialogue_agent")

    events = []
    async for ev in agent._run_async_impl(ctx):
        events.append(ev)

    assert "No dialogue requests" in events[0].content.parts[0].text


@pytest.mark.asyncio
async def test_mixed_tiers():
    """Multiple requests should route to different tiers correctly."""
    npc_ambient = _make_npc("npc-a", importance=0.3)
    npc_important = _make_npc("npc-b", importance=0.9)

    ctx = _make_ctx(
        {
            "npcs": [serialize_npc(npc_ambient), serialize_npc(npc_important)],
            "dialogue_requests": [
                {"npc_id": "npc-a", "player_initiated": False},
                {"npc_id": "npc-b", "player_initiated": True},
            ],
        }
    )
    agent = DialogueAgent(name="dialogue_agent")

    events = []
    async for ev in agent._run_async_impl(ctx):
        events.append(ev)

    assert "1 template" in events[0].content.parts[0].text
    assert "1 queued for LLM" in events[0].content.parts[0].text
