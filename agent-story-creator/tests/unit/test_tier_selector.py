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

"""Tests for dialogue tier selection logic."""

from app.dialogue.tier_selector import (
    DialogueTier,
    InteractionContext,
    select_tier,
)
from app.models.npc_status import NPCVectorialStatus


def _make_npc(importance: float = 0.5) -> NPCVectorialStatus:
    return NPCVectorialStatus(name="test_npc", importance=importance)


def test_non_player_initiated_returns_template():
    """If the player didn't initiate, always use templates."""
    npc = _make_npc()
    ctx = InteractionContext(player_initiated=False, local_llm_available=True)
    assert select_tier(npc, ctx) == DialogueTier.TEMPLATE


def test_important_npc_uses_cloud():
    """High-importance NPCs should use cloud LLM."""
    npc = _make_npc(importance=0.9)
    ctx = InteractionContext(player_initiated=True, local_llm_available=True)
    assert select_tier(npc, ctx) == DialogueTier.CLOUD_LLM


def test_quest_critical_uses_cloud():
    """Quest-critical interactions should use cloud LLM."""
    npc = _make_npc(importance=0.3)
    ctx = InteractionContext(
        player_initiated=True,
        is_quest_critical=True,
        local_llm_available=True,
    )
    assert select_tier(npc, ctx) == DialogueTier.CLOUD_LLM


def test_many_turns_escalates_to_cloud():
    """After several turns, escalate to cloud for coherence."""
    npc = _make_npc(importance=0.3)
    ctx = InteractionContext(
        player_initiated=True,
        turn_count=5,
        local_llm_available=True,
    )
    assert select_tier(npc, ctx) == DialogueTier.CLOUD_LLM


def test_routine_npc_with_local_llm():
    """Normal NPC with local LLM available should use local."""
    npc = _make_npc(importance=0.3)
    ctx = InteractionContext(
        player_initiated=True,
        turn_count=1,
        local_llm_available=True,
    )
    assert select_tier(npc, ctx) == DialogueTier.LOCAL_LLM


def test_routine_npc_without_local_llm_falls_back_to_cloud():
    """Without local LLM, routine NPCs should fall back to cloud."""
    npc = _make_npc(importance=0.3)
    ctx = InteractionContext(
        player_initiated=True,
        turn_count=1,
        local_llm_available=False,
    )
    assert select_tier(npc, ctx) == DialogueTier.CLOUD_LLM
