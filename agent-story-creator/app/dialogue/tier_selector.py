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

"""Dialogue tier selection logic.

Three tiers:
  1. TEMPLATE — ambient barks, no LLM (cost: $0, latency: 0ms)
  2. LOCAL_LLM — Ollama + Llama 3.1 8B on player GPU (cost: $0, latency: 1-3s)
  3. CLOUD_LLM — Gemini Flash via Google ADK (cost: ~$0.001, latency: 1-2s)
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from app.models.npc_status import NPCVectorialStatus


class DialogueTier(enum.Enum):
    TEMPLATE = "template"
    LOCAL_LLM = "local_llm"
    CLOUD_LLM = "cloud_llm"


@dataclass
class InteractionContext:
    """Context about the current player-NPC interaction.

    Attributes:
        player_initiated: Whether the player started the conversation.
        is_quest_critical: Whether this interaction is part of an active quest.
        turn_count: Number of back-and-forth turns so far.
        local_llm_available: Whether a local LLM is running.
    """

    player_initiated: bool = False
    is_quest_critical: bool = False
    turn_count: int = 0
    local_llm_available: bool = False


# Threshold above which an NPC is considered "important"
IMPORTANCE_THRESHOLD = 0.8

# After this many turns, escalate to cloud LLM for coherence
TURN_ESCALATION_THRESHOLD = 3


def select_tier(
    npc: NPCVectorialStatus,
    context: InteractionContext,
) -> DialogueTier:
    """Select the dialogue generation tier for an NPC interaction.

    Args:
        npc: The NPC being interacted with.
        context: Interaction context.

    Returns:
        The appropriate dialogue tier.
    """
    if not context.player_initiated:
        return DialogueTier.TEMPLATE

    needs_cloud = (
        npc.importance >= IMPORTANCE_THRESHOLD
        or context.is_quest_critical
        or context.turn_count >= TURN_ESCALATION_THRESHOLD
    )

    if needs_cloud:
        return DialogueTier.CLOUD_LLM

    if context.local_llm_available:
        return DialogueTier.LOCAL_LLM

    return DialogueTier.CLOUD_LLM
