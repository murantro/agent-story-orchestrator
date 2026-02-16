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

"""DialogueAgent - routes dialogue requests through the 3-tier pipeline.

A custom BaseAgent that wraps the tier selection logic and template engine.
For TEMPLATE tier, generates dialogue directly.
For LOCAL_LLM/CLOUD_LLM tiers, delegates to the child llm_dialogue_agent.

State keys read:
    - "npcs": list[dict] - serialized NPCVectorialStatus objects
    - "dialogue_requests": list[dict] - requests with npc_id + context

State keys written:
    - "dialogue_responses": list[dict] - responses with npc_id + text + tier
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types

from app.dialogue.template_engine import TemplateEngine
from app.dialogue.tier_selector import DialogueTier, InteractionContext, select_tier

from .serialization import deserialize_npc


class DialogueAgent(BaseAgent):
    """Routes dialogue generation through the 3-tier pipeline.

    Template-tier dialogue is generated directly. LLM-tier dialogue
    prepares the NPC character sheet in session.state for the child
    llm_dialogue_agent to process.
    """

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        template_engine = TemplateEngine()

        npc_dicts = ctx.session.state.get("npcs", [])
        requests = ctx.session.state.get("dialogue_requests", [])

        if not requests:
            yield Event(
                author=self.name,
                content=types.Content(
                    parts=[types.Part.from_text(text="No dialogue requests.")]
                ),
            )
            return

        # Build NPC lookup
        npc_map: dict[str, Any] = {}
        for npc_dict in npc_dicts:
            npc_map[npc_dict["npc_id"]] = npc_dict

        responses: list[dict[str, Any]] = []
        llm_requests: list[dict[str, Any]] = []

        for req in requests:
            npc_id = req["npc_id"]
            npc_dict = npc_map.get(npc_id)
            if npc_dict is None:
                responses.append(
                    {
                        "npc_id": npc_id,
                        "text": "...",
                        "tier": "error",
                    }
                )
                continue

            npc = deserialize_npc(npc_dict)

            interaction_ctx = InteractionContext(
                player_initiated=req.get("player_initiated", False),
                is_quest_critical=req.get("is_quest_critical", False),
                turn_count=req.get("turn_count", 0),
                local_llm_available=req.get("local_llm_available", False),
            )

            tier = select_tier(npc, interaction_ctx)

            if tier == DialogueTier.TEMPLATE:
                text = template_engine.generate(npc)
                responses.append(
                    {
                        "npc_id": npc_id,
                        "text": text,
                        "tier": "template",
                    }
                )
            else:
                # Queue for LLM processing
                llm_requests.append(
                    {
                        "npc_id": npc_id,
                        "character_sheet": npc.to_character_sheet(),
                        "tier": tier.value,
                    }
                )

        # For LLM requests, store the character sheets in state
        # so the child llm_dialogue_agent can process them
        if llm_requests:
            ctx.session.state["llm_dialogue_requests"] = llm_requests

        ctx.session.state["dialogue_responses"] = responses

        template_count = len(responses)
        llm_count = len(llm_requests)
        yield Event(
            author=self.name,
            content=types.Content(
                parts=[
                    types.Part.from_text(
                        text=(
                            f"Dialogue routing complete: {template_count} template, "
                            f"{llm_count} queued for LLM."
                        )
                    )
                ]
            ),
        )
