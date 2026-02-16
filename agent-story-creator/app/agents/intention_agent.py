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

"""IntentionAgent - recomputes NPC intention vectors.

A custom BaseAgent (no LLM) that wraps IntentionEngine. Reads NPC data
from session.state (already updated by EmotionAgent), recalculates
intention vectors, and writes results back.

State keys read:
    - "npcs": list[dict] - serialized NPCVectorialStatus objects

State keys written:
    - "npcs": list[dict] - updated NPC data with new intention vectors
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types

from app.simulation.intention_engine import IntentionEngine

from .serialization import deserialize_npc, serialize_npc


class IntentionAgent(BaseAgent):
    """Recomputes intention vectors for all NPCs.

    Wraps IntentionEngine without modifying it. Communicates via session.state.
    Expects to run after EmotionAgent so emotions are already updated.
    """

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        engine = IntentionEngine()

        npc_dicts = ctx.session.state.get("npcs", [])

        if not npc_dicts:
            yield Event(
                author=self.name,
                content=types.Content(
                    parts=[types.Part.from_text(text="No NPCs to process.")]
                ),
            )
            return

        npcs = [deserialize_npc(d) for d in npc_dicts]

        # Recompute intentions in-place
        engine.tick(npcs)

        # Write updated NPCs back to state
        ctx.session.state["npcs"] = [serialize_npc(npc) for npc in npcs]

        yield Event(
            author=self.name,
            content=types.Content(
                parts=[
                    types.Part.from_text(
                        text=f"Intention update complete: {len(npcs)} NPCs recomputed."
                    )
                ]
            ),
        )
