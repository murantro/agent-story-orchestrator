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

"""EmotionAgent - applies emotion decay and event impacts.

A custom BaseAgent (no LLM) that wraps EmotionEngine. Reads NPC and
event data from session.state, applies emotion updates, and writes
updated NPC data back.

State keys read:
    - "npcs": list[dict] - serialized NPCVectorialStatus objects
    - "events": list[dict] - serialized WorldEvent objects

State keys written:
    - "npcs": list[dict] - updated NPC data with new emotion vectors
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types

from app.simulation.emotion_engine import EmotionEngine

from .serialization import deserialize_event, deserialize_npc, serialize_npc


class EmotionAgent(BaseAgent):
    """Applies emotion decay and event-driven emotion shifts to NPCs.

    Wraps EmotionEngine without modifying it. Communicates via session.state.
    """

    decay_rate: float = 0.05
    impact_scale: float = 1.0

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        engine = EmotionEngine(
            decay_rate=self.decay_rate, impact_scale=self.impact_scale
        )

        npc_dicts = ctx.session.state.get("npcs", [])
        event_dicts = ctx.session.state.get("events", [])

        if not npc_dicts:
            yield Event(
                author=self.name,
                content=types.Content(
                    parts=[types.Part.from_text(text="No NPCs to process.")]
                ),
            )
            return

        npcs = [deserialize_npc(d) for d in npc_dicts]
        events = [deserialize_event(d) for d in event_dicts]

        # Apply emotion decay
        engine.tick(npcs)

        # Apply event impacts
        for event in events:
            engine.apply_event_batch(npcs, event)

        # Write updated NPCs back to state
        ctx.session.state["npcs"] = [serialize_npc(npc) for npc in npcs]

        updated_count = len(npcs)
        event_count = len(events)
        yield Event(
            author=self.name,
            content=types.Content(
                parts=[
                    types.Part.from_text(
                        text=(
                            f"Emotion update complete: {updated_count} NPCs decayed, "
                            f"{event_count} events applied."
                        )
                    )
                ]
            ),
        )
