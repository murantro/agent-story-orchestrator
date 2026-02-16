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

"""EventAgent - processes event queue and propagation.

A custom BaseAgent (no LLM) that wraps EventQueue and EventPropagator.
Submits new events to the propagation system, advances the queue clock,
and delivers due events to session.state for downstream agents.

State keys read:
    - "pending_events": list[dict] - new events to submit
    - "current_time": float - in-game time (hours)

State keys written:
    - "events": list[dict] - events due for delivery at current_time
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types

from app.events.event_queue import EventQueue
from app.events.propagation import EventPropagator

from .serialization import deserialize_event, serialize_event


class EventAgent(BaseAgent):
    """Processes event propagation and delivers due events.

    Wraps EventQueue and EventPropagator. Communicates via session.state.
    """

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        queue = EventQueue()
        propagator = EventPropagator(queue)

        pending_dicts = ctx.session.state.get("pending_events", [])
        current_time = ctx.session.state.get("current_time", 0.0)

        # Submit new events and schedule propagation cascade
        submitted = 0
        for event_dict in pending_dicts:
            world_event = deserialize_event(event_dict)
            propagator.submit(world_event)
            submitted += 1

        # Pop events that are due at current_time
        due_events = queue.pop_due(current_time)

        # Write due events to state for EmotionAgent to process
        ctx.session.state["events"] = [serialize_event(e) for e in due_events]

        # Clear pending events
        ctx.session.state["pending_events"] = []

        remaining = len(queue)
        yield Event(
            author=self.name,
            content=types.Content(
                parts=[
                    types.Part.from_text(
                        text=(
                            f"Event processing complete: {submitted} submitted, "
                            f"{len(due_events)} delivered, {remaining} still pending."
                        )
                    )
                ]
            ),
        )
