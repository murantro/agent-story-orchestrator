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

"""OrchestratorAgent - top-level agent that coordinates the simulation pipeline.

Orchestrates:
  1. Event processing (EventAgent)
  2. Simulation pipeline: Emotion -> Intention (SequentialAgent)
  3. Dialogue generation (DialogueAgent)

The orchestrator decides which sub-pipelines to run based on the request
type found in session.state["request_type"]:
  - "tick": Run full simulation (events + emotions + intentions)
  - "dialogue": Run dialogue generation only
  - "full": Run simulation + dialogue

State keys read:
    - "request_type": str - "tick", "dialogue", or "full"

State keys written:
    - "orchestrator_status": str - summary of what was executed
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from google.adk.agents import BaseAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types

from .dialogue_agent import DialogueAgent
from .emotion_agent import EmotionAgent
from .event_agent import EventAgent
from .intention_agent import IntentionAgent


def build_simulation_pipeline() -> SequentialAgent:
    """Build the simulation pipeline: Event -> Emotion -> Intention.

    Returns:
        SequentialAgent that runs the three simulation steps in order.
    """
    return SequentialAgent(
        name="simulation_pipeline",
        description="Runs event processing, emotion updates, and intention recomputation in sequence.",
        sub_agents=[
            EventAgent(name="event_agent"),
            EmotionAgent(name="emotion_agent"),
            IntentionAgent(name="intention_agent"),
        ],
    )


class OrchestratorAgent(BaseAgent):
    """Top-level agent that coordinates simulation and dialogue pipelines.

    Routes requests to the appropriate sub-pipelines based on request_type.
    """

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        request_type = ctx.session.state.get("request_type", "full")
        executed: list[str] = []

        if request_type in ("tick", "full"):
            # Run simulation pipeline
            sim_pipeline = build_simulation_pipeline()
            async for event in sim_pipeline._run_async_impl(ctx):
                yield event
            executed.append("simulation")

        if request_type in ("dialogue", "full"):
            # Run dialogue pipeline
            dialogue = DialogueAgent(name="dialogue_agent")
            async for event in dialogue._run_async_impl(ctx):
                yield event
            executed.append("dialogue")

        status = f"Orchestrator complete: executed [{', '.join(executed)}]"
        ctx.session.state["orchestrator_status"] = status

        yield Event(
            author=self.name,
            content=types.Content(parts=[types.Part.from_text(text=status)]),
        )
