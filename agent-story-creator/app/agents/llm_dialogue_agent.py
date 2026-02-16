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

"""LLM-backed dialogue agent configuration.

Provides a factory function to create an ADK LlmAgent (Gemini Flash)
for generating NPC dialogue. The agent reads the NPC's character sheet
from session.state and produces in-character responses.

This is the Tier 3 (Cloud LLM) dialogue generator.
"""

from __future__ import annotations

from google.adk.agents import Agent

_NPC_DIALOGUE_INSTRUCTION = """\
You are an NPC dialogue generator for an emergent narrative video game.

Given an NPC's character sheet (personality, emotions, drives, recent events),
generate a single in-character dialogue line that the NPC would say.

Rules:
- Stay strictly in character based on the provided character sheet.
- The dialogue should reflect the NPC's current emotional state and dominant drives.
- Keep responses to 1-3 sentences maximum.
- Do not use modern slang unless the NPC's archetype calls for it.
- Do not break the fourth wall or reference game mechanics.
- If the NPC has recent memories, weave them naturally into the dialogue.

The NPC's character sheet is provided in the conversation context.
Respond with ONLY the dialogue line, no quotes, no stage directions.
"""


def create_llm_dialogue_agent(
    name: str = "llm_dialogue_agent",
    model: str = "gemini-3-flash-preview",
) -> Agent:
    """Create an LlmAgent configured for NPC dialogue generation.

    Args:
        name: Agent name (must be unique in the agent tree).
        model: LLM model identifier.

    Returns:
        Configured ADK Agent (LlmAgent).
    """
    return Agent(
        name=name,
        model=model,
        instruction=_NPC_DIALOGUE_INSTRUCTION,
        output_key="llm_dialogue_output",
        description="Generates in-character NPC dialogue using a cloud LLM.",
    )
