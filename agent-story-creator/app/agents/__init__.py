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

"""Multi-agent architecture for the Emergent Narrative Engine.

Agent hierarchy:
    orchestrator (OrchestratorAgent - custom BaseAgent)
    +-- simulation_pipeline (SequentialAgent)
    |   +-- emotion_agent (EmotionAgent - BaseAgent, no LLM)
    |   +-- intention_agent (IntentionAgent - BaseAgent, no LLM)
    |   +-- event_agent (EventAgent - BaseAgent, no LLM)
    +-- dialogue_agent (DialogueAgent - custom BaseAgent)
        +-- llm_dialogue_agent (LlmAgent - Gemini Flash)

Communication between agents uses session.state with these keys:
    - "npcs": list of serialized NPCVectorialStatus dicts
    - "events": list of serialized WorldEvent dicts
    - "current_time": float, in-game hours
    - "dialogue_requests": list of dicts with npc_id + InteractionContext
    - "dialogue_responses": list of dicts with npc_id + dialogue text
"""

__all__ = [
    "DialogueAgent",
    "EmotionAgent",
    "EventAgent",
    "IntentionAgent",
    "OrchestratorAgent",
    "deserialize_event",
    "deserialize_npc",
    "serialize_event",
    "serialize_npc",
]


def __getattr__(name: str):
    """Lazy imports to avoid loading ADK at module import time."""
    if name in (
        "serialize_npc",
        "deserialize_npc",
        "serialize_event",
        "deserialize_event",
    ):
        from .serialization import (
            deserialize_event,
            deserialize_npc,
            serialize_event,
            serialize_npc,
        )

        _exports = {
            "serialize_npc": serialize_npc,
            "deserialize_npc": deserialize_npc,
            "serialize_event": serialize_event,
            "deserialize_event": deserialize_event,
        }
        return _exports[name]
    if name == "EmotionAgent":
        from .emotion_agent import EmotionAgent

        return EmotionAgent
    if name == "IntentionAgent":
        from .intention_agent import IntentionAgent

        return IntentionAgent
    if name == "EventAgent":
        from .event_agent import EventAgent

        return EventAgent
    if name == "DialogueAgent":
        from .dialogue_agent import DialogueAgent

        return DialogueAgent
    if name == "OrchestratorAgent":
        from .orchestrator import OrchestratorAgent

        return OrchestratorAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
