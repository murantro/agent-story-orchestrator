# ruff: noqa
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

"""Root agent definition and agent tree assembly.

Agent hierarchy:
    root_agent (OrchestratorAgent)
    +-- simulation_pipeline (SequentialAgent)
    |   +-- event_agent (EventAgent)
    |   +-- emotion_agent (EmotionAgent)
    |   +-- intention_agent (IntentionAgent)
    +-- dialogue_agent (DialogueAgent)
        +-- llm_dialogue_agent (LlmAgent / Gemini Flash)
"""

import os

import google.auth
from google.adk.apps import App

from app.agents.dialogue_agent import DialogueAgent
from app.agents.llm_dialogue_agent import create_llm_dialogue_agent
from app.agents.orchestrator import OrchestratorAgent, build_simulation_pipeline
from app.config import (
    APP_NAME,
    GOOGLE_CLOUD_LOCATION,
    GOOGLE_GENAI_USE_VERTEXAI,
)

_, project_id = google.auth.default()
os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
os.environ["GOOGLE_CLOUD_LOCATION"] = GOOGLE_CLOUD_LOCATION
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = GOOGLE_GENAI_USE_VERTEXAI

# Build the agent tree
simulation_pipeline = build_simulation_pipeline()
llm_dialogue_agent = create_llm_dialogue_agent()
dialogue_agent = DialogueAgent(
    name="dialogue_agent",
    sub_agents=[llm_dialogue_agent],
)

root_agent = OrchestratorAgent(
    name="orchestrator",
    description="Emergent Narrative Engine orchestrator. Coordinates NPC simulation and dialogue generation.",
    sub_agents=[simulation_pipeline, dialogue_agent],
)

app = App(root_agent=root_agent, name=APP_NAME)
