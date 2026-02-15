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

from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.models import Gemini
from google.genai import types

import os
import google.auth

from app.tools.tools_general import get_current_time, get_weather
from app.prompts.prompt_general import INSTRUCTION
from app.config import (
    AGENT_NAME,
    MODEL_NAME,
    RETRY_ATTEMPTS,
    APP_NAME,
    GOOGLE_CLOUD_LOCATION,
    GOOGLE_GENAI_USE_VERTEXAI,
)

_, project_id = google.auth.default()
os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
os.environ["GOOGLE_CLOUD_LOCATION"] = GOOGLE_CLOUD_LOCATION
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = GOOGLE_GENAI_USE_VERTEXAI


root_agent = Agent(
    name=AGENT_NAME,
    model=Gemini(
        model=MODEL_NAME,
        retry_options=types.HttpRetryOptions(attempts=RETRY_ATTEMPTS),
    ),
    instruction=INSTRUCTION,
    tools=[get_weather, get_current_time],
)

app = App(root_agent=root_agent, name=APP_NAME)
