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

"""Pydantic request/response schemas for the Game API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CreateNPCRequest(BaseModel):
    """Request to create a new NPC."""

    name: str
    archetype: str = "generic"
    location_id: str = "default"
    personality: list[float] | None = Field(
        default=None,
        description="5-dim Big Five personality vector. If None, uses uniform default.",
    )
    importance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="How plot-relevant this NPC is (0-1).",
    )


class NPCResponse(BaseModel):
    """Response with NPC state summary."""

    npc_id: str
    name: str
    archetype: str
    dominant_intention: str
    dominant_emotion: str
    energy: float
    health: float
    importance: float
    location_id: str


class NPCDetailResponse(NPCResponse):
    """Detailed NPC response with full vector data."""

    intention: list[float]
    emotion: list[float]
    personality: list[float]
    social_influence: list[float]
    environment: list[float]
    relationships: dict[str, float]
    recent_memories: list[str]


class SubmitEventRequest(BaseModel):
    """Request to submit a world event."""

    event_type: str
    description: str = ""
    source_npc_id: str | None = None
    location_id: str = "default"
    intensity: float = Field(default=1.0, ge=0.0, le=1.0)
    emotion_impact: list[float] | None = Field(
        default=None,
        description="8-dim Plutchik emotion impact vector.",
    )
    social_impact: list[float] | None = Field(
        default=None,
        description="6-dim social influence impact vector.",
    )


class SubmitEventResponse(BaseModel):
    """Response after submitting an event."""

    event_id: str
    scheduled_deliveries: int


class TickRequest(BaseModel):
    """Request to advance the simulation."""

    delta_hours: float = Field(
        default=1.0,
        gt=0.0,
        description="In-game hours to advance.",
    )


class TickResponse(BaseModel):
    """Response after a simulation tick."""

    game_time: float
    npcs_updated: int
    events_delivered: int
    events_pending: int
    interactions_resolved: int = 0


class DialogueRequest(BaseModel):
    """Request to generate dialogue for an NPC."""

    npc_id: str
    player_message: str | None = None
    player_initiated: bool = True
    is_quest_critical: bool = False
    turn_count: int = 0


class DialogueResponse(BaseModel):
    """Response with generated NPC dialogue."""

    npc_id: str
    text: str
    tier: str
    memories_used: int = 0


class TickRunnerStatusResponse(BaseModel):
    """Response with background tick runner status."""

    running: bool
    ticks_completed: int
    interval_seconds: float


class GameTimeResponse(BaseModel):
    """Response with current game time."""

    game_time: float
    npc_count: int


class WorldSnapshotResponse(BaseModel):
    """Full world state snapshot for save/load."""

    game_time: float
    npcs: dict
