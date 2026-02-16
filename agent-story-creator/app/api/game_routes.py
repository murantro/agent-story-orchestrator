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

"""Game API routes for Unity/game client communication.

Provides REST endpoints for NPC management, event submission,
simulation ticks, dialogue generation, and world state save/load.
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException

from app.dialogue.template_engine import TemplateEngine
from app.dialogue.tier_selector import DialogueTier, InteractionContext, select_tier
from app.models.events import WorldEvent
from app.models.npc_status import (
    EMOTION_DIM,
    PERSONALITY_DIM,
    SOCIAL_INFLUENCE_DIM,
    NPCVectorialStatus,
)
from app.world.world_state import WorldStateManager

from .schemas import (
    CreateNPCRequest,
    DialogueRequest,
    DialogueResponse,
    GameTimeResponse,
    NPCDetailResponse,
    NPCResponse,
    SubmitEventRequest,
    SubmitEventResponse,
    TickRequest,
    TickResponse,
    WorldSnapshotResponse,
)

# Module-level world state singleton (set during app startup)
_world: WorldStateManager | None = None


def get_world() -> WorldStateManager:
    """Get the world state manager singleton.

    Raises:
        RuntimeError: If world state not initialized.
    """
    if _world is None:
        raise RuntimeError("World state not initialized.")
    return _world


def set_world(world: WorldStateManager) -> None:
    """Set the world state manager singleton."""
    global _world
    _world = world


game_router = APIRouter(tags=["game"])

# Template engine for dialogue generation
_template_engine = TemplateEngine()


def _npc_to_response(npc: NPCVectorialStatus) -> NPCResponse:
    """Convert NPC to summary response."""
    return NPCResponse(
        npc_id=npc.npc_id,
        name=npc.name,
        archetype=npc.archetype,
        dominant_intention=npc.dominant_intention(),
        dominant_emotion=npc.dominant_emotion(),
        energy=npc.energy,
        health=npc.health,
        importance=npc.importance,
        location_id=npc.location_id,
    )


def _npc_to_detail_response(npc: NPCVectorialStatus) -> NPCDetailResponse:
    """Convert NPC to detailed response with vectors."""
    return NPCDetailResponse(
        npc_id=npc.npc_id,
        name=npc.name,
        archetype=npc.archetype,
        dominant_intention=npc.dominant_intention(),
        dominant_emotion=npc.dominant_emotion(),
        energy=npc.energy,
        health=npc.health,
        importance=npc.importance,
        location_id=npc.location_id,
        intention=npc.intention.tolist(),
        emotion=npc.emotion.tolist(),
        personality=npc.personality.tolist(),
        social_influence=npc.social_influence.tolist(),
        environment=npc.environment.tolist(),
        relationships=dict(npc.relationships),
        recent_memories=list(npc.recent_memories),
    )


# --- NPC Endpoints ---


@game_router.post("/npc", response_model=NPCDetailResponse, status_code=201)
def create_npc(req: CreateNPCRequest) -> NPCDetailResponse:
    """Create a new NPC in the world."""
    world = get_world()
    npc = NPCVectorialStatus(
        name=req.name,
        archetype=req.archetype,
        location_id=req.location_id,
        importance=req.importance,
    )
    if req.personality is not None:
        if len(req.personality) != PERSONALITY_DIM:
            raise HTTPException(
                status_code=400,
                detail=f"personality must have {PERSONALITY_DIM} dimensions.",
            )
        npc.personality = np.array(req.personality, dtype=np.float32)

    try:
        world.add_npc(npc)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e

    return _npc_to_detail_response(npc)


@game_router.get("/npc/{npc_id}", response_model=NPCDetailResponse)
def get_npc(npc_id: str) -> NPCDetailResponse:
    """Get detailed NPC state by ID."""
    world = get_world()
    npc = world.get_npc(npc_id)
    if npc is None:
        raise HTTPException(status_code=404, detail=f"NPC {npc_id!r} not found.")
    return _npc_to_detail_response(npc)


@game_router.get("/npcs", response_model=list[NPCResponse])
def list_npcs(location_id: str | None = None) -> list[NPCResponse]:
    """List all NPCs, optionally filtered by location."""
    world = get_world()
    npcs = world.list_npcs(location_id=location_id)
    return [_npc_to_response(n) for n in npcs]


@game_router.delete("/npc/{npc_id}")
def delete_npc(npc_id: str) -> dict[str, str]:
    """Remove an NPC from the world."""
    world = get_world()
    if not world.remove_npc(npc_id):
        raise HTTPException(status_code=404, detail=f"NPC {npc_id!r} not found.")
    return {"status": "deleted", "npc_id": npc_id}


# --- Event Endpoints ---


@game_router.post("/event", response_model=SubmitEventResponse, status_code=201)
def submit_event(req: SubmitEventRequest) -> SubmitEventResponse:
    """Submit a world event for propagation."""
    world = get_world()
    event = WorldEvent(
        source_npc_id=req.source_npc_id,
        event_type=req.event_type,
        description=req.description,
        location_id=req.location_id,
        intensity=req.intensity,
    )
    if req.emotion_impact is not None:
        if len(req.emotion_impact) != EMOTION_DIM:
            raise HTTPException(
                status_code=400,
                detail=f"emotion_impact must have {EMOTION_DIM} dimensions.",
            )
        event.emotion_impact = np.array(req.emotion_impact, dtype=np.float32)
    if req.social_impact is not None:
        if len(req.social_impact) != SOCIAL_INFLUENCE_DIM:
            raise HTTPException(
                status_code=400,
                detail=f"social_impact must have {SOCIAL_INFLUENCE_DIM} dimensions.",
            )
        event.social_impact = np.array(req.social_impact, dtype=np.float32)

    count = world.submit_event(event)
    return SubmitEventResponse(event_id=event.event_id, scheduled_deliveries=count)


# --- Simulation Endpoints ---


@game_router.post("/tick", response_model=TickResponse)
async def tick(req: TickRequest) -> TickResponse:
    """Advance the simulation by delta_hours."""
    world = get_world()
    result = await world.tick(delta_hours=req.delta_hours)
    return TickResponse(
        game_time=result.game_time,
        npcs_updated=result.npcs_updated,
        events_delivered=result.events_delivered,
        events_pending=result.events_pending,
    )


@game_router.get("/time", response_model=GameTimeResponse)
def get_time() -> GameTimeResponse:
    """Get current in-game time and NPC count."""
    world = get_world()
    return GameTimeResponse(game_time=world.game_time, npc_count=world.npc_count)


# --- Dialogue Endpoints ---


@game_router.post("/dialogue", response_model=DialogueResponse)
def generate_dialogue(req: DialogueRequest) -> DialogueResponse:
    """Generate dialogue for an NPC.

    For TEMPLATE tier, generates immediately. For LLM tiers,
    returns the template dialogue as a fallback (LLM integration
    requires the full ADK pipeline).
    """
    world = get_world()
    npc = world.get_npc(req.npc_id)
    if npc is None:
        raise HTTPException(status_code=404, detail=f"NPC {req.npc_id!r} not found.")

    context = InteractionContext(
        player_initiated=req.player_initiated,
        is_quest_critical=req.is_quest_critical,
        turn_count=req.turn_count,
        local_llm_available=False,
    )
    tier = select_tier(npc, context)

    if tier == DialogueTier.TEMPLATE:
        text = _template_engine.generate(npc)
    else:
        # For LLM tiers, generate template as fallback
        # Full LLM dialogue requires the ADK agent pipeline
        text = _template_engine.generate(npc)

    return DialogueResponse(npc_id=npc.npc_id, text=text, tier=tier.value)


# --- World State Endpoints ---


@game_router.get("/world/snapshot", response_model=WorldSnapshotResponse)
def get_snapshot() -> WorldSnapshotResponse:
    """Get a full snapshot of the world state for saving."""
    world = get_world()
    data = world.snapshot()
    return WorldSnapshotResponse(game_time=data["game_time"], npcs=data["npcs"])


@game_router.post("/world/restore")
def restore_snapshot(data: WorldSnapshotResponse) -> dict[str, str]:
    """Restore world state from a snapshot."""
    world = get_world()
    world.restore({"game_time": data.game_time, "npcs": data.npcs})
    return {"status": "restored", "npc_count": str(world.npc_count)}
