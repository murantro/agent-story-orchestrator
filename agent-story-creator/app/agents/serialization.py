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

"""Serialization helpers for passing NPC/event data through session.state.

ADK session.state values must be JSON-serializable (no numpy arrays).
These functions convert between domain objects and plain dicts.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.models.events import LocalityScale, WorldEvent
from app.models.npc_status import NPCVectorialStatus


def serialize_npc(npc: NPCVectorialStatus) -> dict[str, Any]:
    """Convert an NPCVectorialStatus to a JSON-serializable dict.

    Args:
        npc: The NPC state to serialize.

    Returns:
        Plain dict with lists instead of numpy arrays.
    """
    return {
        "npc_id": npc.npc_id,
        "name": npc.name,
        "archetype": npc.archetype,
        "intention": npc.intention.tolist(),
        "emotion": npc.emotion.tolist(),
        "personality": npc.personality.tolist(),
        "social_influence": npc.social_influence.tolist(),
        "environment": npc.environment.tolist(),
        "energy": npc.energy,
        "health": npc.health,
        "importance": npc.importance,
        "relationships": dict(npc.relationships),
        "recent_memories": list(npc.recent_memories),
        "location_id": npc.location_id,
        "activity": npc.activity,
    }


def deserialize_npc(data: dict[str, Any]) -> NPCVectorialStatus:
    """Reconstruct an NPCVectorialStatus from a serialized dict.

    Args:
        data: Dict previously produced by serialize_npc.

    Returns:
        Reconstructed NPCVectorialStatus with numpy arrays.
    """
    return NPCVectorialStatus(
        npc_id=data["npc_id"],
        name=data["name"],
        archetype=data["archetype"],
        intention=np.array(data["intention"], dtype=np.float32),
        emotion=np.array(data["emotion"], dtype=np.float32),
        personality=np.array(data["personality"], dtype=np.float32),
        social_influence=np.array(data["social_influence"], dtype=np.float32),
        environment=np.array(data["environment"], dtype=np.float32),
        energy=data["energy"],
        health=data["health"],
        importance=data["importance"],
        relationships=dict(data["relationships"]),
        recent_memories=list(data["recent_memories"]),
        location_id=data["location_id"],
        activity=data.get("activity", "idle"),
    )


def serialize_event(event: WorldEvent) -> dict[str, Any]:
    """Convert a WorldEvent to a JSON-serializable dict.

    Args:
        event: The world event to serialize.

    Returns:
        Plain dict with lists instead of numpy arrays.
    """
    return {
        "event_id": event.event_id,
        "source_npc_id": event.source_npc_id,
        "event_type": event.event_type,
        "description": event.description,
        "origin_scale": event.origin_scale.value,
        "current_scale": event.current_scale.value,
        "location_id": event.location_id,
        "timestamp": event.timestamp,
        "intensity": event.intensity,
        "emotion_impact": event.emotion_impact.tolist(),
        "social_impact": event.social_impact.tolist(),
    }


def deserialize_event(data: dict[str, Any]) -> WorldEvent:
    """Reconstruct a WorldEvent from a serialized dict.

    Args:
        data: Dict previously produced by serialize_event.

    Returns:
        Reconstructed WorldEvent with numpy arrays.
    """
    return WorldEvent(
        event_id=data["event_id"],
        source_npc_id=data["source_npc_id"],
        event_type=data["event_type"],
        description=data["description"],
        origin_scale=LocalityScale(data["origin_scale"]),
        current_scale=LocalityScale(data["current_scale"]),
        location_id=data["location_id"],
        timestamp=data["timestamp"],
        intensity=data["intensity"],
        emotion_impact=np.array(data["emotion_impact"], dtype=np.float32),
        social_impact=np.array(data["social_impact"], dtype=np.float32),
    )
