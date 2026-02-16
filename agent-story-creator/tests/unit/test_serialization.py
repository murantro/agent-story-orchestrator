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

"""Tests for app.agents.serialization module."""

from __future__ import annotations

import json

import numpy as np
import pytest

from app.agents.serialization import (
    deserialize_event,
    deserialize_npc,
    serialize_event,
    serialize_npc,
)
from app.models.events import LocalityScale, WorldEvent
from app.models.npc_status import (
    NPCVectorialStatus,
)


class TestSerializeNPC:
    def test_roundtrip_preserves_data(self):
        npc = NPCVectorialStatus(
            npc_id="test-1",
            name="Alice",
            archetype="merchant",
            energy=0.8,
            health=0.9,
            importance=0.7,
            location_id="market",
        )
        npc.emotion = np.array(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float32
        )
        npc.relationships = {"npc-2": 0.5, "npc-3": -0.3}
        npc.recent_memories = ["Sold a sword", "Heard a rumor"]

        data = serialize_npc(npc)
        restored = deserialize_npc(data)

        assert restored.npc_id == "test-1"
        assert restored.name == "Alice"
        assert restored.archetype == "merchant"
        assert restored.energy == pytest.approx(0.8)
        assert restored.health == pytest.approx(0.9)
        assert restored.importance == pytest.approx(0.7)
        assert restored.location_id == "market"
        np.testing.assert_array_almost_equal(restored.emotion, npc.emotion)
        np.testing.assert_array_almost_equal(restored.intention, npc.intention)
        np.testing.assert_array_almost_equal(restored.personality, npc.personality)
        assert restored.relationships == {"npc-2": 0.5, "npc-3": -0.3}
        assert restored.recent_memories == ["Sold a sword", "Heard a rumor"]

    def test_serialized_is_json_compatible(self):
        npc = NPCVectorialStatus(npc_id="json-test", name="Bob")
        data = serialize_npc(npc)
        json_str = json.dumps(data)
        assert isinstance(json_str, str)
        restored_data = json.loads(json_str)
        restored = deserialize_npc(restored_data)
        assert restored.npc_id == "json-test"
        assert restored.name == "Bob"

    def test_vectors_are_float32(self):
        npc = NPCVectorialStatus(npc_id="dtype-test")
        data = serialize_npc(npc)
        restored = deserialize_npc(data)
        assert restored.intention.dtype == np.float32
        assert restored.emotion.dtype == np.float32
        assert restored.personality.dtype == np.float32
        assert restored.social_influence.dtype == np.float32
        assert restored.environment.dtype == np.float32


class TestSerializeEvent:
    def test_roundtrip_preserves_data(self):
        event = WorldEvent(
            event_id="evt-1",
            source_npc_id="npc-1",
            event_type="murder",
            description="A terrible murder",
            origin_scale=LocalityScale.PERSONAL,
            current_scale=LocalityScale.CITY,
            location_id="tavern",
            timestamp=100.0,
            intensity=0.9,
        )
        event.emotion_impact = np.array(
            [0.0, 0.3, 0.5, 0.4, 0.2, 0.1, -0.2, 0.0], dtype=np.float32
        )

        data = serialize_event(event)
        restored = deserialize_event(data)

        assert restored.event_id == "evt-1"
        assert restored.source_npc_id == "npc-1"
        assert restored.event_type == "murder"
        assert restored.origin_scale == LocalityScale.PERSONAL
        assert restored.current_scale == LocalityScale.CITY
        assert restored.timestamp == pytest.approx(100.0)
        assert restored.intensity == pytest.approx(0.9)
        np.testing.assert_array_almost_equal(
            restored.emotion_impact, event.emotion_impact
        )

    def test_serialized_is_json_compatible(self):
        event = WorldEvent(event_id="json-evt", event_type="trade")
        data = serialize_event(event)
        json_str = json.dumps(data)
        assert isinstance(json_str, str)
        restored_data = json.loads(json_str)
        restored = deserialize_event(restored_data)
        assert restored.event_id == "json-evt"
        assert restored.event_type == "trade"

    def test_locality_scale_roundtrip(self):
        for scale in LocalityScale:
            event = WorldEvent(
                event_id=f"scale-{scale.name}",
                origin_scale=scale,
                current_scale=scale,
            )
            data = serialize_event(event)
            restored = deserialize_event(data)
            assert restored.origin_scale == scale
            assert restored.current_scale == scale
