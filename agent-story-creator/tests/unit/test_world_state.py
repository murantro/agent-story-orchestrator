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

"""Tests for WorldStateManager."""

from __future__ import annotations

import numpy as np
import pytest

from app.models.events import WorldEvent
from app.models.npc_status import NPCVectorialStatus
from app.world.world_state import WorldStateManager


class TestNPCCRUD:
    def test_add_and_get_npc(self):
        world = WorldStateManager()
        npc = NPCVectorialStatus(npc_id="npc-1", name="Guard")
        world.add_npc(npc)

        result = world.get_npc("npc-1")
        assert result is not None
        assert result.name == "Guard"
        assert world.npc_count == 1

    def test_get_nonexistent_returns_none(self):
        world = WorldStateManager()
        assert world.get_npc("missing") is None

    def test_duplicate_id_raises(self):
        world = WorldStateManager()
        npc = NPCVectorialStatus(npc_id="npc-1", name="Guard")
        world.add_npc(npc)

        npc2 = NPCVectorialStatus(npc_id="npc-1", name="Other")
        with pytest.raises(ValueError, match="already exists"):
            world.add_npc(npc2)

    def test_max_npcs_raises(self):
        world = WorldStateManager(max_npcs=2)
        world.add_npc(NPCVectorialStatus(npc_id="a", name="A"))
        world.add_npc(NPCVectorialStatus(npc_id="b", name="B"))

        with pytest.raises(ValueError, match="Maximum"):
            world.add_npc(NPCVectorialStatus(npc_id="c", name="C"))

    def test_list_npcs(self):
        world = WorldStateManager()
        world.add_npc(NPCVectorialStatus(npc_id="a", name="A", location_id="town"))
        world.add_npc(NPCVectorialStatus(npc_id="b", name="B", location_id="forest"))
        world.add_npc(NPCVectorialStatus(npc_id="c", name="C", location_id="town"))

        all_npcs = world.list_npcs()
        assert len(all_npcs) == 3

        town_npcs = world.list_npcs(location_id="town")
        assert len(town_npcs) == 2

        forest_npcs = world.list_npcs(location_id="forest")
        assert len(forest_npcs) == 1

    def test_remove_npc(self):
        world = WorldStateManager()
        world.add_npc(NPCVectorialStatus(npc_id="a", name="A"))

        assert world.remove_npc("a") is True
        assert world.npc_count == 0
        assert world.get_npc("a") is None

    def test_remove_nonexistent_returns_false(self):
        world = WorldStateManager()
        assert world.remove_npc("missing") is False


class TestEvents:
    def test_submit_event_sets_timestamp(self):
        world = WorldStateManager(game_time=50.0)
        event = WorldEvent(event_type="trade", intensity=0.5)
        count = world.submit_event(event)

        assert count >= 1
        assert event.timestamp == 50.0

    def test_submit_event_preserves_timestamp(self):
        world = WorldStateManager(game_time=50.0)
        event = WorldEvent(event_type="trade", timestamp=30.0, intensity=0.5)
        world.submit_event(event)
        assert event.timestamp == 30.0

    def test_get_due_events(self):
        world = WorldStateManager(game_time=10.0)
        event = WorldEvent(event_type="trade", timestamp=10.0, intensity=0.5)
        world.submit_event(event)

        due = world.get_due_events()
        assert len(due) >= 1


class TestTick:
    @pytest.mark.asyncio
    async def test_tick_advances_time(self):
        world = WorldStateManager(game_time=0.0)
        result = await world.tick(delta_hours=5.0)
        assert result.game_time == pytest.approx(5.0)
        assert world.game_time == pytest.approx(5.0)

    @pytest.mark.asyncio
    async def test_tick_updates_npcs(self):
        world = WorldStateManager()
        npc = NPCVectorialStatus(npc_id="npc-1", name="Guard")
        npc.emotion = np.array(
            [0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
        )
        world.add_npc(npc)

        result = await world.tick(delta_hours=1.0)
        assert result.npcs_updated == 1

        # Emotion should have decayed toward baseline
        updated = world.get_npc("npc-1")
        assert updated is not None
        assert updated.emotion[0] < 0.8  # Joy decayed

    @pytest.mark.asyncio
    async def test_tick_delivers_events(self):
        world = WorldStateManager(game_time=0.0)
        npc = NPCVectorialStatus(npc_id="npc-1", name="Guard")
        world.add_npc(npc)

        event = WorldEvent(
            event_type="murder",
            timestamp=1.0,
            intensity=0.9,
            emotion_impact=np.array(
                [0.0, 0.3, 0.5, 0.4, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
            ),
        )
        world.submit_event(event)

        result = await world.tick(delta_hours=2.0)
        assert result.events_delivered >= 1


class TestSnapshot:
    def test_snapshot_roundtrip(self):
        world = WorldStateManager(game_time=42.0)
        world.add_npc(
            NPCVectorialStatus(npc_id="npc-1", name="Guard", archetype="guard")
        )
        world.add_npc(
            NPCVectorialStatus(npc_id="npc-2", name="Merchant", archetype="merchant")
        )

        snap = world.snapshot()

        world2 = WorldStateManager()
        world2.restore(snap)

        assert world2.game_time == pytest.approx(42.0)
        assert world2.npc_count == 2
        assert world2.get_npc("npc-1").name == "Guard"
        assert world2.get_npc("npc-2").archetype == "merchant"
