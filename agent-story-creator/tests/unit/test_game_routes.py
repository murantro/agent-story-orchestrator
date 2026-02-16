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

"""Tests for Game API routes using FastAPI TestClient."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.game_routes import game_router, set_world
from app.world.world_state import WorldStateManager


@pytest.fixture()
def client():
    """Create a test client with a fresh WorldStateManager."""
    app = FastAPI()
    app.include_router(game_router, prefix="/api/game")
    world = WorldStateManager()
    set_world(world)
    yield TestClient(app)
    set_world(None)


class TestNPCEndpoints:
    def test_create_npc(self, client):
        resp = client.post(
            "/api/game/npc",
            json={"name": "Guard", "archetype": "guard", "importance": 0.7},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "Guard"
        assert data["archetype"] == "guard"
        assert data["importance"] == 0.7
        assert "npc_id" in data

    def test_get_npc(self, client):
        create_resp = client.post("/api/game/npc", json={"name": "Merchant"})
        npc_id = create_resp.json()["npc_id"]

        resp = client.get(f"/api/game/npc/{npc_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["npc_id"] == npc_id
        assert data["name"] == "Merchant"
        assert "intention" in data  # Detail response includes vectors

    def test_get_npc_not_found(self, client):
        resp = client.get("/api/game/npc/nonexistent")
        assert resp.status_code == 404

    def test_list_npcs(self, client):
        client.post("/api/game/npc", json={"name": "A", "location_id": "town"})
        client.post("/api/game/npc", json={"name": "B", "location_id": "forest"})

        resp = client.get("/api/game/npcs")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

        resp = client.get("/api/game/npcs?location_id=town")
        assert len(resp.json()) == 1

    def test_delete_npc(self, client):
        create_resp = client.post("/api/game/npc", json={"name": "Temp"})
        npc_id = create_resp.json()["npc_id"]

        resp = client.delete(f"/api/game/npc/{npc_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

        resp = client.get(f"/api/game/npc/{npc_id}")
        assert resp.status_code == 404

    def test_create_npc_with_personality(self, client):
        resp = client.post(
            "/api/game/npc",
            json={
                "name": "Scholar",
                "personality": [0.9, 0.8, 0.3, 0.7, 0.2],
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["personality"][0] == pytest.approx(0.9)

    def test_create_npc_invalid_personality_dims(self, client):
        resp = client.post(
            "/api/game/npc",
            json={"name": "Bad", "personality": [0.5, 0.5]},
        )
        assert resp.status_code == 400


class TestEventEndpoints:
    def test_submit_event(self, client):
        resp = client.post(
            "/api/game/event",
            json={
                "event_type": "murder",
                "description": "A terrible crime",
                "intensity": 0.9,
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "event_id" in data
        assert data["scheduled_deliveries"] >= 1

    def test_submit_event_with_impacts(self, client):
        resp = client.post(
            "/api/game/event",
            json={
                "event_type": "celebration",
                "emotion_impact": [0.5, 0.0, 0.0, 0.0, 0.2, 0.0, 0.3, 0.1],
                "social_impact": [0.1, 0.0, 0.2, 0.0, 0.0, 0.0],
            },
        )
        assert resp.status_code == 201


class TestSimulationEndpoints:
    def test_tick(self, client):
        client.post("/api/game/npc", json={"name": "Guard"})
        resp = client.post("/api/game/tick", json={"delta_hours": 2.0})
        assert resp.status_code == 200
        data = resp.json()
        assert data["game_time"] == pytest.approx(2.0)
        assert data["npcs_updated"] == 1

    def test_get_time(self, client):
        resp = client.get("/api/game/time")
        assert resp.status_code == 200
        data = resp.json()
        assert data["game_time"] == pytest.approx(0.0)
        assert data["npc_count"] == 0


class TestDialogueEndpoints:
    def test_dialogue_template_tier(self, client):
        create_resp = client.post("/api/game/npc", json={"name": "Villager"})
        npc_id = create_resp.json()["npc_id"]

        resp = client.post(
            "/api/game/dialogue",
            json={"npc_id": npc_id, "player_initiated": False},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["tier"] == "template"
        assert data["text"]

    def test_dialogue_npc_not_found(self, client):
        resp = client.post(
            "/api/game/dialogue",
            json={"npc_id": "missing", "player_initiated": False},
        )
        assert resp.status_code == 404


class TestWorldStateEndpoints:
    def test_snapshot_and_restore(self, client):
        client.post("/api/game/npc", json={"name": "Guard", "archetype": "guard"})
        client.post("/api/game/tick", json={"delta_hours": 5.0})

        # Take snapshot
        snap_resp = client.get("/api/game/world/snapshot")
        assert snap_resp.status_code == 200
        snapshot = snap_resp.json()
        assert snapshot["game_time"] == pytest.approx(5.0)
        assert len(snapshot["npcs"]) == 1

        # Restore to new world
        set_world(WorldStateManager())
        restore_resp = client.post("/api/game/world/restore", json=snapshot)
        assert restore_resp.status_code == 200
        assert restore_resp.json()["npc_count"] == "1"

        # Verify restored state
        time_resp = client.get("/api/game/time")
        assert time_resp.json()["game_time"] == pytest.approx(5.0)
        assert time_resp.json()["npc_count"] == 1
