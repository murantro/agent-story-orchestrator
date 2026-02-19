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

"""Tests for the environment engine."""

import numpy as np

from app.models.locations import Location, LocationGraph
from app.models.npc_status import ENVIRONMENT_DIM, NPCVectorialStatus
from app.simulation.environment_engine import EnvironmentEngine


def _make_graph() -> LocationGraph:
    """Create a small test graph."""
    g = LocationGraph()
    g.add_location(Location.from_type("tavern", "The Rusty Mug", "tavern", capacity=10))
    g.add_location(Location.from_type("forest", "Dark Woods", "forest", capacity=0))
    g.add_edge("tavern", "forest", travel_hours=2.0)
    return g


def _make_npc(location_id: str = "tavern", **kwargs) -> NPCVectorialStatus:
    return NPCVectorialStatus(
        name="Test",
        location_id=location_id,
        environment=np.zeros(ENVIRONMENT_DIM, dtype=np.float32),
        **kwargs,
    )


def test_tick_blends_toward_location_environment():
    """NPC environment should move toward location environment."""
    engine = EnvironmentEngine(blend_rate=0.5)
    graph = _make_graph()
    npc = _make_npc("tavern")
    old_env = npc.environment.copy()

    engine.tick([npc], graph)

    tavern = graph.get_location("tavern")
    # Environment should have moved toward tavern's environment
    assert not np.array_equal(npc.environment, old_env)
    # Should be closer to tavern environment than before
    new_dist = np.linalg.norm(npc.environment - tavern.environment)
    old_dist = np.linalg.norm(old_env - tavern.environment)
    assert new_dist < old_dist


def test_tick_full_blend_snaps_to_location():
    """With blend_rate=1.0, NPC environment should equal location environment."""
    engine = EnvironmentEngine(blend_rate=1.0)
    graph = _make_graph()
    npc = _make_npc("forest")

    engine.tick([npc], graph)

    forest = graph.get_location("forest")
    # Should match forest environment (except crowding is dynamic)
    # safety, resource_abundance, weather_comfort should match exactly
    np.testing.assert_allclose(npc.environment[:3], forest.environment[:3], atol=0.01)


def test_crowding_from_capacity():
    """Crowding should reflect NPC count relative to capacity."""
    engine = EnvironmentEngine(blend_rate=1.0)
    graph = _make_graph()
    # Put 5 NPCs in tavern (capacity=10) -> crowding=0.5
    npcs = [_make_npc("tavern") for _ in range(5)]

    engine.tick(npcs, graph)

    # Crowding index is 3
    for npc in npcs:
        assert abs(npc.environment[3] - 0.5) < 0.01


def test_crowding_unlimited_capacity():
    """Forest has capacity=0 (unlimited), crowding should be soft-scaled."""
    engine = EnvironmentEngine(blend_rate=1.0)
    graph = _make_graph()
    npcs = [_make_npc("forest") for _ in range(10)]

    engine.tick(npcs, graph)

    # 10 NPCs / 20 (soft scale) = 0.5 crowding
    assert abs(npcs[0].environment[3] - 0.5) < 0.01


def test_unknown_location_skipped():
    """NPCs at unknown locations should not be modified."""
    engine = EnvironmentEngine(blend_rate=1.0)
    graph = _make_graph()
    npc = _make_npc("nonexistent")
    old_env = npc.environment.copy()

    engine.tick([npc], graph)

    np.testing.assert_array_equal(npc.environment, old_env)


def test_environment_clamped():
    """Environment values should stay in [0, 1]."""
    engine = EnvironmentEngine(blend_rate=1.0)
    graph = _make_graph()
    npc = _make_npc("tavern")
    npc.environment = np.array([2.0, -1.0, 0.5, 0.5], dtype=np.float32)

    engine.tick([npc], graph)

    assert np.all(npc.environment >= 0.0)
    assert np.all(npc.environment <= 1.0)
