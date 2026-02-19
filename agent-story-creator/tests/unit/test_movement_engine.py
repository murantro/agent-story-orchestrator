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

"""Tests for the movement engine."""

import numpy as np

from app.models.locations import Location, LocationGraph
from app.models.npc_status import INTENTION_DIM, NPCVectorialStatus
from app.simulation.movement_engine import MovementEngine


def _make_graph() -> LocationGraph:
    """Create a small test graph: tavern <-> forest <-> market."""
    g = LocationGraph()
    g.add_location(Location.from_type("tavern", "The Rusty Mug", "tavern", capacity=10))
    g.add_location(Location.from_type("forest", "Dark Woods", "forest", capacity=0))
    g.add_location(Location.from_type("market", "Grand Bazaar", "market", capacity=20))
    g.add_edge("tavern", "forest", travel_hours=2.0, danger=0.3)
    g.add_edge("forest", "market", travel_hours=3.0, danger=0.1)
    return g


def _make_npc(
    location_id: str = "tavern",
    dominant_intention_idx: int | None = None,
    energy: float = 1.0,
    **kwargs,
) -> NPCVectorialStatus:
    intention = np.ones(INTENTION_DIM, dtype=np.float32) / INTENTION_DIM
    if dominant_intention_idx is not None:
        intention = np.zeros(INTENTION_DIM, dtype=np.float32)
        intention[dominant_intention_idx] = 1.0
    return NPCVectorialStatus(
        name="Test",
        location_id=location_id,
        intention=intention,
        energy=energy,
        **kwargs,
    )


def test_score_destination_explorer_prefers_different():
    """An explorer should prefer destinations with different environments."""
    engine = MovementEngine()
    npc = _make_npc(dominant_intention_idx=3)  # explore
    npc.environment = np.array([0.7, 0.6, 0.9, 0.6], dtype=np.float32)
    # Very different environment
    far_env = np.array([0.1, 0.3, 0.2, 0.1], dtype=np.float32)
    # Similar environment
    near_env = np.array([0.6, 0.5, 0.8, 0.5], dtype=np.float32)

    score_far = engine.score_destination(npc, far_env, 0.0, 1.0)
    score_near = engine.score_destination(npc, near_env, 0.0, 1.0)
    assert score_far > score_near


def test_score_destination_survivor_prefers_safe():
    """A survivor should prefer safer destinations."""
    engine = MovementEngine()
    npc = _make_npc(dominant_intention_idx=0)  # survive
    npc.environment = np.array([0.3, 0.5, 0.5, 0.5], dtype=np.float32)
    safe_env = np.array([0.9, 0.5, 0.5, 0.5], dtype=np.float32)
    unsafe_env = np.array([0.1, 0.5, 0.5, 0.5], dtype=np.float32)

    score_safe = engine.score_destination(npc, safe_env, 0.0, 1.0)
    score_unsafe = engine.score_destination(npc, unsafe_env, 0.0, 1.0)
    assert score_safe > score_unsafe


def test_low_energy_npc_stays():
    """NPCs with very low energy should not move."""
    engine = MovementEngine(move_probability=1.0)
    graph = _make_graph()
    npc = _make_npc("tavern", dominant_intention_idx=3, energy=0.05)

    dest = engine.decide_movement(npc, graph)
    assert dest is None


def test_no_neighbors_stays():
    """NPCs at isolated locations should not move."""
    engine = MovementEngine(move_probability=1.0)
    graph = LocationGraph()
    graph.add_location(Location.from_type("island", "Lonely Island", "residential"))
    npc = _make_npc("island", dominant_intention_idx=3)

    dest = engine.decide_movement(npc, graph)
    assert dest is None


def test_tick_arrivals():
    """Travelers should arrive when game_time passes their arrival_time."""
    engine = MovementEngine(move_probability=0.0)  # Disable new journeys
    graph = _make_graph()
    npc = _make_npc("tavern")

    # Manually set up a travel state
    from app.simulation.movement_engine import TravelState

    engine._travelers[npc.npc_id] = TravelState(
        npc_id=npc.npc_id,
        origin_id="tavern",
        destination_id="forest",
        departure_time=0.0,
        arrival_time=2.0,
    )

    # At time 1.0, NPC should still be traveling
    engine.tick([npc], graph, game_time=1.0)
    assert npc.location_id == "tavern"
    assert engine.is_traveling(npc.npc_id)

    # At time 2.0, NPC should arrive (move_probability=0 prevents new journeys)
    engine.tick([npc], graph, game_time=2.0)
    assert npc.location_id == "forest"
    assert not engine.is_traveling(npc.npc_id)


def test_tick_traveling_npc_cannot_start_new_journey():
    """An NPC already traveling should not start a new journey."""
    engine = MovementEngine(move_probability=1.0)
    graph = _make_graph()
    npc = _make_npc("tavern", dominant_intention_idx=3)  # explore

    from app.simulation.movement_engine import TravelState

    engine._travelers[npc.npc_id] = TravelState(
        npc_id=npc.npc_id,
        origin_id="tavern",
        destination_id="forest",
        departure_time=0.0,
        arrival_time=5.0,
    )

    journeys = engine.tick([npc], graph, game_time=1.0)
    assert len(journeys) == 0


def test_tick_energy_cost():
    """Starting a journey should cost energy."""
    engine = MovementEngine(move_probability=1.0)
    graph = _make_graph()
    # Force explorer with high escape intention to guarantee movement
    npc = _make_npc("tavern", dominant_intention_idx=7, energy=1.0)  # escape
    npc.intention = np.zeros(INTENTION_DIM, dtype=np.float32)
    npc.intention[7] = 1.0  # escape

    old_energy = npc.energy
    journeys = engine.tick([npc], graph, game_time=0.0)
    if journeys:
        assert npc.energy < old_energy


def test_capacity_prevents_movement():
    """NPCs should not move to locations at capacity."""
    engine = MovementEngine(move_probability=1.0)
    graph = LocationGraph()
    graph.add_location(Location.from_type("a", "Place A", "plaza", capacity=0))
    graph.add_location(Location.from_type("b", "Place B", "tavern", capacity=1))
    graph.add_edge("a", "b", travel_hours=1.0)

    # Already 1 NPC at B (at capacity)
    npc_at_b = _make_npc("b")
    # NPC at A wants to escape to B
    npc_at_a = _make_npc("a", dominant_intention_idx=7)  # escape
    npc_at_a.intention = np.zeros(INTENTION_DIM, dtype=np.float32)
    npc_at_a.intention[7] = 1.0

    journeys = engine.tick([npc_at_b, npc_at_a], graph, game_time=0.0)
    # Should not move to B because it's full
    for j in journeys:
        assert j.destination_id != "b"
