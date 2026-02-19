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

"""Tests for the interaction engine."""

import numpy as np

from app.models.npc_status import (
    INTENTION_DIM,
    NPCVectorialStatus,
)
from app.simulation.interaction_engine import InteractionEngine


def _make_npc(
    name: str = "NPC",
    location_id: str = "tavern",
    dominant_intention_idx: int | None = None,
    energy: float = 1.0,
    **kwargs,
) -> NPCVectorialStatus:
    """Create an NPC with optional dominant intention."""
    intention = np.ones(INTENTION_DIM, dtype=np.float32) / INTENTION_DIM
    if dominant_intention_idx is not None:
        intention = np.zeros(INTENTION_DIM, dtype=np.float32)
        intention[dominant_intention_idx] = 1.0
    return NPCVectorialStatus(
        name=name,
        location_id=location_id,
        intention=intention,
        energy=energy,
        **kwargs,
    )


def test_resolve_interaction_type_socialize_socialize():
    """Two socializers should have a friendly chat."""
    engine = InteractionEngine()
    a = _make_npc("Alice", dominant_intention_idx=1)  # socialize
    b = _make_npc("Bob", dominant_intention_idx=1)  # socialize
    assert engine.resolve_interaction_type(a, b) == "friendly_chat"


def test_resolve_interaction_type_dominate_dominate():
    """Two dominators should have a conflict."""
    engine = InteractionEngine()
    a = _make_npc("Alice", dominant_intention_idx=5)  # dominate
    b = _make_npc("Bob", dominant_intention_idx=5)  # dominate
    assert engine.resolve_interaction_type(a, b) == "conflict"


def test_resolve_interaction_type_fallback():
    """Unmatched pair should fall back to casual_encounter."""
    engine = InteractionEngine()
    a = _make_npc("Alice", dominant_intention_idx=0)  # survive
    b = _make_npc("Bob", dominant_intention_idx=3)  # explore
    assert engine.resolve_interaction_type(a, b) == "casual_encounter"


def test_resolve_interaction_type_asymmetric():
    """Asymmetric pair (dominate+survive) should produce intimidation."""
    engine = InteractionEngine()
    a = _make_npc("Bully", dominant_intention_idx=5)  # dominate
    b = _make_npc("Victim", dominant_intention_idx=0)  # survive
    assert engine.resolve_interaction_type(a, b) == "intimidation"


def test_probability_zero_for_low_energy():
    """NPCs with energy below threshold should not interact."""
    engine = InteractionEngine(min_energy=0.2)
    a = _make_npc("Alice", energy=0.05)
    b = _make_npc("Bob", energy=1.0)
    assert engine.compute_interaction_probability(a, b) == 0.0


def test_probability_positive_for_same_intentions():
    """NPCs with aligned intentions should have positive interaction prob."""
    engine = InteractionEngine(interaction_rate=1.0)
    a = _make_npc("Alice", dominant_intention_idx=1)  # socialize
    b = _make_npc("Bob", dominant_intention_idx=1)  # socialize
    prob = engine.compute_interaction_probability(a, b)
    assert prob > 0.0


def test_resolve_generates_event():
    """resolve() should produce a valid WorldEvent."""
    engine = InteractionEngine()
    a = _make_npc("Alice", dominant_intention_idx=1)
    b = _make_npc("Bob", dominant_intention_idx=1)
    outcome = engine.resolve(a, b, game_time=100.0)
    assert outcome.event.timestamp == 100.0
    assert "Alice" in outcome.event.description
    assert "Bob" in outcome.event.description
    assert outcome.event.event_type.startswith("interaction_")
    assert outcome.event.location_id == "tavern"


def test_resolve_relationship_delta_sign():
    """Friendly interactions should produce positive delta; conflicts negative."""
    engine = InteractionEngine()
    # Friendly chat
    a = _make_npc("Alice", dominant_intention_idx=1)  # socialize
    b = _make_npc("Bob", dominant_intention_idx=1)  # socialize
    outcome = engine.resolve(a, b, game_time=0.0)
    assert outcome.relationship_delta > 0.0

    # Conflict
    c = _make_npc("Chuck", dominant_intention_idx=5)  # dominate
    d = _make_npc("Dave", dominant_intention_idx=5)  # dominate
    outcome2 = engine.resolve(c, d, game_time=0.0)
    assert outcome2.relationship_delta < 0.0


def test_tick_only_pairs_same_location():
    """NPCs in different locations should NOT interact."""
    engine = InteractionEngine(interaction_rate=100.0)  # Force interactions
    a = _make_npc("Alice", location_id="tavern", dominant_intention_idx=1)
    b = _make_npc("Bob", location_id="market", dominant_intention_idx=1)
    outcomes = engine.tick([a, b], game_time=0.0)
    assert len(outcomes) == 0


def test_tick_respects_max_per_location():
    """Should not exceed max_per_location interactions."""
    engine = InteractionEngine(
        interaction_rate=100.0, max_per_location=2
    )
    npcs = [
        _make_npc(f"NPC_{i}", location_id="plaza", dominant_intention_idx=1)
        for i in range(10)
    ]
    outcomes = engine.tick(npcs, game_time=0.0)
    assert len(outcomes) <= 2


def test_tick_each_npc_interacts_at_most_once():
    """Each NPC should participate in at most one interaction per tick."""
    engine = InteractionEngine(interaction_rate=100.0)
    npcs = [
        _make_npc(f"NPC_{i}", location_id="plaza", dominant_intention_idx=1)
        for i in range(6)
    ]
    outcomes = engine.tick(npcs, game_time=0.0)
    participants = []
    for o in outcomes:
        participants.append(o.npc_a_id)
        participants.append(o.npc_b_id)
    assert len(participants) == len(set(participants)), (
        "No NPC should appear in two interactions"
    )
