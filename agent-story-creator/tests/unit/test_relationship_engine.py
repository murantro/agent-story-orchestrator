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

"""Tests for the relationship engine."""

import numpy as np

from app.models.npc_status import PERSONALITY_DIM, NPCVectorialStatus
from app.simulation.relationship_engine import RelationshipEngine


def _make_npc(**kwargs) -> NPCVectorialStatus:
    return NPCVectorialStatus(**kwargs)


def test_apply_delta_creates_symmetric_relationship():
    """Both NPCs should get the relationship entry."""
    engine = RelationshipEngine()
    a = _make_npc(name="Alice")
    b = _make_npc(name="Bob")
    engine.apply_delta(a, b, delta=0.3)
    assert b.npc_id in a.relationships
    assert a.npc_id in b.relationships


def test_apply_delta_positive_increases_affinity():
    """Positive delta should increase affinity."""
    engine = RelationshipEngine()
    a = _make_npc(name="Alice")
    b = _make_npc(name="Bob")
    engine.apply_delta(a, b, delta=0.3)
    assert a.relationships[b.npc_id] > 0.0
    assert b.relationships[a.npc_id] > 0.0


def test_apply_delta_negative_decreases_affinity():
    """Negative delta should decrease affinity."""
    engine = RelationshipEngine()
    a = _make_npc(name="Alice")
    b = _make_npc(name="Bob")
    engine.apply_delta(a, b, delta=-0.3)
    assert a.relationships[b.npc_id] < 0.0


def test_extreme_affinity_resists_change():
    """An NPC with affinity near 1.0 should be hard to push further."""
    engine = RelationshipEngine()
    a = _make_npc(name="Alice")
    b = _make_npc(name="Bob")
    # Set high affinity first
    a.relationships[b.npc_id] = 0.9
    b.relationships[a.npc_id] = 0.9
    engine.apply_delta(a, b, delta=0.5)
    # Should move less than 0.5 due to (1 - abs(old)) damping
    assert a.relationships[b.npc_id] < 0.9 + 0.5
    assert a.relationships[b.npc_id] <= 1.0


def test_affinity_clamped_to_range():
    """Affinity should never exceed [-1, 1]."""
    engine = RelationshipEngine(delta_scale=10.0)
    a = _make_npc(name="Alice")
    b = _make_npc(name="Bob")
    engine.apply_delta(a, b, delta=5.0)
    assert a.relationships[b.npc_id] <= 1.0
    engine.apply_delta(a, b, delta=-50.0)
    assert a.relationships[b.npc_id] >= -1.0


def test_decay_weakens_relationships():
    """Decay should push affinities toward zero."""
    engine = RelationshipEngine(decay_rate=0.5)
    a = _make_npc(name="Alice")
    a.relationships["bob"] = 0.4
    a.relationships["carol"] = -0.3
    engine.decay([a])
    assert abs(a.relationships["bob"]) < 0.4
    assert abs(a.relationships["carol"]) < 0.3


def test_decay_prunes_negligible_relationships():
    """Very weak affinities should be removed entirely."""
    engine = RelationshipEngine(decay_rate=0.5)
    a = _make_npc(name="Alice")
    a.relationships["bob"] = 0.015  # Will drop below 0.01 after decay
    engine.decay([a])
    assert "bob" not in a.relationships


def test_personality_compatibility_identical():
    """Identical personalities should be maximally compatible."""
    engine = RelationshipEngine()
    personality = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    a = _make_npc(name="Alice", personality=personality.copy())
    b = _make_npc(name="Bob", personality=personality.copy())
    compat = engine.personality_compatibility(a, b)
    assert compat == 1.0


def test_personality_compatibility_opposite():
    """Opposite personalities should have low compatibility."""
    engine = RelationshipEngine()
    a = _make_npc(
        name="Alice",
        personality=np.ones(PERSONALITY_DIM, dtype=np.float32),
    )
    b = _make_npc(
        name="Bob",
        personality=np.zeros(PERSONALITY_DIM, dtype=np.float32),
    )
    compat = engine.personality_compatibility(a, b)
    assert compat < 0.0
