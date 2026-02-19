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

"""Tests for the location model and graph."""

import numpy as np
import pytest

from app.models.locations import Location, LocationGraph, LocationType


def test_location_from_type_sets_environment():
    """from_type should set environment based on location type."""
    loc = Location.from_type("t1", "Tavern", LocationType.TAVERN)
    assert loc.location_id == "t1"
    assert loc.location_type == LocationType.TAVERN
    assert loc.environment.shape == (4,)
    # Tavern should have high safety
    assert loc.environment[0] > 0.5


def test_location_from_type_unknown_uses_default():
    """Unknown type should get 0.5 default environment."""
    loc = Location.from_type("u1", "Unknown", "mystery_zone")
    np.testing.assert_allclose(loc.environment, [0.5, 0.5, 0.5, 0.5])


def test_add_and_get_location():
    """Should be able to add and retrieve locations."""
    g = LocationGraph()
    g.add_location(Location.from_type("a", "Place A", "tavern"))
    loc = g.get_location("a")
    assert loc is not None
    assert loc.name == "Place A"


def test_duplicate_location_raises():
    """Adding a location with existing ID should raise."""
    g = LocationGraph()
    g.add_location(Location.from_type("a", "A", "tavern"))
    with pytest.raises(ValueError, match="already exists"):
        g.add_location(Location.from_type("a", "A2", "market"))


def test_add_edge_bidirectional():
    """Bidirectional edge should create edges in both directions."""
    g = LocationGraph()
    g.add_location(Location.from_type("a", "A", "tavern"))
    g.add_location(Location.from_type("b", "B", "market"))
    g.add_edge("a", "b", travel_hours=2.0, danger=0.3)

    neighbors_a = g.get_neighbors("a")
    neighbors_b = g.get_neighbors("b")
    assert len(neighbors_a) == 1
    assert neighbors_a[0].target_id == "b"
    assert neighbors_a[0].travel_hours == 2.0
    assert len(neighbors_b) == 1
    assert neighbors_b[0].target_id == "a"


def test_add_edge_unidirectional():
    """Unidirectional edge should only create one direction."""
    g = LocationGraph()
    g.add_location(Location.from_type("a", "A", "tavern"))
    g.add_location(Location.from_type("b", "B", "market"))
    g.add_edge("a", "b", travel_hours=1.0, bidirectional=False)

    assert len(g.get_neighbors("a")) == 1
    assert len(g.get_neighbors("b")) == 0


def test_edge_to_nonexistent_location_raises():
    """Adding edge to missing location should raise."""
    g = LocationGraph()
    g.add_location(Location.from_type("a", "A", "tavern"))
    with pytest.raises(ValueError, match="not found"):
        g.add_edge("a", "nonexistent")


def test_get_edge():
    """get_edge should return the specific edge or None."""
    g = LocationGraph()
    g.add_location(Location.from_type("a", "A", "tavern"))
    g.add_location(Location.from_type("b", "B", "market"))
    g.add_edge("a", "b", travel_hours=2.5)

    edge = g.get_edge("a", "b")
    assert edge is not None
    assert edge.travel_hours == 2.5

    assert g.get_edge("a", "nonexistent") is None


def test_remove_location_cleans_edges():
    """Removing a location should also remove edges pointing to it."""
    g = LocationGraph()
    g.add_location(Location.from_type("a", "A", "tavern"))
    g.add_location(Location.from_type("b", "B", "market"))
    g.add_location(Location.from_type("c", "C", "forest"))
    g.add_edge("a", "b", travel_hours=1.0)
    g.add_edge("b", "c", travel_hours=2.0)

    g.remove_location("b")
    assert g.get_location("b") is None
    # Edges from a to b should be gone
    assert len(g.get_neighbors("a")) == 0
    # Edges from c to b should be gone
    assert len(g.get_neighbors("c")) == 0


def test_serialization_roundtrip():
    """to_dict / from_dict should preserve the graph."""
    g = LocationGraph()
    g.add_location(Location.from_type("a", "Tavern", "tavern", capacity=10))
    g.add_location(Location.from_type("b", "Forest", "forest"))
    g.add_edge("a", "b", travel_hours=3.0, danger=0.5)

    data = g.to_dict()
    g2 = LocationGraph.from_dict(data)

    assert g2.location_count == 2
    loc_a = g2.get_location("a")
    assert loc_a is not None
    assert loc_a.name == "Tavern"
    assert loc_a.capacity == 10
    np.testing.assert_array_almost_equal(
        loc_a.environment, g.get_location("a").environment
    )

    edge = g2.get_edge("a", "b")
    assert edge is not None
    assert edge.travel_hours == 3.0
    assert edge.danger == 0.5
