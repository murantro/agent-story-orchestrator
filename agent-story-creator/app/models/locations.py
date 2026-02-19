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

"""Location model and world graph.

Locations are nodes in a graph with weighted edges (travel time in
game-hours). Each location has environmental properties that feed
into NPC environment vectors and a type that influences NPC behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.models.npc_status import ENVIRONMENT_DIM


class LocationType:
    """Common location type constants."""

    TAVERN = "tavern"
    MARKET = "market"
    RESIDENTIAL = "residential"
    TEMPLE = "temple"
    BARRACKS = "barracks"
    FOREST = "forest"
    ROAD = "road"
    PLAZA = "plaza"
    DOCKS = "docks"
    DUNGEON = "dungeon"


# Default environment vectors per location type.
# Index: safety, resource_abundance, weather_comfort, crowding
_DEFAULT_ENVIRONMENT: dict[str, np.ndarray] = {
    LocationType.TAVERN: np.array([0.7, 0.6, 0.9, 0.6], dtype=np.float32),
    LocationType.MARKET: np.array([0.6, 0.9, 0.7, 0.8], dtype=np.float32),
    LocationType.RESIDENTIAL: np.array([0.8, 0.5, 0.8, 0.4], dtype=np.float32),
    LocationType.TEMPLE: np.array([0.9, 0.3, 0.8, 0.3], dtype=np.float32),
    LocationType.BARRACKS: np.array([0.9, 0.4, 0.6, 0.5], dtype=np.float32),
    LocationType.FOREST: np.array([0.3, 0.7, 0.5, 0.1], dtype=np.float32),
    LocationType.ROAD: np.array([0.4, 0.2, 0.5, 0.2], dtype=np.float32),
    LocationType.PLAZA: np.array([0.6, 0.4, 0.7, 0.7], dtype=np.float32),
    LocationType.DOCKS: np.array([0.5, 0.8, 0.4, 0.5], dtype=np.float32),
    LocationType.DUNGEON: np.array([0.1, 0.3, 0.2, 0.1], dtype=np.float32),
}


@dataclass
class Location:
    """A location node in the world graph.

    Attributes:
        location_id: Unique identifier (matches NPC.location_id).
        name: Human-readable name.
        location_type: Category (tavern, market, forest, etc.).
        environment: 4-dim environment vector (safety, resource_abundance,
            weather_comfort, crowding). Feeds into NPC environment vectors.
        capacity: Maximum NPCs the location can hold (0 = unlimited).
    """

    location_id: str = ""
    name: str = ""
    location_type: str = "generic"
    environment: np.ndarray = field(
        default_factory=lambda: np.zeros(ENVIRONMENT_DIM, dtype=np.float32)
    )
    capacity: int = 0

    @staticmethod
    def from_type(
        location_id: str,
        name: str,
        location_type: str,
        capacity: int = 0,
    ) -> Location:
        """Create a location with default environment for its type.

        Args:
            location_id: Unique identifier.
            name: Display name.
            location_type: One of LocationType constants.
            capacity: Max NPCs (0 = unlimited).

        Returns:
            A new Location with environment pre-filled from type defaults.
        """
        env = _DEFAULT_ENVIRONMENT.get(
            location_type,
            np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
        ).copy()
        return Location(
            location_id=location_id,
            name=name,
            location_type=location_type,
            environment=env,
            capacity=capacity,
        )


@dataclass
class LocationEdge:
    """A directed edge in the location graph.

    Attributes:
        target_id: Destination location ID.
        travel_hours: Travel time in game-hours.
        danger: How dangerous the path is (0-1). Affects NPC willingness.
    """

    target_id: str
    travel_hours: float = 1.0
    danger: float = 0.0


class LocationGraph:
    """Directed weighted graph of world locations.

    Locations are nodes; edges represent paths with travel times.
    Used by MovementEngine to decide NPC movement.
    """

    def __init__(self) -> None:
        self._locations: dict[str, Location] = {}
        self._edges: dict[str, list[LocationEdge]] = {}

    @property
    def location_count(self) -> int:
        return len(self._locations)

    def add_location(self, location: Location) -> None:
        """Add a location to the graph.

        Args:
            location: The location to add.

        Raises:
            ValueError: If a location with the same ID already exists.
        """
        if location.location_id in self._locations:
            raise ValueError(f"Location {location.location_id!r} already exists.")
        self._locations[location.location_id] = location
        self._edges.setdefault(location.location_id, [])

    def get_location(self, location_id: str) -> Location | None:
        """Get a location by ID, or None if not found."""
        return self._locations.get(location_id)

    def list_locations(self) -> list[Location]:
        """Return all locations."""
        return list(self._locations.values())

    def remove_location(self, location_id: str) -> bool:
        """Remove a location and all edges to/from it.

        Args:
            location_id: ID of the location to remove.

        Returns:
            True if removed, False if not found.
        """
        if location_id not in self._locations:
            return False
        del self._locations[location_id]
        del self._edges[location_id]
        # Remove edges pointing to this location
        for edges in self._edges.values():
            edges[:] = [e for e in edges if e.target_id != location_id]
        return True

    def add_edge(
        self,
        from_id: str,
        to_id: str,
        travel_hours: float = 1.0,
        danger: float = 0.0,
        bidirectional: bool = True,
    ) -> None:
        """Add an edge (path) between two locations.

        Args:
            from_id: Source location ID.
            to_id: Destination location ID.
            travel_hours: Travel time in game-hours.
            danger: Path danger level (0-1).
            bidirectional: If True, also add the reverse edge.

        Raises:
            ValueError: If either location does not exist.
        """
        if from_id not in self._locations:
            raise ValueError(f"Location {from_id!r} not found.")
        if to_id not in self._locations:
            raise ValueError(f"Location {to_id!r} not found.")
        self._edges[from_id].append(
            LocationEdge(target_id=to_id, travel_hours=travel_hours, danger=danger)
        )
        if bidirectional:
            self._edges[to_id].append(
                LocationEdge(
                    target_id=from_id, travel_hours=travel_hours, danger=danger
                )
            )

    def get_neighbors(self, location_id: str) -> list[LocationEdge]:
        """Get all edges from a location.

        Args:
            location_id: The source location.

        Returns:
            List of edges, or empty list if location not found.
        """
        return list(self._edges.get(location_id, []))

    def get_edge(self, from_id: str, to_id: str) -> LocationEdge | None:
        """Get a specific edge between two locations.

        Args:
            from_id: Source location ID.
            to_id: Destination location ID.

        Returns:
            The edge, or None if no direct path exists.
        """
        for edge in self._edges.get(from_id, []):
            if edge.target_id == to_id:
                return edge
        return None

    def to_dict(self) -> dict:
        """Serialize the graph for JSON snapshots.

        Returns:
            Dict with locations and edges.
        """
        return {
            "locations": {
                loc_id: {
                    "name": loc.name,
                    "location_type": loc.location_type,
                    "environment": loc.environment.tolist(),
                    "capacity": loc.capacity,
                }
                for loc_id, loc in self._locations.items()
            },
            "edges": {
                from_id: [
                    {
                        "target_id": e.target_id,
                        "travel_hours": e.travel_hours,
                        "danger": e.danger,
                    }
                    for e in edges
                ]
                for from_id, edges in self._edges.items()
            },
        }

    @staticmethod
    def from_dict(data: dict) -> LocationGraph:
        """Deserialize a graph from a JSON snapshot.

        Args:
            data: Dict previously produced by to_dict().

        Returns:
            A restored LocationGraph.
        """
        graph = LocationGraph()
        for loc_id, loc_data in data.get("locations", {}).items():
            loc = Location(
                location_id=loc_id,
                name=loc_data["name"],
                location_type=loc_data["location_type"],
                environment=np.array(loc_data["environment"], dtype=np.float32),
                capacity=loc_data.get("capacity", 0),
            )
            graph._locations[loc_id] = loc
            graph._edges.setdefault(loc_id, [])
        for from_id, edges_data in data.get("edges", {}).items():
            for e_data in edges_data:
                graph._edges.setdefault(from_id, []).append(
                    LocationEdge(
                        target_id=e_data["target_id"],
                        travel_hours=e_data["travel_hours"],
                        danger=e_data.get("danger", 0.0),
                    )
                )
        return graph
