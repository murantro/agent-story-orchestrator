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

"""Vitality engine - energy and health dynamics per tick.

Manages the physical state of NPCs: energy drain, regeneration,
health damage from unsafe environments, and passive healing.
Actions have costs, and the environment determines recovery rates.

Design:
  - Energy drains passively each tick (NPCs expend effort just by existing).
  - Energy regenerates based on environmental safety and comfort.
  - Health is damaged by unsafe environments (low safety).
  - Health regenerates slowly, faster in safe locations.
  - Low health caps effective energy (injured NPCs tire faster).

All math is pure numpy - no LLM calls.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.config import (
    DANGER_HEALTH_DRAIN,
    DANGER_SAFETY_THRESHOLD,
    ENERGY_DRAIN_PER_TICK,
    ENERGY_REGEN_BASE,
    HEALTH_ENERGY_CAP_THRESHOLD,
    HEALTH_REGEN_RATE,
)
from app.models.events import WorldEvent
from app.models.npc_status import NPCVectorialStatus

# Event types that deal health damage to affected NPCs.
# Maps event_type prefix to (health_damage, energy_drain) applied per NPC.
_DAMAGING_EVENT_TYPES: dict[str, tuple[float, float]] = {
    "attack": (-0.15, -0.05),
    "battle": (-0.2, -0.1),
    "disaster": (-0.1, -0.05),
    "plague": (-0.12, -0.03),
    "fire": (-0.1, -0.04),
    "collapse": (-0.08, -0.02),
}

# Event types that restore health/energy.
_HEALING_EVENT_TYPES: dict[str, tuple[float, float]] = {
    "healing": (0.15, 0.05),
    "feast": (0.05, 0.15),
    "rest": (0.0, 0.2),
    "celebration": (0.02, 0.1),
}

# Environment vector indices (matching ENVIRONMENT_LABELS order).
_ENV_SAFETY = 0
_ENV_WEATHER_COMFORT = 2


@dataclass
class VitalityEngine:
    """Manages NPC energy and health dynamics per tick.

    Args:
        energy_drain: Passive energy cost per tick.
        energy_regen_base: Base energy recovery per tick (scaled by environment).
        health_regen_rate: Passive health recovery per tick (scaled by safety).
        danger_health_drain: Health drain per tick in unsafe environments.
        danger_safety_threshold: Safety below this causes health damage.
        health_energy_cap_threshold: Below this health, energy is capped.
    """

    energy_drain: float = ENERGY_DRAIN_PER_TICK
    energy_regen_base: float = ENERGY_REGEN_BASE
    health_regen_rate: float = HEALTH_REGEN_RATE
    danger_health_drain: float = DANGER_HEALTH_DRAIN
    danger_safety_threshold: float = DANGER_SAFETY_THRESHOLD
    health_energy_cap_threshold: float = HEALTH_ENERGY_CAP_THRESHOLD

    def compute_energy_regen(self, npc: NPCVectorialStatus) -> float:
        """Compute energy regeneration for an NPC based on environment.

        Safe, comfortable environments restore more energy.

        Args:
            npc: The NPC (not mutated).

        Returns:
            Energy regeneration amount (non-negative).
        """
        safety = float(npc.environment[_ENV_SAFETY])
        comfort = float(npc.environment[_ENV_WEATHER_COMFORT])
        env_factor = 0.5 * safety + 0.5 * comfort
        return self.energy_regen_base * env_factor

    def compute_health_change(self, npc: NPCVectorialStatus) -> float:
        """Compute net health change for an NPC based on environment.

        Unsafe environments damage health; safe environments allow healing.

        Args:
            npc: The NPC (not mutated).

        Returns:
            Net health change (can be negative for damage, positive for regen).
        """
        safety = float(npc.environment[_ENV_SAFETY])
        change = 0.0

        # Damage from unsafe environments
        if safety < self.danger_safety_threshold:
            damage = self.danger_health_drain * (self.danger_safety_threshold - safety)
            change -= damage

        # Passive healing (scaled by safety)
        if npc.health < 1.0:
            change += self.health_regen_rate * safety

        return change

    def apply_health_energy_cap(self, npc: NPCVectorialStatus) -> None:
        """Cap energy based on health level.

        Injured NPCs cannot sustain high energy levels - their body
        diverts resources to healing.

        Args:
            npc: The NPC (mutated in-place).
        """
        if npc.health < self.health_energy_cap_threshold:
            # At health=0.5, cap=1.0. At health=0.25, cap=0.5. At health=0, cap=0.
            energy_cap = npc.health / self.health_energy_cap_threshold
            npc.energy = min(npc.energy, energy_cap)

    def update_npc(self, npc: NPCVectorialStatus) -> None:
        """Apply one tick of vitality dynamics to a single NPC.

        Order: drain energy -> regen energy -> health change -> cap energy.

        Args:
            npc: The NPC (mutated in-place).
        """
        # 1. Passive energy drain
        npc.energy -= self.energy_drain

        # 2. Energy regeneration from environment
        regen = self.compute_energy_regen(npc)
        npc.energy += regen

        # 3. Health changes (damage from danger + passive healing)
        health_delta = self.compute_health_change(npc)
        npc.health += health_delta

        # 4. Clamp both to [0, 1]
        npc.energy = float(np.clip(npc.energy, 0.0, 1.0))
        npc.health = float(np.clip(npc.health, 0.0, 1.0))

        # 5. Health caps energy (after clamping health)
        self.apply_health_energy_cap(npc)

    def apply_event(self, npc: NPCVectorialStatus, event: WorldEvent) -> None:
        """Apply health/energy impact from a world event to an NPC.

        Checks event_type against known damaging and healing event types.
        Impact is scaled by event intensity.

        Args:
            npc: The NPC (mutated in-place).
            event: The world event to apply.
        """
        for prefix, (health_delta, energy_delta) in _DAMAGING_EVENT_TYPES.items():
            if event.event_type.startswith(prefix):
                npc.health += health_delta * event.intensity
                npc.energy += energy_delta * event.intensity
                npc.health = float(np.clip(npc.health, 0.0, 1.0))
                npc.energy = float(np.clip(npc.energy, 0.0, 1.0))
                return

        for prefix, (health_delta, energy_delta) in _HEALING_EVENT_TYPES.items():
            if event.event_type.startswith(prefix):
                npc.health += health_delta * event.intensity
                npc.energy += energy_delta * event.intensity
                npc.health = float(np.clip(npc.health, 0.0, 1.0))
                npc.energy = float(np.clip(npc.energy, 0.0, 1.0))
                return

    def apply_event_batch(
        self, npcs: list[NPCVectorialStatus], event: WorldEvent
    ) -> None:
        """Apply an event's health/energy impact in-place to multiple NPCs.

        Args:
            npcs: NPCs to mutate.
            event: The event to apply.
        """
        for npc in npcs:
            self.apply_event(npc, event)

    def apply_interaction_costs(
        self,
        npc_a: NPCVectorialStatus,
        npc_b: NPCVectorialStatus,
        energy_cost: float,
        health_delta_a: float,
        health_delta_b: float,
    ) -> None:
        """Apply energy and health costs from an interaction.

        Args:
            npc_a: First participant (mutated in-place).
            npc_b: Second participant (mutated in-place).
            energy_cost: Energy deducted from both.
            health_delta_a: Health change for NPC A.
            health_delta_b: Health change for NPC B.
        """
        npc_a.energy = float(np.clip(npc_a.energy - energy_cost, 0.0, 1.0))
        npc_b.energy = float(np.clip(npc_b.energy - energy_cost, 0.0, 1.0))
        npc_a.health = float(np.clip(npc_a.health + health_delta_a, 0.0, 1.0))
        npc_b.health = float(np.clip(npc_b.health + health_delta_b, 0.0, 1.0))

    def tick(self, npcs: list[NPCVectorialStatus]) -> None:
        """Apply vitality dynamics in-place for all NPCs.

        Args:
            npcs: NPCs to mutate.
        """
        for npc in npcs:
            self.update_npc(npc)
