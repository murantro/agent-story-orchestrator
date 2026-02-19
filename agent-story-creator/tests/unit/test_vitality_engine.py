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

"""Tests for the vitality engine (energy and health dynamics)."""

import numpy as np

from app.models.events import WorldEvent
from app.models.npc_status import NPCVectorialStatus
from app.simulation.vitality_engine import VitalityEngine


def _make_npc(**kwargs) -> NPCVectorialStatus:
    return NPCVectorialStatus(**kwargs)


def _safe_env() -> np.ndarray:
    """Environment: safe, resource-rich, comfortable, uncrowded."""
    return np.array([0.9, 0.8, 0.9, 0.2], dtype=np.float32)


def _dangerous_env() -> np.ndarray:
    """Environment: unsafe, scarce, harsh, empty."""
    return np.array([0.1, 0.2, 0.1, 0.0], dtype=np.float32)


# --- Energy drain/regen ---


def test_passive_energy_drain():
    """Energy should decrease each tick even in a neutral environment."""
    engine = VitalityEngine(energy_regen_base=0.0)  # No regen
    npc = _make_npc(name="drainer", energy=0.5, environment=_safe_env())
    engine.update_npc(npc)
    assert npc.energy < 0.5, "Energy should drain passively"


def test_energy_regen_in_safe_environment():
    """Energy should recover in safe, comfortable environments."""
    engine = VitalityEngine(energy_drain=0.0)  # No drain
    npc = _make_npc(name="rester", energy=0.5, environment=_safe_env())
    engine.update_npc(npc)
    assert npc.energy > 0.5, "Energy should regenerate in safe environments"


def test_net_energy_gain_in_safe_location():
    """In a safe location, regen should exceed drain (net positive)."""
    engine = VitalityEngine()
    npc = _make_npc(name="safe_npc", energy=0.5, environment=_safe_env())
    engine.update_npc(npc)
    assert npc.energy > 0.5, "Net energy should be positive in safe location"


def test_net_energy_loss_in_dangerous_location():
    """In a dangerous location, drain exceeds regen (net negative)."""
    engine = VitalityEngine()
    npc = _make_npc(name="danger_npc", energy=0.5, environment=_dangerous_env())
    engine.update_npc(npc)
    assert npc.energy < 0.5, "Net energy should be negative in dangerous location"


def test_energy_clamped_to_0_1():
    """Energy should never go below 0 or above 1."""
    engine = VitalityEngine(energy_drain=0.5, energy_regen_base=0.0)
    npc = _make_npc(name="low", energy=0.01, environment=_dangerous_env())
    engine.update_npc(npc)
    assert npc.energy >= 0.0, "Energy should not go below 0"

    engine2 = VitalityEngine(energy_drain=0.0, energy_regen_base=5.0)
    npc2 = _make_npc(name="high", energy=0.99, environment=_safe_env())
    engine2.update_npc(npc2)
    assert npc2.energy <= 1.0, "Energy should not exceed 1"


# --- Health dynamics ---


def test_health_damage_in_unsafe_environment():
    """Health should decrease in environments below safety threshold."""
    engine = VitalityEngine()
    npc = _make_npc(name="wounded", health=0.8, environment=_dangerous_env())
    engine.update_npc(npc)
    assert npc.health < 0.8, "Health should decrease in dangerous environments"


def test_health_regen_in_safe_environment():
    """Health should recover (slowly) in safe environments."""
    engine = VitalityEngine(danger_health_drain=0.0)  # No damage
    npc = _make_npc(name="healer", health=0.5, environment=_safe_env())
    engine.update_npc(npc)
    assert npc.health > 0.5, "Health should regenerate in safe environments"


def test_no_health_damage_above_threshold():
    """No health damage if safety is above the danger threshold."""
    engine = VitalityEngine()
    npc = _make_npc(name="safe", health=0.8, environment=_safe_env())
    old_health = npc.health
    engine.update_npc(npc)
    # Health should only go up (regen) since safety > threshold
    assert npc.health >= old_health, "No damage in safe location"


def test_health_clamped_to_0_1():
    """Health should never go below 0 or above 1."""
    engine = VitalityEngine(danger_health_drain=5.0)
    npc = _make_npc(name="dying", health=0.01, environment=_dangerous_env())
    engine.update_npc(npc)
    assert npc.health >= 0.0, "Health should not go below 0"

    engine2 = VitalityEngine(health_regen_rate=5.0, danger_health_drain=0.0)
    npc2 = _make_npc(name="overhealed", health=0.99, environment=_safe_env())
    engine2.update_npc(npc2)
    assert npc2.health <= 1.0, "Health should not exceed 1"


def test_full_health_no_regen():
    """At health=1.0, no regeneration should be applied."""
    engine = VitalityEngine(danger_health_drain=0.0)
    npc = _make_npc(name="full_hp", health=1.0, environment=_safe_env())
    change = engine.compute_health_change(npc)
    assert change == 0.0, "No regen at full health"


# --- Health-energy cap ---


def test_health_caps_energy():
    """Low health should cap available energy."""
    engine = VitalityEngine(
        energy_drain=0.0,
        energy_regen_base=0.0,
        danger_health_drain=0.0,
        health_regen_rate=0.0,
    )
    npc = _make_npc(
        name="injured",
        energy=0.9,
        health=0.25,
        environment=_safe_env(),
    )
    engine.update_npc(npc)
    # health=0.25, cap_threshold=0.5 -> cap = 0.25/0.5 = 0.5
    assert npc.energy <= 0.5 + 1e-6, "Energy should be capped by low health"


def test_health_above_threshold_no_cap():
    """Health above threshold should not cap energy."""
    engine = VitalityEngine(
        energy_drain=0.0,
        energy_regen_base=0.0,
        danger_health_drain=0.0,
    )
    npc = _make_npc(
        name="healthy",
        energy=0.9,
        health=0.8,
        environment=_safe_env(),
    )
    engine.update_npc(npc)
    assert npc.energy == 0.9, (
        "Energy should not be capped when health is above threshold"
    )


# --- Event-driven health/energy ---


def test_damaging_event_reduces_health():
    """Attack events should reduce NPC health."""
    engine = VitalityEngine()
    npc = _make_npc(name="victim", health=0.8, energy=0.8)
    event = WorldEvent(event_type="attack_bandit", intensity=1.0)
    engine.apply_event(npc, event)
    assert npc.health < 0.8, "Attack should reduce health"
    assert npc.energy < 0.8, "Attack should reduce energy"


def test_healing_event_restores_health():
    """Healing events should restore NPC health."""
    engine = VitalityEngine()
    npc = _make_npc(name="patient", health=0.5, energy=0.5)
    event = WorldEvent(event_type="healing_potion", intensity=1.0)
    engine.apply_event(npc, event)
    assert npc.health > 0.5, "Healing should restore health"
    assert npc.energy > 0.5, "Healing should restore energy"


def test_event_intensity_scales_impact():
    """Lower intensity events should have smaller health impact."""
    engine = VitalityEngine()
    npc_full = _make_npc(name="full_hit", health=1.0, energy=1.0)
    npc_half = _make_npc(name="half_hit", health=1.0, energy=1.0)

    engine.apply_event(npc_full, WorldEvent(event_type="attack_x", intensity=1.0))
    engine.apply_event(npc_half, WorldEvent(event_type="attack_x", intensity=0.5))

    assert npc_half.health > npc_full.health, "Half intensity should deal less damage"


def test_unknown_event_type_no_impact():
    """Events with unknown types should not affect health/energy."""
    engine = VitalityEngine()
    npc = _make_npc(name="bystander", health=0.8, energy=0.8)
    event = WorldEvent(event_type="trade_deal", intensity=1.0)
    engine.apply_event(npc, event)
    assert npc.health == 0.8, "Unknown event should not affect health"
    assert npc.energy == 0.8, "Unknown event should not affect energy"


def test_apply_event_batch():
    """Batch application should affect all NPCs."""
    engine = VitalityEngine()
    npcs = [
        _make_npc(name="a", health=0.8, energy=0.8),
        _make_npc(name="b", health=0.9, energy=0.7),
    ]
    old_healths = [npc.health for npc in npcs]
    event = WorldEvent(event_type="disaster_earthquake", intensity=0.8)
    engine.apply_event_batch(npcs, event)
    for i, npc in enumerate(npcs):
        assert npc.health < old_healths[i], f"{npc.name} should take damage"


def test_event_clamps_values():
    """Events should not push health/energy outside [0, 1]."""
    engine = VitalityEngine()
    npc = _make_npc(name="overkill", health=0.05, energy=0.05)
    event = WorldEvent(event_type="battle_siege", intensity=1.0)
    engine.apply_event(npc, event)
    assert npc.health >= 0.0
    assert npc.energy >= 0.0


# --- Interaction costs ---


def test_interaction_costs_deduct_energy():
    """Interactions should cost energy for both participants."""
    engine = VitalityEngine()
    npc_a = _make_npc(name="a", energy=0.8)
    npc_b = _make_npc(name="b", energy=0.7)
    engine.apply_interaction_costs(
        npc_a, npc_b, energy_cost=0.03, health_delta_a=0.0, health_delta_b=0.0
    )
    assert abs(npc_a.energy - 0.77) < 1e-6
    assert abs(npc_b.energy - 0.67) < 1e-6


def test_conflict_interaction_damages_health():
    """Conflict interactions should damage both participants' health."""
    engine = VitalityEngine()
    npc_a = _make_npc(name="fighter_a", health=0.8, energy=0.8)
    npc_b = _make_npc(name="fighter_b", health=0.9, energy=0.8)
    engine.apply_interaction_costs(
        npc_a,
        npc_b,
        energy_cost=0.03,
        health_delta_a=-0.08,
        health_delta_b=-0.08,
    )
    assert abs(npc_a.health - 0.72) < 1e-6
    assert abs(npc_b.health - 0.82) < 1e-6


def test_aid_interaction_heals_recipient():
    """Aid interactions should heal the recipient."""
    engine = VitalityEngine()
    npc_a = _make_npc(name="helper", health=0.8, energy=0.8)
    npc_b = _make_npc(name="helped", health=0.5, energy=0.8)
    engine.apply_interaction_costs(
        npc_a,
        npc_b,
        energy_cost=0.01,
        health_delta_a=0.0,
        health_delta_b=0.05,
    )
    assert abs(npc_b.health - 0.55) < 1e-6
    assert npc_a.health == 0.8  # Helper not affected


def test_interaction_costs_clamp():
    """Interaction costs should not push values below 0."""
    engine = VitalityEngine()
    npc_a = _make_npc(name="low_a", energy=0.01, health=0.02)
    npc_b = _make_npc(name="low_b", energy=0.01, health=0.02)
    engine.apply_interaction_costs(
        npc_a,
        npc_b,
        energy_cost=0.5,
        health_delta_a=-0.5,
        health_delta_b=-0.5,
    )
    assert npc_a.energy >= 0.0
    assert npc_b.energy >= 0.0
    assert npc_a.health >= 0.0
    assert npc_b.health >= 0.0


# --- Tick (batch) ---


def test_tick_mutates_all_npcs():
    """tick() should update energy and health for all NPCs."""
    engine = VitalityEngine()
    npcs = [
        _make_npc(name="a", energy=0.8, health=0.8, environment=_safe_env()),
        _make_npc(name="b", energy=0.6, health=0.6, environment=_dangerous_env()),
    ]
    old_energies = [npc.energy for npc in npcs]
    engine.tick(npcs)
    for i, npc in enumerate(npcs):
        assert npc.energy != old_energies[i], f"NPC {npc.name} energy should change"


def test_multiple_ticks_converge():
    """Over many ticks in a safe env, energy should converge near 1.0."""
    engine = VitalityEngine()
    npc = _make_npc(name="recoverer", energy=0.2, health=1.0, environment=_safe_env())
    for _ in range(200):
        engine.update_npc(npc)
    assert npc.energy > 0.8, "Energy should converge toward 1.0 in safe environment"


def test_dangerous_env_drains_to_zero():
    """Over many ticks in danger, health should approach 0."""
    engine = VitalityEngine()
    env = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    npc = _make_npc(name="doomed", energy=1.0, health=1.0, environment=env)
    for _ in range(500):
        engine.update_npc(npc)
    assert npc.health < 0.1, "Health should approach 0 in zero-safety environment"
