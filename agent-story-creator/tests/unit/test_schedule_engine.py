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

"""Tests for the schedule engine (NPC daily routines)."""

from app.models.npc_status import NPCVectorialStatus
from app.simulation.schedule_engine import (
    LEISURE,
    RESTING,
    SLEEPING,
    WORKING,
    ScheduleEngine,
    get_schedule,
    resolve_activity,
)


def _make_npc(**kwargs) -> NPCVectorialStatus:
    return NPCVectorialStatus(**kwargs)


# --- Schedule templates ---


def test_default_schedule_covers_24h():
    """Default schedule should cover all 24 hours."""
    schedule = get_schedule("generic")
    for hour in [0, 3, 6, 9, 12, 15, 18, 21, 23.5]:
        activity = resolve_activity(schedule, hour)
        assert activity in (SLEEPING, RESTING, WORKING, LEISURE)


def test_unknown_archetype_gets_default():
    """Unknown archetypes should get the default schedule."""
    schedule = get_schedule("alien_species")
    default_schedule = get_schedule("generic")
    assert schedule == default_schedule


def test_guard_schedule_covers_24h():
    """Guard schedule should cover all 24 hours."""
    schedule = get_schedule("guard")
    for hour in range(24):
        activity = resolve_activity(schedule, hour)
        assert activity in (SLEEPING, RESTING, WORKING, LEISURE)


# --- Default schedule behavior ---


def test_default_sleeping_at_midnight():
    """Default NPCs should be sleeping at midnight."""
    schedule = get_schedule("generic")
    assert resolve_activity(schedule, 2.0) == SLEEPING


def test_default_working_at_morning():
    """Default NPCs should be working in the morning."""
    schedule = get_schedule("generic")
    assert resolve_activity(schedule, 9.0) == WORKING


def test_default_leisure_at_evening():
    """Default NPCs should be at leisure in the evening."""
    schedule = get_schedule("generic")
    assert resolve_activity(schedule, 19.0) == LEISURE


def test_default_resting_at_dawn():
    """Default NPCs should be resting at dawn."""
    schedule = get_schedule("generic")
    assert resolve_activity(schedule, 6.5) == RESTING


# --- Guard schedule (night shift) ---


def test_guard_working_at_night():
    """Guards should be working during night hours."""
    schedule = get_schedule("guard")
    assert resolve_activity(schedule, 2.0) == WORKING


def test_guard_sleeping_during_day():
    """Guards should sleep during daytime."""
    schedule = get_schedule("guard")
    assert resolve_activity(schedule, 10.0) == SLEEPING


# --- Merchant schedule ---


def test_merchant_working_long_hours():
    """Merchants should work long market hours."""
    schedule = get_schedule("merchant")
    assert resolve_activity(schedule, 8.0) == WORKING
    assert resolve_activity(schedule, 15.0) == WORKING


def test_merchant_sleeping_at_night():
    """Merchants should sleep at night."""
    schedule = get_schedule("merchant")
    assert resolve_activity(schedule, 3.0) == SLEEPING


# --- Criminal schedule (nocturnal) ---


def test_criminal_working_at_night():
    """Criminals should work at night."""
    schedule = get_schedule("criminal")
    assert resolve_activity(schedule, 1.0) == WORKING


def test_criminal_sleeping_during_day():
    """Criminals should sleep during daytime."""
    schedule = get_schedule("criminal")
    assert resolve_activity(schedule, 10.0) == SLEEPING


# --- Noble schedule (sleeps in) ---


def test_noble_sleeping_late():
    """Nobles should still be sleeping at 7am."""
    schedule = get_schedule("noble")
    assert resolve_activity(schedule, 7.0) == SLEEPING


def test_noble_leisure_evening():
    """Nobles should be at leisure in the evening."""
    schedule = get_schedule("noble")
    assert resolve_activity(schedule, 20.0) == LEISURE


# --- Priest schedule (early riser) ---


def test_priest_working_early():
    """Priests should be working at 6am (morning prayers)."""
    schedule = get_schedule("priest")
    assert resolve_activity(schedule, 6.5) == WORKING


# --- ScheduleEngine ---


def test_engine_assigns_activity():
    """Engine should assign activity based on game_time and archetype."""
    engine = ScheduleEngine()
    npc = _make_npc(name="worker", archetype="generic")
    activity = engine.compute_activity(npc, game_time=9.0)  # 9am
    assert activity == WORKING


def test_engine_wraps_time():
    """Game time > 24 should wrap correctly."""
    engine = ScheduleEngine()
    npc = _make_npc(name="wrap", archetype="generic")
    # game_time=50.0 -> hour 2am -> sleeping
    activity = engine.compute_activity(npc, game_time=50.0)
    assert activity == SLEEPING


def test_engine_wraps_large_time():
    """Very large game times should still wrap correctly."""
    engine = ScheduleEngine()
    npc = _make_npc(name="ancient", archetype="generic")
    # game_time=1000.0 -> 1000 % 24 = 16.0 -> working for default (13-18)
    activity = engine.compute_activity(npc, game_time=1000.0)
    assert activity == WORKING


def test_exhaustion_overrides_schedule():
    """Exhausted NPCs should be forced to sleep regardless of schedule."""
    engine = ScheduleEngine()
    npc = _make_npc(name="exhausted", archetype="generic", energy=0.02)
    # 9am would normally be working, but exhaustion forces sleep
    activity = engine.compute_activity(npc, game_time=9.0)
    assert activity == SLEEPING


def test_low_energy_not_exhausted():
    """NPCs above exhaustion threshold should follow schedule normally."""
    engine = ScheduleEngine()
    npc = _make_npc(name="tired", archetype="generic", energy=0.1)
    activity = engine.compute_activity(npc, game_time=9.0)
    assert activity == WORKING


def test_tick_assigns_all_npcs():
    """tick() should set activity for all NPCs."""
    engine = ScheduleEngine()
    npcs = [
        _make_npc(name="guard", archetype="guard"),
        _make_npc(name="merchant", archetype="merchant"),
        _make_npc(name="noble", archetype="noble"),
    ]
    engine.tick(npcs, game_time=3.0)  # 3am
    assert npcs[0].activity == WORKING  # Guard night shift
    assert npcs[1].activity == SLEEPING  # Merchant sleeping
    assert npcs[2].activity == SLEEPING  # Noble sleeping


def test_tick_midday():
    """At noon, most NPCs should be working."""
    engine = ScheduleEngine()
    npcs = [
        _make_npc(name="guard", archetype="guard"),
        _make_npc(name="merchant", archetype="merchant"),
        _make_npc(name="farmer", archetype="farmer"),
        _make_npc(name="criminal", archetype="criminal"),
    ]
    engine.tick(npcs, game_time=10.0)  # 10am
    assert npcs[0].activity == SLEEPING  # Guard sleeping (day)
    assert npcs[1].activity == WORKING  # Merchant working
    assert npcs[2].activity == WORKING  # Farmer working
    assert npcs[3].activity == SLEEPING  # Criminal sleeping (nocturnal)


def test_tick_evening():
    """In the evening, leisure archetypes should be free."""
    engine = ScheduleEngine()
    npcs = [
        _make_npc(name="generic", archetype="generic"),
        _make_npc(name="noble", archetype="noble"),
    ]
    engine.tick(npcs, game_time=19.0)  # 7pm
    assert npcs[0].activity == LEISURE  # Generic: evening leisure
    assert npcs[1].activity == LEISURE  # Noble: entertaining


def test_activity_field_updated_in_place():
    """tick() should mutate the NPC's activity field."""
    engine = ScheduleEngine()
    npc = _make_npc(name="mutable", archetype="generic", activity="idle")
    engine.tick([npc], game_time=3.0)
    assert npc.activity != "idle", "Activity should be updated from idle"
    assert npc.activity == SLEEPING


def test_different_archetypes_different_activities():
    """At the same time, different archetypes should have different activities."""
    engine = ScheduleEngine()
    guard = _make_npc(name="guard", archetype="guard")
    criminal = _make_npc(name="criminal", archetype="criminal")
    engine.tick([guard, criminal], game_time=2.0)  # 2am
    assert guard.activity == WORKING  # Night watch
    assert criminal.activity == WORKING  # Night activities


def test_soldier_uses_guard_schedule():
    """Soldiers should use the same schedule as guards."""
    engine = ScheduleEngine()
    soldier = _make_npc(name="soldier", archetype="soldier")
    guard = _make_npc(name="guard", archetype="guard")
    engine.tick([soldier, guard], game_time=10.0)
    assert soldier.activity == guard.activity


def test_bard_uses_artist_schedule():
    """Bards should use the same schedule as artists."""
    engine = ScheduleEngine()
    bard = _make_npc(name="bard", archetype="bard")
    artist = _make_npc(name="artist", archetype="artist")
    engine.tick([bard, artist], game_time=10.0)
    assert bard.activity == artist.activity
