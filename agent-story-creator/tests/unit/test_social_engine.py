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

"""Tests for the social influence engine (peer pressure and contagion)."""

import numpy as np

from app.models.events import WorldEvent
from app.models.npc_status import (
    PERSONALITY_DIM,
    SOCIAL_INFLUENCE_DIM,
    NPCVectorialStatus,
)
from app.simulation.social_engine import (
    SocialInfluenceEngine,
    get_archetype_profile,
)


def _make_npc(**kwargs) -> NPCVectorialStatus:
    return NPCVectorialStatus(**kwargs)


# --- Archetype profiles ---


def test_known_archetype_returns_profile():
    """Known archetypes should return a non-zero radiation profile."""
    profile = get_archetype_profile("merchant")
    assert profile.shape == (SOCIAL_INFLUENCE_DIM,)
    assert np.any(profile > 0), "Merchant profile should have non-zero values"


def test_unknown_archetype_returns_zeros():
    """Unknown archetypes should return a zero profile."""
    profile = get_archetype_profile("unknown_type")
    assert np.all(profile == 0.0)


def test_merchant_radiates_economic_pressure():
    """Merchants should radiate economic_pressure (index 1)."""
    profile = get_archetype_profile("merchant")
    assert profile[1] > 0.0, "Merchants should radiate economic_pressure"
    assert profile[1] >= profile[4], "Economic > religious for merchants"


def test_priest_radiates_religious_devotion():
    """Priests should radiate religious_devotion (index 4)."""
    profile = get_archetype_profile("priest")
    assert profile[4] > 0.0, "Priests should radiate religious_devotion"
    assert profile[4] >= profile[1], "Religious > economic for priests"


def test_noble_radiates_status_and_political():
    """Nobles should radiate status_seeking and political_alignment."""
    profile = get_archetype_profile("noble")
    assert profile[3] > 0.0, "Nobles should radiate status_seeking"
    assert profile[5] > 0.0, "Nobles should radiate political_alignment"


# --- Susceptibility ---


def test_high_agreeableness_more_susceptible():
    """High agreeableness should increase susceptibility."""
    engine = SocialInfluenceEngine()
    agreeable = _make_npc(
        name="agreeable",
        personality=np.array([0.5, 0.5, 0.5, 0.9, 0.2], dtype=np.float32),
    )
    stubborn = _make_npc(
        name="stubborn",
        personality=np.array([0.5, 0.5, 0.5, 0.1, 0.2], dtype=np.float32),
    )
    assert engine.compute_susceptibility(agreeable) > engine.compute_susceptibility(
        stubborn
    ), "High agreeableness should mean higher susceptibility"


def test_susceptibility_in_range():
    """Susceptibility should always be in [0.2, 1.0]."""
    engine = SocialInfluenceEngine()
    for _ in range(20):
        personality = np.random.rand(PERSONALITY_DIM).astype(np.float32)
        npc = _make_npc(personality=personality)
        s = engine.compute_susceptibility(npc)
        assert 0.2 <= s <= 1.0, f"Susceptibility {s} out of range"


# --- Peer signal computation ---


def test_peer_signal_from_archetype():
    """Co-located archetype NPCs should produce a non-zero peer signal."""
    engine = SocialInfluenceEngine()
    receiver = _make_npc(name="receiver", location_id="tavern")
    merchant = _make_npc(name="merchant", archetype="merchant", location_id="tavern")
    signal = engine.compute_peer_signal(receiver, [receiver, merchant])
    assert signal[1] > 0.0, "Merchant nearby should push economic_pressure"


def test_peer_signal_excludes_self():
    """NPC should not be influenced by their own social vector."""
    engine = SocialInfluenceEngine()
    npc = _make_npc(
        name="alone",
        location_id="cave",
        social_influence=np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32),
    )
    signal = engine.compute_peer_signal(npc, [npc])
    assert np.all(signal == 0.0), "Self should not influence self"


def test_peer_signal_empty_location():
    """Empty location should produce zero signal."""
    engine = SocialInfluenceEngine()
    npc = _make_npc(name="alone")
    signal = engine.compute_peer_signal(npc, [])
    assert np.all(signal == 0.0)


def test_friends_influence_more_than_strangers():
    """NPCs with positive relationships should have stronger influence."""
    engine = SocialInfluenceEngine()
    receiver = _make_npc(name="receiver", location_id="market")
    friend = _make_npc(
        name="friend",
        archetype="priest",
        location_id="market",
    )
    stranger = _make_npc(
        name="stranger",
        archetype="priest",
        location_id="market",
    )
    # Set relationship: friend is known, stranger is not
    receiver.relationships[friend.npc_id] = 0.8

    signal_with_friend = engine.compute_peer_signal(receiver, [receiver, friend])
    signal_with_stranger = engine.compute_peer_signal(receiver, [receiver, stranger])

    # Friend signal should be stronger (higher weight)
    assert np.linalg.norm(signal_with_friend) > np.linalg.norm(signal_with_stranger), (
        "Friends should exert stronger social influence"
    )


def test_enemy_reduces_influence():
    """Negative relationships should reduce or reverse influence."""
    engine = SocialInfluenceEngine()
    receiver = _make_npc(name="receiver", location_id="square")
    enemy = _make_npc(
        name="enemy",
        archetype="priest",
        location_id="square",
    )
    receiver.relationships[enemy.npc_id] = -0.8

    signal = engine.compute_peer_signal(receiver, [receiver, enemy])
    # Enemy has weight 0.5 + (-0.8)*0.5 = 0.1 (very low)
    neutral_signal = engine.compute_peer_signal(
        _make_npc(name="neutral", location_id="square"),
        [_make_npc(name="neutral", location_id="square"), enemy],
    )
    assert np.linalg.norm(signal) < np.linalg.norm(neutral_signal), (
        "Enemy influence should be weaker than neutral"
    )


def test_social_contagion():
    """NPC social vectors should spread to peers (contagion effect)."""
    engine = SocialInfluenceEngine()
    receiver = _make_npc(name="receiver", location_id="plaza")
    influencer = _make_npc(
        name="influencer",
        location_id="plaza",
        archetype="generic",
        social_influence=np.array([0.0, 0.8, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    signal = engine.compute_peer_signal(receiver, [receiver, influencer])
    assert signal[1] > 0.0, (
        "Peer's high economic_pressure should propagate via contagion"
    )


# --- Event-driven social shifts ---


def test_event_social_impact():
    """Events with social_impact should shift NPC social vectors."""
    engine = SocialInfluenceEngine()
    npc = _make_npc(
        name="citizen",
        social_influence=np.zeros(SOCIAL_INFLUENCE_DIM, dtype=np.float32),
    )
    event = WorldEvent(
        event_type="political_rally",
        intensity=0.8,
        social_impact=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.3], dtype=np.float32),
    )
    engine.apply_event(npc, event)
    assert npc.social_influence[5] > 0.0, (
        "Political event should increase political_alignment"
    )


def test_event_zero_social_impact_no_change():
    """Events with zero social_impact should not change the vector."""
    engine = SocialInfluenceEngine()
    npc = _make_npc(
        name="bystander",
        social_influence=np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3], dtype=np.float32),
    )
    old = npc.social_influence.copy()
    event = WorldEvent(
        event_type="trade",
        intensity=1.0,
        social_impact=np.zeros(SOCIAL_INFLUENCE_DIM, dtype=np.float32),
    )
    engine.apply_event(npc, event)
    np.testing.assert_array_equal(npc.social_influence, old)


def test_event_intensity_scales_social_impact():
    """Lower intensity should produce smaller social shifts."""
    engine = SocialInfluenceEngine()
    impact = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.0], dtype=np.float32)

    npc_full = _make_npc(name="full")
    npc_half = _make_npc(name="half")

    engine.apply_event(
        npc_full, WorldEvent(event_type="a", intensity=1.0, social_impact=impact)
    )
    engine.apply_event(
        npc_half, WorldEvent(event_type="a", intensity=0.5, social_impact=impact)
    )

    assert npc_full.social_influence[4] > npc_half.social_influence[4]


def test_event_batch_applies_to_all():
    """Batch event application should affect all NPCs."""
    engine = SocialInfluenceEngine()
    npcs = [_make_npc(name="a"), _make_npc(name="b")]
    event = WorldEvent(
        event_type="sermon",
        intensity=1.0,
        social_impact=np.array([0.0, 0.0, 0.0, 0.0, 0.4, 0.0], dtype=np.float32),
    )
    engine.apply_event_batch(npcs, event)
    for npc in npcs:
        assert npc.social_influence[4] > 0.0


def test_social_influence_clamped_after_event():
    """Social influence should be clamped to [0, 1] after event."""
    engine = SocialInfluenceEngine()
    npc = _make_npc(
        name="extreme",
        social_influence=np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9], dtype=np.float32),
    )
    event = WorldEvent(
        event_type="big",
        intensity=1.0,
        social_impact=np.ones(SOCIAL_INFLUENCE_DIM, dtype=np.float32),
    )
    engine.apply_event(npc, event)
    assert np.all(npc.social_influence <= 1.0)
    assert np.all(npc.social_influence >= 0.0)


# --- Tick (per-tick dynamics) ---


def test_tick_with_peer_pressure():
    """NPCs near an archetype should drift toward that archetype's profile."""
    engine = SocialInfluenceEngine(blend_rate=0.5, decay_rate=0.0)
    receiver = _make_npc(
        name="receiver",
        location_id="temple",
        personality=np.array([0.5, 0.5, 0.5, 0.8, 0.2], dtype=np.float32),
    )
    priest = _make_npc(
        name="priest",
        archetype="priest",
        location_id="temple",
    )
    engine.tick([receiver, priest])
    assert receiver.social_influence[4] > 0.0, (
        "Receiver near priest should gain religious_devotion"
    )


def test_tick_decay_without_peers():
    """Without peers, social influence should decay toward zero."""
    engine = SocialInfluenceEngine(blend_rate=0.0, decay_rate=0.1)
    npc = _make_npc(
        name="isolated",
        location_id="wilderness",
        social_influence=np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32),
    )
    engine.tick([npc])
    assert np.all(npc.social_influence < 0.5), (
        "Isolated NPC social influence should decay"
    )


def test_tick_clamps_values():
    """Social influence should stay in [0, 1] after tick."""
    engine = SocialInfluenceEngine()
    npc = _make_npc(
        name="edge",
        social_influence=np.array([0.99, 0.0, 0.5, 0.5, 0.5, 0.5], dtype=np.float32),
    )
    engine.tick([npc])
    assert np.all(npc.social_influence >= 0.0)
    assert np.all(npc.social_influence <= 1.0)


def test_multiple_ticks_convergence():
    """Over many ticks with a priest, receiver should converge toward religious."""
    engine = SocialInfluenceEngine(blend_rate=0.2, decay_rate=0.01)
    receiver = _make_npc(
        name="convert",
        location_id="temple",
        personality=np.array([0.5, 0.5, 0.5, 0.9, 0.1], dtype=np.float32),
    )
    priest = _make_npc(
        name="priest",
        archetype="priest",
        location_id="temple",
    )
    for _ in range(100):
        engine.tick([receiver, priest])
    assert receiver.social_influence[4] > 0.1, (
        "After many ticks with priest, religious_devotion should be significant"
    )


def test_multiple_ticks_decay_to_near_zero():
    """Over many ticks alone, social influence should approach zero."""
    engine = SocialInfluenceEngine(blend_rate=0.0, decay_rate=0.05)
    npc = _make_npc(
        name="fading",
        social_influence=np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8], dtype=np.float32),
    )
    for _ in range(100):
        engine.tick([npc])
    assert np.all(npc.social_influence < 0.01), (
        "Social influence should decay to near zero without peers"
    )


def test_tick_multiple_archetypes_blend():
    """Multiple archetypes at same location should blend their signals."""
    engine = SocialInfluenceEngine(blend_rate=0.5, decay_rate=0.0)
    receiver = _make_npc(
        name="receiver",
        location_id="plaza",
        personality=np.array([0.5, 0.5, 0.5, 0.8, 0.2], dtype=np.float32),
    )
    merchant = _make_npc(name="merchant", archetype="merchant", location_id="plaza")
    priest = _make_npc(name="priest", archetype="priest", location_id="plaza")
    engine.tick([receiver, merchant, priest])
    # Should have some of both economic and religious
    assert receiver.social_influence[1] > 0.0, "Should absorb economic from merchant"
    assert receiver.social_influence[4] > 0.0, "Should absorb religious from priest"


def test_different_locations_no_cross_influence():
    """NPCs in different locations should not influence each other."""
    engine = SocialInfluenceEngine(blend_rate=0.5, decay_rate=0.0)
    npc_a = _make_npc(name="a", location_id="forest")
    npc_b = _make_npc(
        name="b",
        archetype="priest",
        location_id="temple",
    )
    engine.tick([npc_a, npc_b])
    assert np.all(npc_a.social_influence == 0.0), (
        "NPC in forest should not be influenced by priest in temple"
    )
