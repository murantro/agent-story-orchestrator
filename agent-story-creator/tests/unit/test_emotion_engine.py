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

"""Tests for the emotion engine."""

import numpy as np

from app.models.events import WorldEvent
from app.models.npc_status import (
    EMOTION_DIM,
    PERSONALITY_DIM,
    NPCVectorialStatus,
)
from app.simulation.emotion_engine import EmotionEngine


def _make_npc(**kwargs) -> NPCVectorialStatus:
    return NPCVectorialStatus(**kwargs)


def test_decay_moves_toward_baseline():
    """Emotion should drift toward personality-derived baseline."""
    engine = EmotionEngine(decay_rate=0.5)
    npc = _make_npc(
        name="decayer",
        emotion=np.ones(EMOTION_DIM, dtype=np.float32),
        personality=np.zeros(PERSONALITY_DIM, dtype=np.float32),
    )
    baseline = engine.compute_baseline(npc.personality)
    original_distance = np.linalg.norm(npc.emotion - baseline)

    new_emotion = engine.decay(npc)
    new_distance = np.linalg.norm(new_emotion - baseline)

    assert new_distance < original_distance, "Decay should reduce distance to baseline"


def test_decay_clamps_to_0_1():
    """Emotion values should never go below 0 or above 1."""
    engine = EmotionEngine(decay_rate=0.1)
    npc = _make_npc(
        name="clamped",
        emotion=np.array([2.0, -1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32),
        personality=np.ones(PERSONALITY_DIM, dtype=np.float32),
    )
    result = engine.decay(npc)
    assert np.all(result >= 0.0), "Emotions should not go below 0"
    assert np.all(result <= 1.0), "Emotions should not exceed 1"


def test_apply_event_shifts_emotion():
    """Event impact should shift emotion vector."""
    engine = EmotionEngine()
    npc = _make_npc(
        name="reactor",
        emotion=np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0], dtype=np.float32),
    )
    event = WorldEvent(
        event_type="murder",
        intensity=1.0,
        emotion_impact=np.array(
            [-0.3, 0.5, 0.2, 0.4, 0.1, 0.1, -0.3, 0.0], dtype=np.float32
        ),
    )
    result = engine.apply_event(npc, event)
    # Joy should decrease, sadness should increase
    assert result[0] < npc.emotion[0], "Joy should decrease after murder"
    assert result[1] > npc.emotion[1], "Sadness should increase after murder"


def test_apply_event_respects_intensity():
    """Lower intensity events should have smaller impact."""
    engine = EmotionEngine()
    npc = _make_npc(
        name="intensity_test",
        emotion=np.zeros(EMOTION_DIM, dtype=np.float32),
    )
    impact = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    event_full = WorldEvent(event_type="big", intensity=1.0, emotion_impact=impact)
    event_half = WorldEvent(event_type="small", intensity=0.5, emotion_impact=impact)

    result_full = engine.apply_event(npc, event_full)
    result_half = engine.apply_event(npc, event_half)

    assert np.all(result_full >= result_half), (
        "Higher intensity should produce larger shifts"
    )


def test_tick_mutates_emotions():
    """tick() should update emotion vectors in-place."""
    engine = EmotionEngine(decay_rate=0.5)
    npc = _make_npc(
        name="tick_test",
        emotion=np.ones(EMOTION_DIM, dtype=np.float32),
        personality=np.zeros(PERSONALITY_DIM, dtype=np.float32),
    )
    old_emotion = npc.emotion.copy()
    engine.tick([npc])
    assert not np.array_equal(npc.emotion, old_emotion), "tick should mutate emotion"


def test_compute_baseline_from_personality():
    """Baseline should be deterministic for a given personality."""
    engine = EmotionEngine()
    personality = np.array([0.8, 0.6, 0.9, 0.7, 0.2], dtype=np.float32)
    b1 = engine.compute_baseline(personality)
    b2 = engine.compute_baseline(personality)
    np.testing.assert_array_equal(b1, b2)
    assert np.all(b1 >= 0.0) and np.all(b1 <= 1.0)
