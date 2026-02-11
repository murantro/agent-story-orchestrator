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

"""Tests for the intention computation engine."""

import numpy as np

from app.models.npc_status import (
    EMOTION_DIM,
    ENVIRONMENT_DIM,
    INTENTION_DIM,
    PERSONALITY_DIM,
    SOCIAL_INFLUENCE_DIM,
    NPCVectorialStatus,
)
from app.simulation.intention_engine import ArchetypeWeights, IntentionEngine


def _make_npc(**kwargs) -> NPCVectorialStatus:
    """Helper to create an NPC with specific vector values."""
    return NPCVectorialStatus(**kwargs)


def test_compute_returns_normalized_vector():
    """Intention output should always be L2-normalized."""
    engine = IntentionEngine()
    npc = _make_npc(
        name="test",
        emotion=np.ones(EMOTION_DIM, dtype=np.float32),
        personality=np.ones(PERSONALITY_DIM, dtype=np.float32),
    )
    result = engine.compute(npc)
    assert result.shape == (INTENTION_DIM,)
    norm = np.linalg.norm(result)
    assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"


def test_compute_with_zero_inputs_returns_uniform():
    """If all inputs are zero, result should be uniform (fallback)."""
    engine = IntentionEngine()
    npc = _make_npc(
        name="zero",
        intention=np.zeros(INTENTION_DIM, dtype=np.float32),
        emotion=np.zeros(EMOTION_DIM, dtype=np.float32),
        personality=np.zeros(PERSONALITY_DIM, dtype=np.float32),
        social_influence=np.zeros(SOCIAL_INFLUENCE_DIM, dtype=np.float32),
        environment=np.zeros(ENVIRONMENT_DIM, dtype=np.float32),
    )
    result = engine.compute(npc)
    expected = np.ones(INTENTION_DIM) / INTENTION_DIM
    np.testing.assert_allclose(result, expected, atol=1e-5)


def test_momentum_preserves_direction():
    """With only momentum weight, intention should stay roughly the same."""
    weights = ArchetypeWeights(
        w_personality=0.0,
        w_emotion=0.0,
        w_social=0.0,
        w_environment=0.0,
        w_momentum=1.0,
    )
    engine = IntentionEngine({"test_arch": weights})
    original = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    npc = _make_npc(
        name="momentum_test",
        archetype="test_arch",
        intention=original,
    )
    result = engine.compute(npc)
    # Should point in the same direction as original
    cosine = np.dot(result, original) / (
        np.linalg.norm(result) * np.linalg.norm(original)
    )
    assert cosine > 0.99, f"Expected same direction, cosine={cosine}"


def test_tick_mutates_npcs():
    """tick() should update intention vectors in-place."""
    engine = IntentionEngine()
    npc = _make_npc(
        name="mutable",
        emotion=np.ones(EMOTION_DIM, dtype=np.float32) * 0.5,
    )
    old_intention = npc.intention.copy()
    engine.tick([npc])
    # Intention should have changed (not identical to old)
    assert not np.allclose(npc.intention, old_intention) or np.allclose(
        old_intention, npc.intention
    )
    # Should still be normalized
    assert abs(np.linalg.norm(npc.intention) - 1.0) < 1e-5


def test_compute_batch_matches_individual():
    """Batch computation should give same results as individual."""
    engine = IntentionEngine()
    npcs = [
        _make_npc(
            name=f"npc_{i}", emotion=np.random.rand(EMOTION_DIM).astype(np.float32)
        )
        for i in range(5)
    ]
    batch_results = engine.compute_batch(npcs)
    individual_results = [engine.compute(npc) for npc in npcs]

    for batch_r, ind_r in zip(batch_results, individual_results, strict=True):
        np.testing.assert_allclose(batch_r, ind_r, atol=1e-6)


def test_register_archetype():
    """Custom archetype weights should be retrievable."""
    engine = IntentionEngine()
    custom = ArchetypeWeights(w_momentum=0.9, w_emotion=0.1)
    engine.register_archetype("guard", custom)
    assert engine.get_weights("guard") is custom
    assert engine.get_weights("unknown") is not custom
