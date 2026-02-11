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

"""Tests for the NPC vectorial status model."""

import numpy as np

from app.models.npc_status import (
    EMOTION_DIM,
    INTENTION_DIM,
    PERSONALITY_DIM,
    NPCVectorialStatus,
)


def test_default_vectors_have_correct_dimensions():
    """Default NPC should have correctly-sized vectors."""
    npc = NPCVectorialStatus(name="test")
    assert npc.intention.shape == (INTENTION_DIM,)
    assert npc.emotion.shape == (EMOTION_DIM,)
    assert npc.personality.shape == (PERSONALITY_DIM,)


def test_dominant_intention():
    """dominant_intention should return the label of the largest dimension."""
    npc = NPCVectorialStatus(name="explorer")
    npc.intention = np.zeros(INTENTION_DIM, dtype=np.float32)
    npc.intention[3] = 1.0  # index 3 = "explore"
    assert npc.dominant_intention() == "explore"


def test_dominant_emotion():
    """dominant_emotion should return the label of the largest dimension."""
    npc = NPCVectorialStatus(name="angry")
    npc.emotion = np.zeros(EMOTION_DIM, dtype=np.float32)
    npc.emotion[2] = 1.0  # index 2 = "anger"
    assert npc.dominant_emotion() == "anger"


def test_to_character_sheet_contains_name():
    """Character sheet should include the NPC name."""
    npc = NPCVectorialStatus(name="Elara", archetype="merchant")
    sheet = npc.to_character_sheet()
    assert "Elara" in sheet
    assert "merchant" in sheet


def test_to_character_sheet_contains_memories():
    """Character sheet should include recent memories."""
    npc = NPCVectorialStatus(name="test")
    npc.recent_memories = ["saw a dragon", "traded goods"]
    sheet = npc.to_character_sheet()
    assert "saw a dragon" in sheet
    assert "traded goods" in sheet


def test_unique_ids():
    """Each NPC should get a unique ID by default."""
    npc1 = NPCVectorialStatus(name="a")
    npc2 = NPCVectorialStatus(name="b")
    assert npc1.npc_id != npc2.npc_id
