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

"""Template-based dialogue engine for ambient NPC barks (Tier 1).

Zero-cost, zero-latency dialogue via parameterized templates.
Templates are selected by the NPC's dominant intention + emotion.
"""

from __future__ import annotations

import random

from app.models.npc_status import NPCVectorialStatus

# Template library keyed by (dominant_intention, dominant_emotion)
# Each entry is a list of templates with {name} placeholder.
_TEMPLATES: dict[tuple[str, str], list[str]] = {
    ("survive", "fear"): [
        "I don't feel safe here...",
        "We need to be careful. Something feels wrong.",
        "I heard strange noises last night.",
    ],
    ("survive", "anger"): [
        "I won't let anyone threaten my family!",
        "These are dangerous times. We must fight back.",
    ],
    ("socialize", "joy"): [
        "What a wonderful day to meet friends!",
        "Have you heard the latest news? Come, let me tell you!",
        "It's good to see a friendly face around here.",
    ],
    ("socialize", "sadness"): [
        "I've been feeling lonely lately...",
        "I wish I had someone to talk to.",
    ],
    ("achieve", "anticipation"): [
        "I have big plans. Just you wait.",
        "Every day I'm closer to my goal.",
        "Hard work pays off. I truly believe that.",
    ],
    ("achieve", "joy"): [
        "Business is booming! What a time to be alive!",
        "I just closed an excellent deal.",
    ],
    ("explore", "anticipation"): [
        "I wonder what lies beyond those mountains...",
        "There's so much of the world I haven't seen.",
    ],
    ("explore", "surprise"): [
        "Did you see that? I've never seen anything like it!",
        "This place is full of wonders.",
    ],
    ("create", "joy"): [
        "I've been working on something special.",
        "Inspiration struck me this morning!",
    ],
    ("create", "anticipation"): [
        "I can see it in my mind... it will be magnificent.",
        "I just need a few more materials to finish my work.",
    ],
    ("dominate", "anger"): [
        "This town needs stronger leadership.",
        "People should know their place.",
    ],
    ("dominate", "trust"): [
        "Follow my lead and everything will be fine.",
        "I've got everything under control.",
    ],
    ("nurture", "trust"): [
        "How are you feeling today? You look tired.",
        "If you need anything, don't hesitate to ask.",
    ],
    ("nurture", "sadness"): [
        "I worry about the children in times like these.",
        "We need to take care of each other.",
    ],
    ("escape", "fear"): [
        "I need to get out of here...",
        "I'm saving up to leave this place for good.",
    ],
    ("escape", "sadness"): [
        "There's nothing left for me here.",
        "Sometimes I dream of a different life...",
    ],
}

# Fallback templates when no specific match exists
_FALLBACK_TEMPLATES: list[str] = [
    "...",
    "Hmm.",
    "Another day, I suppose.",
    "The weather's been something, hasn't it?",
    "Stay safe out there.",
]


class TemplateEngine:
    """Generates ambient NPC dialogue from parameterized templates.

    Templates are selected based on the NPC's dominant intention and emotion.
    If no specific template matches, a generic fallback is used.
    """

    def __init__(
        self,
        custom_templates: dict[tuple[str, str], list[str]] | None = None,
    ):
        self._templates = dict(_TEMPLATES)
        if custom_templates:
            self._templates.update(custom_templates)

    def generate(self, npc: NPCVectorialStatus) -> str:
        """Generate an ambient dialogue line for an NPC.

        Args:
            npc: The NPC to generate dialogue for.

        Returns:
            A dialogue string.
        """
        key = (npc.dominant_intention(), npc.dominant_emotion())
        candidates = self._templates.get(key, _FALLBACK_TEMPLATES)
        return random.choice(candidates)

    def generate_batch(self, npcs: list[NPCVectorialStatus]) -> list[str]:
        """Generate ambient dialogue for multiple NPCs.

        Args:
            npcs: List of NPCs.

        Returns:
            List of dialogue strings (same order as input).
        """
        return [self.generate(npc) for npc in npcs]
