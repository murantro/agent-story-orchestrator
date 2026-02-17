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

"""Background tick runner for autonomous simulation.

Provides an optional push-based tick loop that advances the simulation
automatically. Can coexist with pull-based ticks (POST /tick).
"""

from __future__ import annotations

import asyncio
import logging

from app.config import BACKGROUND_TICK_DELTA_HOURS, BACKGROUND_TICK_INTERVAL_SECONDS
from app.world.world_state import WorldStateManager

logger = logging.getLogger(__name__)


class TickRunner:
    """Background asyncio task that periodically advances the simulation.

    Pull-based: Unity calls POST /tick when it wants.
    Push-based: TickRunner calls world.tick() every interval_seconds.
    Both can coexist safely because WorldStateManager uses an asyncio lock.

    Args:
        world: The world state manager to tick.
        interval_seconds: Real seconds between auto-ticks.
        delta_hours: In-game hours to advance per tick.
    """

    def __init__(
        self,
        world: WorldStateManager,
        interval_seconds: float = BACKGROUND_TICK_INTERVAL_SECONDS,
        delta_hours: float = BACKGROUND_TICK_DELTA_HOURS,
    ):
        self._world = world
        self._interval_seconds = interval_seconds
        self._delta_hours = delta_hours
        self._task: asyncio.Task | None = None
        self._running = False
        self._ticks_completed = 0

    @property
    def running(self) -> bool:
        """Whether the background loop is currently running."""
        return self._running

    @property
    def ticks_completed(self) -> int:
        """Number of ticks completed since last start."""
        return self._ticks_completed

    @property
    def interval_seconds(self) -> float:
        """Current interval between ticks in real seconds."""
        return self._interval_seconds

    async def start(self) -> None:
        """Start the background tick loop.

        Raises:
            RuntimeError: If already running.
        """
        if self._running:
            raise RuntimeError("Tick runner is already running.")
        self._running = True
        self._ticks_completed = 0
        self._task = asyncio.create_task(self._loop())
        logger.info(
            "Tick runner started: interval=%.1fs, delta=%.1fh",
            self._interval_seconds,
            self._delta_hours,
        )

    async def stop(self) -> None:
        """Gracefully stop the background tick loop."""
        if not self._running:
            return
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Tick runner stopped after %d ticks.", self._ticks_completed)

    async def _loop(self) -> None:
        """Internal loop that runs ticks until stopped."""
        while self._running:
            try:
                await self._world.tick(self._delta_hours)
                self._ticks_completed += 1
            except Exception:
                logger.exception("Error during background tick")
            await asyncio.sleep(self._interval_seconds)
