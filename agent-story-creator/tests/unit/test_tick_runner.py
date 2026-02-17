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

"""Tests for TickRunner."""

from __future__ import annotations

import asyncio

import pytest

from app.models.npc_status import NPCVectorialStatus
from app.world.tick_runner import TickRunner
from app.world.world_state import WorldStateManager


@pytest.mark.asyncio
async def test_start_and_stop():
    world = WorldStateManager()
    runner = TickRunner(world, interval_seconds=0.05, delta_hours=1.0)

    assert not runner.running
    await runner.start()
    assert runner.running

    await asyncio.sleep(0.15)
    await runner.stop()

    assert not runner.running
    assert runner.ticks_completed >= 1


@pytest.mark.asyncio
async def test_advances_game_time():
    world = WorldStateManager(game_time=0.0)
    world.add_npc(NPCVectorialStatus(npc_id="npc-1", name="Guard"))

    runner = TickRunner(world, interval_seconds=0.05, delta_hours=2.0)
    await runner.start()
    await asyncio.sleep(0.15)
    await runner.stop()

    assert world.game_time > 0.0
    assert runner.ticks_completed >= 1


@pytest.mark.asyncio
async def test_start_twice_raises():
    world = WorldStateManager()
    runner = TickRunner(world, interval_seconds=0.1)

    await runner.start()
    with pytest.raises(RuntimeError, match="already running"):
        await runner.start()
    await runner.stop()


@pytest.mark.asyncio
async def test_stop_when_not_running_is_safe():
    world = WorldStateManager()
    runner = TickRunner(world, interval_seconds=0.1)
    await runner.stop()  # Should not raise


@pytest.mark.asyncio
async def test_properties():
    world = WorldStateManager()
    runner = TickRunner(world, interval_seconds=3.0, delta_hours=0.5)

    assert runner.interval_seconds == 3.0
    assert runner.ticks_completed == 0
    assert not runner.running
