# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Exercise the real integration mixin against a small engine API double."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from uni_agent.agent_aware_router.server.cache_lifecycle import CacheLifecycle
from uni_agent.agent_aware_router.server.lifecycle_server import CacheLifecycleServerMixin
from verl.workers.rollout.replica import RolloutMode

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


class Base:
    async def abort_all_requests(self, reset_prefix_cache=True, reject_request=False, abort_only=False):
        if self.node_rank != 0:
            return {"aborted_count": 0, "request_ids": []}
        if abort_only:
            return await self.engine.abort_current_requests()
        await self.engine.pause_generation(clear_cache=reset_prefix_cache)
        self._rejecting = reject_request
        return {"aborted_count": 0}

    async def sleep(self):
        if self.config.free_cache_engine and self.rollout_mode != RolloutMode.STANDALONE:
            await self.engine.sleep(level=2 if self.rollout_mode == RolloutMode.HYBRID else 1)

    async def wake_up(self, tags=None):
        pass

    async def release_kv_cache(self):
        if self.config.free_cache_engine and self.rollout_mode != RolloutMode.COLOCATED:
            await self.engine.sleep(level=2)
            await self.engine.wake_up(tags=["weights"])

    async def resume_kv_cache(self):
        pass

    async def open_submission_gate(self):
        self._submission_paused = self._rejecting = False
        self._resume_event.set()

    def _get_wake_up_tags(self):
        return ["weights", "kv_cache"]


class Server(CacheLifecycleServerMixin, Base):
    def __init__(self):
        self.node_rank = 0
        self.config = SimpleNamespace(free_cache_engine=True)
        self.rollout_mode = RolloutMode.COLOCATED
        self._submission_paused = self._rejecting = False
        self._admitting = 0
        self._resume_event = asyncio.Event()
        self._resume_event.set()
        self._cache_lifecycle = CacheLifecycle("0.29.0")
        self.engine = SimpleNamespace(
            reset_prefix_cache=AsyncMock(return_value=True),
            reset_mm_cache=AsyncMock(),
            reset_encoder_cache=AsyncMock(),
            pause_generation=AsyncMock(),
            sleep=AsyncMock(),
            wake_up=AsyncMock(),
            is_sleeping=AsyncMock(return_value=False),
            wait_for_requests_to_drain=AsyncMock(),
        )

    async def _ensure_cache_bridge(self):
        return self._cache_lifecycle


@pytest.mark.parametrize("paused,rejecting", [(False, False), (True, True)])
def test_abort_only_keeps_cache_state_and_existing_admission_state(paused, rejecting):
    async def run():
        server = Server()
        server._submission_paused, server._rejecting = paused, rejecting
        if paused:
            server._resume_event.clear()
        before = server._cache_lifecycle.state.to_dict()
        result = {"aborted_count": 2, "request_ids": ["first", "second"]}
        server.engine.abort_current_requests = AsyncMock(return_value=result)
        server._ensure_cache_bridge = AsyncMock(
            side_effect=AssertionError("cancellation must not start a cache operation")
        )
        server._cache_lifecycle.notifier = AsyncMock()
        observed = await server.abort_all_requests(reset_prefix_cache=True, reject_request=True, abort_only=True)
        assert observed == result
        server.engine.abort_current_requests.assert_awaited_once_with()
        server.engine.pause_generation.assert_not_awaited()
        server.engine.reset_prefix_cache.assert_not_awaited()
        server._cache_lifecycle.notifier.assert_not_awaited()
        assert server._cache_lifecycle.state.to_dict() == before
        assert (server._submission_paused, server._rejecting) == (paused, rejecting)
        assert server._resume_event.is_set() is not paused

    asyncio.run(run())


def test_abort_only_headless_server_does_not_touch_engine_or_cache_control():
    async def run():
        server = Server()
        server.node_rank = 1
        del server.engine
        server._ensure_cache_bridge = AsyncMock(side_effect=AssertionError("headless server has no cache bridge"))
        before = server._cache_lifecycle.state.to_dict()
        assert await server.abort_all_requests(abort_only=True) == {"aborted_count": 0, "request_ids": []}
        assert server._cache_lifecycle.state.to_dict() == before
        assert not server._submission_paused

    asyncio.run(run())


def test_failed_abort_only_does_not_change_cache_or_admission():
    async def run():
        server = Server()
        server.engine.abort_current_requests = AsyncMock(side_effect=RuntimeError("cancel failed"))
        before = server._cache_lifecycle.state.to_dict()
        with pytest.raises(RuntimeError, match="cancel failed"):
            await server.abort_all_requests(abort_only=True)
        assert server._cache_lifecycle.state.to_dict() == before
        assert not server._submission_paused and server._resume_event.is_set()
        server.engine.pause_generation.assert_not_awaited()

    asyncio.run(run())


def test_clear_ack_precedes_reset_and_reopens_admission():
    async def run():
        server = Server()
        seen = []

        async def notify(sink, state):
            if state["serving"] is False:
                server.engine.reset_prefix_cache.assert_not_awaited()
            seen.append(state)

        server._cache_lifecycle.notifier = notify
        server._cache_lifecycle.bind({"token": "sink"})
        await server.clear_kv_cache()
        assert seen[0]["index_state"] == "awaiting_clear"
        server.engine.reset_prefix_cache.assert_awaited_once_with(reset_connector=True)
        assert not server._submission_paused and server._cache_lifecycle.state.serving
        assert server._resume_event.is_set()
        assert server._cache_lifecycle.state.index_state == "awaiting_clear"

    asyncio.run(run())


def test_failed_reset_keeps_gate_closed_and_cannot_resume():
    async def run():
        server = Server()
        server.engine.reset_prefix_cache.return_value = False
        with pytest.raises(RuntimeError, match="confirm"):
            await server.clear_kv_cache()
        assert server._submission_paused and not server._cache_lifecycle.state.serving
        with pytest.raises(RuntimeError, match="repaired"):
            await server.open_submission_gate()

    asyncio.run(run())


def test_sleep_wake_restores_unpaused_server_but_abort_requires_explicit_resume():
    async def run():
        server = Server()
        await server.sleep()
        assert server._submission_paused and not server._cache_lifecycle.state.serving
        await server.wake_up()
        assert not server._submission_paused and server._cache_lifecycle.state.expected_clears == 2
        await server.abort_all_requests()
        await server.sleep()
        await server.wake_up()
        assert server._submission_paused and not server._cache_lifecycle.state.serving
        await server.open_submission_gate()
        assert server._cache_lifecycle.state.serving

    asyncio.run(run())


def test_partial_wake_and_noop_paths_do_not_open_routing():
    async def run():
        server = Server()
        server.rollout_mode = RolloutMode.HYBRID
        await server.sleep()
        server.engine.is_sleeping.return_value = True
        await server.wake_up(tags=["weights"])
        assert not server._cache_lifecycle.state.serving
        server.rollout_mode = RolloutMode.STANDALONE
        before = server._cache_lifecycle.state
        await server.sleep()
        await server.wake_up()
        assert server._cache_lifecycle.state == before

    asyncio.run(run())


def test_pause_keep_cache_and_hybrid_protocol_count_each_real_reset_once():
    async def run():
        server = Server()
        await server.abort_all_requests(reset_prefix_cache=False)
        assert server._cache_lifecycle.state.epoch == 0
        await server.open_submission_gate()
        server.rollout_mode = RolloutMode.HYBRID
        await server.abort_all_requests()
        await server.sleep()
        server.engine.is_sleeping.return_value = True
        await server.wake_up(tags=["weights"])
        await server.clear_kv_cache()
        server.engine.is_sleeping.return_value = False
        await server.resume_kv_cache()
        assert server._cache_lifecycle.state.expected_clears == 5
        assert not server._cache_lifecycle.state.serving
        await server.open_submission_gate()
        assert server._cache_lifecycle.state.serving

    asyncio.run(run())
