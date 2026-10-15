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

"""Paused replicas are excluded before every strategy path."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.uni_agent.agent_aware_router.balancer._helpers import _make_balancer
from uni_agent.agent_aware_router.types.cache_lifecycle import CacheLifecycleState

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


def test_sticky_and_random_fallback_cannot_select_paused_replica(monkeypatch):
    router = _make_balancer()
    router.acquire_server("session")
    bound = router._store.get_sticky_binding("session")
    state = CacheLifecycleState("engine", serving=False, index_state="awaiting_clear")
    router.update_cache_lifecycle(bound, state.to_dict())
    chosen, _ = router.acquire_server("session")
    assert chosen != bound

    def fail(*args, **kwargs):
        raise RuntimeError("scoring failure")

    monkeypatch.setattr(router._strategy, "score", fail)
    assert router.acquire_server("new-session")[0] != bound
    other = "s1" if bound == "s0" else "s0"
    router.update_cache_lifecycle(other, state.to_dict())
    with pytest.raises(RuntimeError, match="no available replicas"):
        router.acquire_server("none")


def test_late_control_update_does_not_reopen_and_does_not_erase_inflight():
    router = _make_balancer({"only": "server"})
    chosen, _ = router.acquire_server("session")
    state = CacheLifecycleState("engine", revision=2, serving=False, index_state="awaiting_clear")
    router.update_cache_lifecycle(chosen, state.to_dict())
    assert router.get_total_inflight() == 1
    router.update_cache_lifecycle(chosen, replace(state, revision=1, serving=True).to_dict())
    with pytest.raises(RuntimeError, match="no available replicas"):
        router.acquire_server("blocked")
    router.release_server(chosen, "session")
    assert router.get_total_inflight() == 0


def test_readded_paused_server_with_equal_revision_stays_unavailable(monkeypatch):
    from tests.uni_agent.agent_aware_router.balancer._helpers import _FakeCollectorManager, _router_config
    from uni_agent.agent_aware_router.balancer import KVCAwareBalancer

    state = CacheLifecycleState("engine", revision=1, serving=False)
    handle = SimpleNamespace(
        register_cache_lifecycle_sink=SimpleNamespace(
            remote=Mock(return_value={"state": state.to_dict(), "sources": {}})
        ),
        unregister_cache_lifecycle_sink=SimpleNamespace(remote=Mock(return_value=None)),
    )
    monkeypatch.setattr("ray.get", lambda value, **kwargs: value)
    monkeypatch.setattr("ray.util.get_node_ip_address", lambda: "127.0.0.1")
    router = KVCAwareBalancer({"s0": handle}, _router_config(), provider_factory=_FakeCollectorManager)
    try:
        router.remove_servers(["s0"])
        router.add_servers({"s0": handle})
        with pytest.raises(RuntimeError, match="no available replicas"):
            router.acquire_server("paused")
    finally:
        router.close()
