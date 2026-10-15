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

"""KVCacheStore — backend-agnostic data carrier for KV cache mapping tables."""

from __future__ import annotations

import threading
from collections.abc import Iterable, Mapping

from ..types import Layer
from .cache_registry import CacheRegistryState, build_cache_registry
from .group_cache import GroupCache, PrefixEstimate


class KVCacheStore:
    """Mutable data carrier for KV cache mapping tables.

    Attributes:
        block_size: Learned block size (None until first BlockStored event).
        replicas_by_block: local prefix hash → set of replica_ids that cache
            it on GPU.  CPU/SSD blocks are counted but not indexed here.
    """

    _instance: KVCacheStore | None = None

    def __init__(self) -> None:
        self.block_size: int | None = None
        self._registry_state = CacheRegistryState()
        self.registry_configured = False
        self.registry_generation = 0
        self.group_caches: dict[str, GroupCache] = {}
        self.replicas_by_block: dict[str, set[str]] = {}
        # Per-layer per-replica block counts, maintained alongside replicas_by_block.
        self._replica_layer_counts: dict[Layer, dict[str, int]] = {
            Layer.GPU: {},
            Layer.CPU: {},
            Layer.SSD: {},
        }
        self._lock: threading.Lock = threading.Lock()

    @classmethod
    def singleton(cls) -> KVCacheStore:
        """Return the shared singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def registry_state(self) -> CacheRegistryState:
        """The pool's immutable layout, or the reason registration is unavailable."""
        with self._lock:
            return self._registry_state

    def configure_registry(self, metadata) -> CacheRegistryState:
        """Install a complete validated layout before collectors start."""
        state = build_cache_registry(metadata)
        with self._lock:
            self._registry_state = state
            # Missing/unavailable metadata keeps the original event path.
            # An explicit layout (even unsupported/invalid) or disabled prefix
            # caching must not silently fall back to Full-style indexing.
            self.registry_configured = (
                state.registry is not None
                or state.status == "disabled"
                or (isinstance(metadata, Mapping) and "groups" in metadata)
            )
            self.registry_generation += 1
            self.group_caches.clear()
        return state

    def apply_group_events(self, node_id, events) -> None:
        with self._lock:
            registry = self._registry_state.registry
            if registry is None:
                return
            index = self.group_caches.get(node_id)
            if index is None:
                index = self.group_caches[node_id] = GroupCache(registry, node_id)
            index.apply(events)

    def estimate_prefixes(self, node_ids, hashes, query_tokens) -> dict[str, PrefixEstimate] | None:
        with self._lock:
            if not self.registry_configured:
                return None  # Original single-group protocol remains available.
            state = self._registry_state
            result = {}
            for node in node_ids:
                index = self.group_caches.get(node)
                if state.registry is None:
                    result[node] = PrefixEstimate(state.status, 0, query_tokens, 0, None, state.reason)
                elif index is None:
                    result[node] = PrefixEstimate(
                        "warming",
                        0,
                        query_tokens,
                        max(0, query_tokens - 1),
                        state.registry.local_hash_unit,
                        "no_events",
                    )
                else:
                    result[node] = index.estimate(hashes, query_tokens)
            return result

    # ── Replica management ──────────────────────────────────────────────

    def clear_replica(self, replica_id: str) -> None:
        """Clear all blocks for a replica from the reverse index.

        O(n) in the number of unique blocks, but replica count is typically small (< 100).
        """
        with self._lock:
            self.group_caches.pop(replica_id, None)
            stale_hashes: list[str] = []
            for bh, replicas in self.replicas_by_block.items():
                if replica_id not in replicas:
                    continue
                replicas.discard(replica_id)
                if not replicas:
                    stale_hashes.append(bh)
            for bh in stale_hashes:
                del self.replicas_by_block[bh]
            for layer_counts in self._replica_layer_counts.values():
                layer_counts.pop(replica_id, None)

    # ── Block management ────────────────────────────────────────────────

    def add_blocks(self, replica_id: str, block_hashes: Iterable[str], layer: Layer = Layer.GPU) -> None:
        """Add blocks to a replica at a layer, updating the reverse index.

        Only GPU blocks are indexed in ``replicas_by_block`` (they drive
        prefix-hit routing); CPU/SSD blocks are counted only.
        """
        with self._lock:
            layer_counts = self._replica_layer_counts.setdefault(layer, {})
            for bh in block_hashes:
                layer_counts[replica_id] = layer_counts.get(replica_id, 0) + 1
                if layer != Layer.GPU:
                    continue
                if bh not in self.replicas_by_block:
                    self.replicas_by_block[bh] = set()
                self.replicas_by_block[bh].add(replica_id)

    def remove_blocks(self, replica_id: str, block_hashes: Iterable[str], layer: Layer = Layer.GPU) -> None:
        """Remove blocks from a replica at a layer, updating the reverse index."""
        with self._lock:
            layer_counts = self._replica_layer_counts.setdefault(layer, {})
            for bh in block_hashes:
                layer_counts[replica_id] = layer_counts.get(replica_id, 0) - 1
                if layer != Layer.GPU:
                    continue
                replicas = self.replicas_by_block.get(bh)
                if replicas is not None and replica_id in replicas:
                    replicas.discard(replica_id)
                    if not replicas:
                        del self.replicas_by_block[bh]

    # ── Retained-cache size ─────────────────────────────────────────────

    def per_replica_block_counts(self) -> dict[str, int]:
        """Return per-replica GPU cache retention estimates in O(replicas).

        Hybrid counts published group/hash identities, including unresolved
        events, not local hash units or deduplicated physical blocks. Legacy
        keeps its original incremental event counts.
        """
        with self._lock:
            if self.registry_configured:
                return {node: len(index.retained_hashes) for node, index in self.group_caches.items()}
            return dict(self._replica_layer_counts.get(Layer.GPU, {}))

    # ── Prefix hit rate queries ─────────────────────────────────────────

    def get_layer_prefix_hit_rate(
        self,
        node_id: str,
        hash_strs: list[str],
        layer: Layer = Layer.GPU,
    ) -> float:
        """Prefix-cache hit rate for a node at a layer, ∈ [0.0, 1.0].

        GPU: walk the local reverse index (``replicas_by_block``) along the
        supplied ``hash_strs`` chain until a hash isn't cached on this node.

        CPU/SSD return 0.0 today — not just because the mooncake collector is
        unwired, but because ``add_blocks`` only indexes GPU blocks into
        ``replicas_by_block`` (CPU/SSD are counted in ``_replica_layer_counts``
        only). Supporting CPU/SSD hit queries requires extending the reverse
        index to be layer-keyed (``dict[Layer, dict[str, set[str]]]``) and
        indexing those blocks in ``add_blocks``; until then ``layer`` here is a
        placeholder that short-circuits to 0.0.
        """
        if layer != Layer.GPU or self.block_size is None:
            return 0.0
        with self._lock:
            return self._gpu_hit_rate_locked(node_id, hash_strs)

    def _gpu_hit_rate_locked(self, node_id: str, hash_strs: list[str]) -> float:
        """Walk the reverse index along the hash chain. Caller holds ``_lock``."""
        if not hash_strs:
            return 0.0
        matched = 0
        for i, hs in enumerate(hash_strs):
            cached = self.replicas_by_block.get(hs)
            if cached is None or node_id not in cached:
                break  # chain break — this node doesn't cache this hash
            matched = i + 1
        return matched / len(hash_strs)
