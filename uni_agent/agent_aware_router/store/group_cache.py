"""Replica-local event endpoints and four prefix-cache matching rules.

The caller (KVCacheStore) serializes writes and queries. Intermediate local
hashes are temporary; only reported remote endpoints become residency entries.
"""

from __future__ import annotations

import logging
import struct
import time
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from ..utils.hash import compute_hash
from .cache_registry import CacheGroup, CacheRegistry

if TYPE_CHECKING:
    from ..collectors.parse.vllm.kv_event import KVCacheEvent

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CacheEndpoint:
    local_hash: str
    tokens: int


@dataclass(frozen=True)
class PrefixEstimate:
    status: str
    estimated_hit_tokens: int
    query_tokens: int
    lookup_limit_tokens: int
    local_hash_unit: int | None
    reason: str | None = None
    replayed_tokens: int = 0

    @property
    def hit_rate(self) -> float:
        return self.estimated_hit_tokens / self.query_tokens if self.query_tokens else 0.0

    @property
    def estimated_saved_tokens(self) -> int:
        return max(0, self.estimated_hit_tokens - self.replayed_tokens)

    @property
    def saved_token_rate(self) -> float:
        return self.estimated_saved_tokens / self.query_tokens if self.query_tokens else 0.0


class GroupCache:
    """One replica's cache generation, shared hash identities, and group entries."""

    def __init__(
        self, registry: CacheRegistry, node_id: str, *, pending_limit=1024, pending_ttl=60.0, history_limit=8192
    ):
        self.registry = registry
        self.node_id = node_id
        self.epoch = 0
        self.anchors: OrderedDict[str, CacheEndpoint] = OrderedDict()
        # group, (medium, locality, ownership), remote hash -> actual endpoint
        self.entries: dict[tuple, CacheEndpoint] = {}
        # Published local GPU identities, independent of local-hash resolution.
        # Partial aliases may share physical blocks; this is a retention estimate.
        self.retained_hashes: set[tuple[int, str]] = set()
        self.by_group: dict[int, dict[str, set[str]]] = defaultdict(dict)
        self.references: Counter[str] = Counter()
        self.pending: list[tuple[float, KVCacheEvent]] = []
        self.pending_limit = pending_limit
        self.pending_ttl = pending_ttl
        self.history_limit = history_limit
        self.status = "warming"
        self.reason: str | None = "no_events"
        self.diagnostics: Counter[str] = Counter()
        self._small_events: set[tuple[int, int]] = set()

    def clear(self) -> None:
        self.epoch += 1
        self.anchors.clear()
        self.entries.clear()
        self.retained_hashes.clear()
        self.by_group.clear()
        self.references.clear()
        self.pending.clear()
        self._small_events.clear()
        self.status, self.reason = "ready", None

    def _reject_event(self, reason: str) -> bool:
        """Consume an uninterpretable event without erasing known residency."""
        self.diagnostics[reason] += 1
        if self.diagnostics[reason] == 1:
            logger.warning("Skipping KV event: replica=%s epoch=%s reason=%s", self.node_id, self.epoch, reason)
        return True  # Only missing dependencies should enter pending.

    @staticmethod
    def _domain(event: KVCacheEvent) -> tuple[str, str, str | None] | None:
        # Native older events leave locality unspecified. Never promote an
        # explicitly REMOTE/owned/offloaded event into local GPU residency.
        medium = event.medium.upper() if isinstance(event.medium, str) else None
        locality = event.locality.upper() if isinstance(event.locality, str) else "LOCAL"
        if medium not in {"GPU", "CPU", "STORAGE", "SSD"}:
            return None
        return medium, locality, event.ownership

    @staticmethod
    def _is_local_gpu(domain: tuple) -> bool:
        return domain == ("GPU", "LOCAL", None)

    def apply(self, events: list[KVCacheEvent]) -> None:
        # Share token hashing between group events within this received batch.
        chains: dict[tuple, dict[int, str]] = {}
        for event in events:
            if event.is_clear:
                self.clear()
                chains.clear()
                continue
            try:
                group = self.registry.group(event.group_idx)
            except KeyError:
                reason = f"unregistered_group:{event.group_idx}"
                self._reject_event(reason)
                # An unknown required group prevents a complete hybrid estimate,
                # but known groups can still process their Stored/Removed events.
                self.status, self.reason = "unsupported", reason
                continue
            if not group.participates:
                continue
            domain = self._domain(event)
            if domain is None:
                self._reject_event("missing_or_unknown_medium")
                continue
            if not self._is_local_gpu(domain):
                # These sources cannot establish identities for local GPU
                # events either. Their removal cannot delete local entries.
                self.diagnostics["excluded_storage_domain"] += 1
                continue
            if event.is_remove:
                self._remove(group.group_idx, domain, event.block_hashes)
            elif event.is_store:
                if event.kind != group.kind or (
                    group.match_kind == "sliding" and event.sliding_window != group.sliding_window
                ):
                    reason = f"group_metadata_mismatch:{group.group_idx}"
                    self._reject_event(reason)
                    if event.kind == "unknown":
                        self.status, self.reason = "unsupported", reason
                    continue
                if event.lora_id not in (None, 0) or event.lora_name or any(event.extra_keys or ()):
                    self.diagnostics["excluded_cache_identity"] += 1
                    continue
                self.retained_hashes.update((group.group_idx, remote) for remote in event.block_hashes)
                if event.block_size < group.block_size:
                    self.diagnostics["smaller_event_block_size"] += 1
                    warning_key = (group.group_idx, event.block_size)
                    if warning_key not in self._small_events:
                        self._small_events.add(warning_key)
                        logger.warning(
                            "Smaller KV event block size: replica=%s epoch=%s group=%s group_block_size=%s "
                            "event_block_size=%s local_hash_unit=%s",
                            self.node_id,
                            self.epoch,
                            group.group_idx,
                            group.block_size,
                            event.block_size,
                            self.registry.local_hash_unit,
                        )
                if not self._store(event, chains):
                    if self.status == "unsupported":
                        continue
                    if not any(pending == event for _, pending in self.pending):
                        self.pending.append((time.monotonic(), event))
                    excess = len(self.pending) - self.pending_limit
                    if excess > 0:
                        del self.pending[:excess]
                        self.diagnostics["pending_overflow"] += excess
            if self.status == "warming":
                self.status, self.reason = "ready", None
        self._retry(chains)
        self._prune_anchors()

    def _store(self, event: KVCacheEvent, chains: dict) -> bool:
        if not event.block_hashes:
            return True
        if event.raw_token_ids is None:
            return self._reject_event("missing_tokens")
        unit = self.registry.local_hash_unit
        parent = CacheEndpoint("0", 0)
        if event.parent_block_hash is not None:
            parent = self.anchors.get(event.parent_block_hash)
            if parent is None:
                return False
        tokens = event.raw_token_ids
        if not tokens or len(tokens) % unit or parent.tokens % unit:
            return self._reject_event("unrepresentable_event_boundary")
        key = (parent, tokens)
        chain = chains.get(key)
        if chain is None:
            chain = {}
            value = int(parent.local_hash)
            for offset in range(0, len(tokens), unit):
                value = compute_hash(value, struct.pack(f">{unit}I", *tokens[offset : offset + unit]))
                chain[parent.tokens + offset + unit] = str(value)
            chains[key] = chain
        dense = len(tokens) == len(event.block_hashes) * event.block_size
        resolved = []
        previous = parent.tokens
        for i, remote in enumerate(event.block_hashes):
            if dense:
                end = parent.tokens + (i + 1) * event.block_size
                if end not in chain:
                    return self._reject_event("unrepresentable_remote_endpoint")
                endpoint = CacheEndpoint(chain[end], end)
            else:
                # Sparse hashes omit null/masked blocks while tokens describe
                # the entire range. Only an existing identity can locate them.
                endpoint = self.anchors.get(remote)
                if endpoint is None:
                    return False
                if chain.get(endpoint.tokens) != endpoint.local_hash or endpoint.tokens <= previous:
                    return self._reject_event("sparse_endpoint_mismatch")
            resolved.append((remote, endpoint))
            previous = endpoint.tokens
        domain = self._domain(event)
        for remote, endpoint in resolved:
            # Preserve the original parser's last-write-wins mapping behavior.
            self.anchors[remote] = endpoint
            self.anchors.move_to_end(remote)
            entry_key = (event.group_idx, domain, remote)
            existing = self.entries.get(entry_key)
            if existing == endpoint:
                continue
            if existing is not None:
                old_hashes = self.by_group[event.group_idx][existing.local_hash]
                old_hashes.discard(remote)
                if not old_hashes:
                    del self.by_group[event.group_idx][existing.local_hash]
            else:
                self.references[remote] += 1
            self.entries[entry_key] = endpoint
            self.by_group[event.group_idx].setdefault(endpoint.local_hash, set()).add(remote)
        return True

    def _remove(self, group: int, domain: tuple, hashes: list[str]) -> None:
        removed = set(hashes)
        for remote in removed:
            self.retained_hashes.discard((group, remote))
            endpoint = self.entries.pop((group, domain, remote), None)
            if endpoint is None:
                continue
            self.references[remote] -= 1
            if not self.references[remote]:
                del self.references[remote]
            local = self.by_group[group][endpoint.local_hash]
            local.discard(remote)
            if not local:
                del self.by_group[group][endpoint.local_hash]
        # Cancel older unresolved stores so later parent arrivals cannot
        # resurrect an already removed checkpoint.
        pending = []
        for stamp, event in self.pending:
            if event.group_idx == group and self._domain(event) == domain:
                retained = [h for h in event.block_hashes if h not in removed]
                event = replace(event, block_hashes=retained)
            if event.block_hashes:
                pending.append((stamp, event))
        self.pending = pending

    def _retry(self, chains: dict) -> None:
        if self.status == "unsupported":
            return
        self._expire_pending()
        while self.pending:
            old = self.pending
            self.pending = []
            for stamp, event in old:
                if not self._store(event, chains):
                    if self.status == "unsupported":
                        return
                    self.pending.append((stamp, event))
            if len(self.pending) == len(old):
                break

    def _expire_pending(self) -> None:
        # Missing Stored dependencies only hide additional residency; they do
        # not invalidate entries already established by other events.
        now = time.monotonic()
        retained = [(stamp, event) for stamp, event in self.pending if now - stamp <= self.pending_ttl]
        self.diagnostics["pending_expired"] += len(self.pending) - len(retained)
        self.pending = retained

    def _prune_anchors(self) -> None:
        excess = len(self.anchors) - len(self.references) - self.history_limit
        if excess <= 0:
            return
        for remote in list(self.anchors):
            if remote not in self.references:
                del self.anchors[remote]
                excess -= 1
                if excess <= 0:
                    break

    def estimate(self, hashes: list[str], query_tokens: int) -> PrefixEstimate:
        unit = self.registry.local_hash_unit
        limit = max(0, (query_tokens - 1) // unit * unit)
        self._expire_pending()
        status, reason = self.status, self.reason
        if status != "ready" or query_tokens == 0:
            return PrefixEstimate(status if query_tokens else "empty", 0, query_tokens, limit, unit, reason)
        matches = {}
        for group in self.registry.required_groups:
            index = self.by_group.get(group.group_idx, {})
            matches[group.group_idx] = {
                (i + 1) * unit for i, value in enumerate(hashes[: limit // unit]) if value in index
            }
        candidate = min(limit, len(hashes) * unit)
        while candidate:
            for group in self.registry.required_groups:
                hit = _find_hit(group, matches[group.group_idx], candidate, unit)
                if hit < candidate:
                    candidate = hit
                    break
            else:
                break
        replay = self.registry.prefix_replay_tokens
        # vLLM abandons hits no longer than the replay window. Longer hits
        # still count as H cached tokens, while only H - W tokens save prefill.
        if candidate <= replay:
            candidate = 0
        return PrefixEstimate("ready", candidate, query_tokens, limit, unit, replayed_tokens=replay if candidate else 0)


def _find_hit(group: CacheGroup, present: set[int], limit: int, unit: int) -> int:
    """Largest valid endpoint <= limit; residency only contains actual events."""
    block = group.block_size
    if group.match_kind == "mamba":
        return max((p for p in present if p <= limit), default=0)
    if group.match_kind == "full":
        hit = 0
        while hit + block <= limit and hit + block in present:
            hit += block
        # Only explicit partial endpoints can extend the first incomplete block.
        tail = max((p for p in present if hit < p <= min(limit, hit + block - 1)), default=hit)
        return tail
    if group.match_kind == "sliding":
        needed = max(1, (group.sliding_window - 1 + block - 1) // block)
        run = 0
        best = 0
        for end in range(block, limit + 1, block):
            run = run + 1 if end in present else 0
            if end in present and (run >= needed or run == end // block) and end % unit == 0:
                best = end
        return best
    if group.match_kind == "chunked":
        # At a chunk boundary the preceding chunk requires no cached blocks.
        # If a miss moves the candidate into another chunk, re-evaluate there.
        candidate = limit // block * block
        while candidate > 0:
            start = candidate // group.attention_chunk_size * group.attention_chunk_size
            hit = start // block * block
            while hit + block <= candidate and hit + block in present:
                hit += block
            hit = hit // unit * unit
            if hit == candidate:
                return hit
            candidate = hit
        return 0
    raise ValueError(f"Unsupported match kind: {group.match_kind}")
