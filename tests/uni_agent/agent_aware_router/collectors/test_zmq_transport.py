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

"""KV event replay and lifecycle notification protocols."""

import asyncio

import msgpack
import pytest
import zmq
import zmq.asyncio

from uni_agent.agent_aware_router.collectors.transport.lifecycle import LifecycleControlReceiver, notify_lifecycle
from uni_agent.agent_aware_router.collectors.transport.zmq import ZMQTransport
from uni_agent.agent_aware_router.types.cache_lifecycle import CacheLifecycleState

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


def test_dealer_reads_multiple_batches_and_end_marker_and_resumes_at_cursor():
    async def run():
        ctx = zmq.asyncio.Context()
        pub, replay = ctx.socket(zmq.PUB), ctx.socket(zmq.ROUTER)
        pub_port = pub.bind_to_random_port("tcp://127.0.0.1")
        replay_port = replay.bind_to_random_port("tcp://127.0.0.1")
        transport = ZMQTransport(
            {"a": [f"127.0.0.1:{pub_port}", f"127.0.0.1:{replay_port}", "zmq", "kv"]}, deliver_frames=True
        )
        received = []
        await transport._connect_zmq_for("a", f"tcp://127.0.0.1:{pub_port}", f"tcp://127.0.0.1:{replay_port}")

        async def serve(expected_start, sequences):
            identity, delimiter, start = await replay.recv_multipart()
            assert delimiter == b"" and int.from_bytes(start, "big") == expected_start
            for sequence in sequences:
                await replay.send_multipart(
                    [identity, b"", b"kv", sequence.to_bytes(8, "big"), msgpack.packb([sequence, []])]
                )
            await replay.send_multipart([identity, b"", b"", (-1).to_bytes(8, "big", signed=True), b""])

        try:
            worker = asyncio.create_task(serve(0, [0, 1, 2]))
            assert await asyncio.wait_for(
                transport._replay_historical_data_for("a", lambda frame, node: received.append(frame)), 2
            )
            await worker
            assert [frame.sequence for frame in received] == [0, 1, 2]
            worker = asyncio.create_task(serve(3, [2, 3]))
            assert await asyncio.wait_for(
                transport._replay_historical_data_for("a", lambda frame, node: received.append(frame)), 2
            )
            await worker
            assert [frame.sequence for frame in received] == [0, 1, 2, 3]
            assert not any(frame.error for frame in received)
        finally:
            transport.stop()
            pub.close(0)
            replay.close(0)
            ctx.term()

    asyncio.run(run())


def test_missing_sequence_and_duplicate_are_not_silently_applied():
    transport = ZMQTransport({}, deliver_frames=True)
    frames = []
    handler = lambda frame, node: frames.append(frame)
    transport._deliver("a", 0, b"first", handler)
    transport._deliver("a", 0, b"duplicate", handler)
    transport._deliver("a", 2, b"gap", handler)
    assert [frame.payload for frame in frames] == [b"first", b"", b"gap"]
    assert frames[1].error == "kv_sequence_gap:1:2"


def test_control_ack_and_capability_check():
    applied = []
    receiver = LifecycleControlReceiver(
        lambda sid, state: applied.append((sid, state)) or {"applied": True}, "127.0.0.1"
    )

    async def run():
        sink = receiver.descriptor("a")
        assert (await notify_lifecycle(sink, CacheLifecycleState("engine").to_dict()))["applied"]
        with pytest.raises(RuntimeError, match="capability"):
            await notify_lifecycle({**sink, "token": "wrong"}, CacheLifecycleState("engine").to_dict())

    try:
        asyncio.run(run())
        assert len(applied) == 1 and applied[0][0] == "a"
    finally:
        receiver.close()
