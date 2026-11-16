# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""NIXL routed connector, AFD_NIXL_SERIAL_WRITES: at most one WRITE in flight
per peer, the rest posted in the order they were issued once it is done."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("nixl")

from afd_plugin.connectors.gpu import nixl_routed  # noqa: E402
from afd_plugin.connectors.gpu.nixl_routed import (  # noqa: E402
    XFER_DONE,
    XFER_IN_FLIGHT,
    XFER_QUEUED,
    NixlRoutedAFDConnector,
    QueuedWrite,
)

FFN_0 = 8
FFN_1 = 9


class FakeHandle:
    def __init__(self, peer: str, notif: bytes) -> None:
        self.peer = peer
        self.notif = notif
        self.state = XFER_IN_FLIGHT


class FakeAgent:
    """Records the WRITEs in the order they are posted."""

    def __init__(self) -> None:
        self.posted: list[FakeHandle] = []
        self.released: list[FakeHandle] = []
        self.done_at_post = False

    def get_xfer_descs(self, descs, _mem_type):
        return descs

    def initialize_xfer(self, _op, _src, _dst, peer, notif):
        return FakeHandle(peer, notif)

    def transfer(self, handle):
        self.posted.append(handle)
        if self.done_at_post:
            handle.state = XFER_DONE
        return handle.state

    def check_xfer_state(self, handle):
        return handle.state

    def release_xfer_handle(self, handle):
        self.released.append(handle)


def _connector(serial: bool, monkeypatch) -> tuple[NixlRoutedAFDConnector, FakeAgent]:
    monkeypatch.setattr(nixl_routed, "_SERIAL_WRITES", serial)
    connector = object.__new__(NixlRoutedAFDConnector)
    agent = FakeAgent()
    connector.agent = agent
    connector._writes = {}
    connector._active_write = {}
    connector._queued_writes = {}
    connector._pending_combines = []
    connector._peer_agents = {FFN_0: "ffn-0", FFN_1: "ffn-1"}
    return connector, agent


def _write(slot: int, peer: int) -> QueuedWrite:
    return QueuedWrite(slot, peer, [(0, 16, 0)], [(64, 16, 0)], b"d%d" % slot)


def _posted(agent: FakeAgent) -> list[tuple[str, bytes]]:
    return [(handle.peer, handle.notif) for handle in agent.posted]


def test_without_the_setting_every_write_is_posted_at_once(monkeypatch):
    connector, agent = _connector(False, monkeypatch)
    states = [connector._post_write(_write(slot, FFN_0)) for slot in range(3)]
    assert states == [XFER_IN_FLIGHT] * 3
    assert _posted(agent) == [("ffn-0", b"d0"), ("ffn-0", b"d1"), ("ffn-0", b"d2")]
    assert not connector._active_write
    assert not connector._queued_writes


def test_one_write_in_flight_per_peer_in_issue_order(monkeypatch):
    connector, agent = _connector(True, monkeypatch)
    assert connector._post_write(_write(0, FFN_0)) == XFER_IN_FLIGHT
    assert connector._post_write(_write(1, FFN_0)) == XFER_QUEUED
    assert connector._post_write(_write(2, FFN_0)) == XFER_QUEUED
    # Another peer is not held back by the WRITE in flight to the first.
    assert connector._post_write(_write(0, FFN_1)) == XFER_IN_FLIGHT
    assert _posted(agent) == [("ffn-0", b"d0"), ("ffn-1", b"d0")]
    # A queued WRITE keeps its slot pending, as a posted one does.
    assert set(connector._writes) == {0, 1, 2}

    connector._service_writes()
    assert _posted(agent) == [("ffn-0", b"d0"), ("ffn-1", b"d0")]

    agent.posted[0].state = XFER_DONE
    connector._service_writes()
    assert _posted(agent)[2:] == [("ffn-0", b"d1")]
    assert set(connector._writes) == {0, 1, 2}

    agent.posted[1].state = XFER_DONE
    agent.posted[2].state = XFER_DONE
    connector._service_writes()
    assert _posted(agent)[3:] == [("ffn-0", b"d2")]
    assert set(connector._writes) == {2}

    agent.posted[3].state = XFER_DONE
    connector._service_writes()
    assert not connector._writes
    assert not connector._active_write
    assert len(agent.released) == 4


def test_a_write_done_at_post_holds_nothing_back(monkeypatch):
    connector, agent = _connector(True, monkeypatch)
    agent.done_at_post = True
    assert connector._post_write(_write(0, FFN_0)) == XFER_DONE
    assert connector._post_write(_write(1, FFN_0)) == XFER_DONE
    assert _posted(agent) == [("ffn-0", b"d0"), ("ffn-0", b"d1")]
    assert not connector._active_write
    connector._service_writes()
    assert not connector._writes
    assert len(agent.released) == 2


def test_queued_writes_that_complete_at_post_are_all_posted(monkeypatch):
    connector, agent = _connector(True, monkeypatch)
    assert connector._post_write(_write(0, FFN_0)) == XFER_IN_FLIGHT
    assert connector._post_write(_write(1, FFN_0)) == XFER_QUEUED
    assert connector._post_write(_write(2, FFN_0)) == XFER_QUEUED
    agent.done_at_post = True
    agent.posted[0].state = XFER_DONE
    connector._service_writes()
    assert _posted(agent) == [("ffn-0", b"d0"), ("ffn-0", b"d1"), ("ffn-0", b"d2")]
    assert not connector._active_write
    assert not connector._queued_writes[FFN_0]
    connector._service_writes()
    assert not connector._writes
    assert len(agent.released) == 3
