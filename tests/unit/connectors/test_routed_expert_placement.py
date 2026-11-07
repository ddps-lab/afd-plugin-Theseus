# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Routed connector with a per-layer expert placement (--expert-placement-file):
the owner tables, the cross-rank digest, the config check and the plan."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm.distributed.device_communicators.p2p_nccl_all2all import (  # noqa: E402
    P2pAll2AllEngine,
)

from afd_plugin.connectors.gpu import routed  # noqa: E402
from afd_plugin.expert_copies import CopyRouter  # noqa: E402

PLACEMENT = {
    1: [[3, 0], [1, 2]],
    2: [[2], [0, 3, 1]],
}


def test_owner_tables_give_each_expert_its_ffn_rank():
    tables = routed.expert_owner_tables(PLACEMENT, 4, "cpu")
    assert tables[1].tolist() == [0, 1, 1, 0]
    assert tables[2].tolist() == [1, 1, 0, 1]
    assert tables[1].dtype == torch.int64


@pytest.mark.parametrize(
    "layers",
    [
        {1: [[0, 1], [1, 2, 3]]},  # expert 1 on two ranks (a copy)
        {1: [[0, 1], [2]]},  # expert 3 on no rank
    ],
)
def test_owner_tables_reject_copies_and_missing_experts(layers):
    with pytest.raises(RuntimeError):
        routed.expert_owner_tables(layers, 4, "cpu")


def test_placement_digest_follows_the_content_only():
    same = {2: [[2], [0, 3, 1]], 1: [[3, 0], [1, 2]]}
    other = {1: [[0, 3], [1, 2]], 2: [[2], [0, 3, 1]]}  # slot order differs
    digest = routed.expert_placement_digest(PLACEMENT)
    assert 0 < digest < 2**63
    assert routed.expert_placement_digest(same) == digest
    assert routed.expert_placement_digest(other) != digest


def test_settings_digest_tells_configs_apart():
    from afd_plugin.delta_dispatch import CombineConfig, DeltaDispatchConfig

    base = routed.wire_settings_digest(DeltaDispatchConfig(), CombineConfig())
    assert 0 < base < 2**63
    assert base == routed.wire_settings_digest(DeltaDispatchConfig(), CombineConfig())
    others = {
        routed.wire_settings_digest(
            DeltaDispatchConfig(mode="delta_int4"), CombineConfig()
        ),
        routed.wire_settings_digest(
            DeltaDispatchConfig(mode="delta_int4"), CombineConfig(mode="int4")
        ),
        routed.wire_settings_digest(
            DeltaDispatchConfig(mode="delta_int4"), CombineConfig(mode="int4", ef=False)
        ),
        routed.wire_settings_digest(
            DeltaDispatchConfig(mode="delta_int4", late=1), CombineConfig()
        ),
        routed.wire_settings_digest(
            DeltaDispatchConfig(mode="delta_int4", first="int4"), CombineConfig()
        ),
        routed.wire_settings_digest(
            DeltaDispatchConfig(mode="delta_int3"), CombineConfig()
        ),
        routed.wire_settings_digest(
            DeltaDispatchConfig(mode="delta_int3", first="int4"), CombineConfig()
        ),
        routed.wire_settings_digest(
            DeltaDispatchConfig(mode="delta_int3", first="int4", revisit="int4"),
            CombineConfig(),
        ),
        routed.wire_settings_digest(
            DeltaDispatchConfig(mode="x_int8"), CombineConfig()
        ),
        routed.wire_settings_digest(
            DeltaDispatchConfig(mode="delta_int8"), CombineConfig()
        ),
        routed.wire_settings_digest(
            DeltaDispatchConfig(mode="delta_int8", first="int8"), CombineConfig()
        ),
        routed.wire_settings_digest(
            DeltaDispatchConfig(mode="delta_int4", first="int8"), CombineConfig()
        ),
        routed.wire_settings_digest(
            DeltaDispatchConfig(mode="delta_int4"),
            CombineConfig(mode="int4", early=4, late=4),
        ),
        routed.wire_settings_digest(
            DeltaDispatchConfig(mode="delta_int4"), CombineConfig(mode="int4", early=4)
        ),
    }
    assert base not in others and len(others) == 14
    # Naming the mode's own format is the same setting.
    for mode, own in (("delta_int4", "int4"), ("delta_int3", "int3")):
        named = DeltaDispatchConfig.from_mapping({"mode": mode, "revisit": own})
        plain = DeltaDispatchConfig.from_mapping({"mode": mode})
        digests = {
            routed.wire_settings_digest(c, CombineConfig()) for c in (named, plain)
        }
        assert len(digests) == 1


def _plan_owner(owners, router=None):
    conn = routed.P2pNcclRoutedAFDConnector.__new__(routed.P2pNcclRoutedAFDConnector)
    # Two FFN ranks and one Attention rank (world rank 2), four experts.
    conn.engine = P2pAll2AllEngine(None, 2, 3, 2, 4, "cpu")
    conn._layer_owner, conn._copy_router, conn._copy_rank_of = owners, router, {}
    placed = owners is not None or router is not None
    conn.expert_placement_file = "placement.json" if placed else None
    return conn


def test_plan_routes_by_the_layer_table():
    conn = _plan_owner(routed.expert_owner_tables(PLACEMENT, 4, "cpu"))
    topk_ids = torch.tensor([[0, 1], [3, -1], [2, 3]])
    conn._begin_plan(topk_ids, 0, 1)
    # Layer 1: experts 0, 3 on rank 0 and 1, 2 on rank 1.
    assert conn.engine._count_buf(0)[0].tolist() == [3, 2, 0]
    conn._begin_plan(topk_ids, 0, 2)
    # Layer 2: expert 2 on rank 0, the others on rank 1.
    assert conn.engine._count_buf(0)[0].tolist() == [1, 3, 0]
    with pytest.raises(RuntimeError, match="no MoE layer 5"):
        conn._begin_plan(topk_ids, 0, 5)


def test_plan_by_pair_ranks_matches_the_engine_table_path():
    """_begin_plan hands the engine per-pair ranks (a table with -1 at the
    end, one gather); the plan must be the engine's own expert_owner plan,
    padding ids included."""
    owners = routed.expert_owner_tables(PLACEMENT, 4, "cpu")
    conn = _plan_owner(owners)
    topk_ids = torch.tensor([[0, 1], [3, -1], [-1, -1], [2, 3], [1, 0]])
    for layer in (1, 2):
        pending = conn._begin_plan(topk_ids, 0, layer)
        got = conn.engine._count_buf(0)[0].tolist()
        engine = P2pAll2AllEngine(None, 2, 3, 2, 4, "cpu")
        expected = engine.begin_plan(topk_ids, expert_owner=owners[layer])
        assert torch.equal(pending.order, expected.order)
        assert got == engine._count_buf(0)[0].tolist()


def test_plan_without_a_placement_routes_like_the_engine():
    conn = _plan_owner(None)
    topk_ids = torch.tensor([[0, 1], [3, -1], [2, 3]])
    pending = conn._begin_plan(topk_ids, 0, None)
    expected = P2pAll2AllEngine(None, 2, 3, 2, 4, "cpu").begin_plan(topk_ids)
    assert torch.equal(pending.order, expected.order)
    assert conn.engine._count_buf(0)[0].tolist() == [1, 2, 0]


def _runtime_config(placement_file):
    parallel = SimpleNamespace(
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        expert_placement_counts=None,
        expert_placement_file=placement_file,
    )
    return SimpleNamespace(
        parallel_config=parallel, model_config=SimpleNamespace(enforce_eager=True)
    )


@pytest.mark.parametrize(
    ("num_ranks", "num_experts", "ok"), [(2, 4, True), (3, 4, False), (2, 6, False)]
)
def test_runtime_check_matches_the_file_to_the_deployment(
    monkeypatch, num_ranks, num_experts, ok
):
    monkeypatch.setattr(
        routed,
        "load_expert_placement",
        lambda path: SimpleNamespace(num_ranks=num_ranks, num_experts=num_experts),
    )
    afd = SimpleNamespace(num_ffn_ranks=2, role="attention")
    spec = SimpleNamespace(num_experts=4)
    if ok:
        routed._validate_routed_runtime(_runtime_config("p.json"), afd, spec)
    else:
        with pytest.raises(RuntimeError, match="expert placement file"):
            routed._validate_routed_runtime(_runtime_config("p.json"), afd, spec)


def test_runtime_check_accepts_indivisible_experts_with_a_file(monkeypatch):
    monkeypatch.setattr(
        routed,
        "load_expert_placement",
        lambda path: SimpleNamespace(num_ranks=2, num_experts=5),
    )
    afd = SimpleNamespace(num_ffn_ranks=2, role="attention")
    spec = SimpleNamespace(num_experts=5)
    routed._validate_routed_runtime(_runtime_config("p.json"), afd, spec)
    with pytest.raises(RuntimeError, match="divisible"):
        routed._validate_routed_runtime(_runtime_config(None), afd, spec)


def test_plan_with_copies_sends_each_pair_to_its_copy():
    # Layer 1: expert 1 on both ranks (home 1), the others on one rank.
    layers = {1: [[0, 1, 3], [1, 2]]}
    router = CopyRouter(layers, {1: [0, 1, 1, 0]}, 4, 2, 2, 4, 1, "cpu")
    conn = _plan_owner(None, router)
    topk_ids = torch.tensor([[0, 1], [1, 2], [3, -1]])
    pending = conn._begin_plan(topk_ids, 0, 1)
    # Token 0 visits rank 0 only (expert 1's copy there), token 1 rank 1 only.
    assert conn.engine._count_buf(0)[0].tolist() == [2, 1, 0]
    plan = conn.engine.finish_plan_local(pending)
    rows = topk_ids[plan.perm]
    conn._mask_copied_ids(rows, 0, plan, pending)
    assert rows.tolist() == [[0, 1], [3, -1], [1, 2]]
    assert conn._copy_rank_of == {}
    # Without a routed plan the rows keep all their ids.
    rows2 = topk_ids[plan.perm]
    conn._mask_copied_ids(rows2, 0, plan, pending)
    assert torch.equal(rows2, topk_ids[plan.perm])


def test_rows_leave_out_the_copies_another_rank_computes():
    # Expert 1 on both ranks (home 1); the token also needs 0 (rank 0 only)
    # and 2 (rank 1 only), so it visits both and expert 1 goes home. Each
    # row's other experts become an expert its rank does not hold (rank 0:
    # 2, rank 1: 0), never -1 in a real row.
    layers = {1: [[0, 1, 3], [1, 2]]}
    router = CopyRouter(layers, {1: [0, 1, 1, 0]}, 4, 2, 3, 4, 1, "cpu")
    conn = _plan_owner(None, router)
    topk_ids = torch.tensor([[0, 1, 2]])
    pending = conn._begin_plan(topk_ids, 0, 1)
    plan = conn.engine.finish_plan_local(pending)
    rows = topk_ids[plan.perm]
    conn._mask_copied_ids(rows, 0, plan, pending)
    assert rows.tolist() == [[0, 2, 2], [0, 1, 2]]
