# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Expert copies (afd_plugin/expert_copies.py) against a plain loop of the
documented rules, on random placements with copies."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from afd_plugin.expert_copies import (  # noqa: E402
    MU,
    CopyRouter,
    mask_other_ranks,
    placement_has_copies,
)

NUM_EXPERTS = 8
LAYERS = (1, 2, 3, 4)


def _placement(num_ranks: int, seed: int):
    """Per layer: every expert on a home rank plus random copies."""
    gen = torch.Generator().manual_seed(seed)
    layers, home = {}, {}
    for layer in LAYERS:
        h = torch.randint(0, num_ranks, (NUM_EXPERTS,), generator=gen)
        h[:num_ranks] = torch.arange(num_ranks)  # no rank left empty
        held = torch.rand(NUM_EXPERTS, num_ranks, generator=gen) < 0.35
        held[torch.arange(NUM_EXPERTS), h] = True
        layers[layer] = [
            held[:, r].nonzero().flatten().tolist() for r in range(num_ranks)
        ]
        home[layer] = h.tolist()
    return layers, home


def _reference(layers, home, num_ranks, ids_per_layer):
    """The rules of the module docstring, one token and pair at a time."""
    pop = [bin(s).count("1") for s in range(1 << num_ranks)]
    full = (1 << num_ranks) - 1
    num_tokens = ids_per_layer[0].shape[0]
    prev = [0] * num_tokens
    out = []
    for layer, ids in zip(LAYERS, ids_per_layer, strict=True):
        mask = [0] * NUM_EXPERTS
        for r, experts in enumerate(layers[layer]):
            for e in experts:
                mask[e] |= 1 << r
        hm = home[layer]
        visit = []
        for t in range(num_tokens):
            row = [int(e) for e in ids[t] if e >= 0]
            best = None
            for s in range(1, full + 1):
                if all(mask[e] & s for e in row):
                    cost = pop[s] + MU * pop[s & ~prev[t] & full]
                    at_home = sum(1 for e in row if (s >> hm[e]) & 1)
                    key = (cost, -at_home, s)
                    best = key if best is None or key < best else best
            visit.append(best[2])
        fixed = [0] * num_ranks
        for t in range(num_tokens):
            for e in ids[t].tolist():
                if e >= 0 and pop[mask[e] & visit[t]] == 1:
                    fixed[(mask[e] & visit[t]).bit_length() - 1] += 1
        rank_of = torch.full(ids.shape, -1, dtype=torch.int64)
        for t in range(num_tokens):
            for i, e in enumerate(ids[t].tolist()):
                if e < 0:
                    continue
                opts = [r for r in range(num_ranks) if (mask[e] & visit[t]) >> r & 1]
                rank_of[t, i] = min(opts, key=lambda r: (fixed[r], r != hm[e], r))
        out.append(rank_of)
        prev = visit
    return out


@pytest.mark.parametrize("num_ranks,top_k", [(2, 2), (3, 2), (4, 3)])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_routing_follows_the_rules(num_ranks, top_k, seed):
    layers, home = _placement(num_ranks, seed)
    assert placement_has_copies(layers)
    gen = torch.Generator().manual_seed(100 + seed)
    stages = []
    for stage in range(2):
        num_tokens = 9 + stage
        ids = [
            torch.stack(
                [
                    torch.randperm(NUM_EXPERTS, generator=gen)[:top_k]
                    for _ in range(num_tokens)
                ]
            )
            for _ in LAYERS
        ]
        ids[1][0, -1] = -1  # a padding id
        stages.append(ids)
    router = CopyRouter(layers, home, NUM_EXPERTS, num_ranks, top_k, 16, 2, "cpu")
    # Two forwards, the stages interleaved per layer as in the pipeline: each
    # stage keeps its own previous ranks and every forward starts afresh.
    for _forward in range(2):
        got = [[], []]
        for pos, layer in enumerate(LAYERS):
            for stage in range(2):
                got[stage].append(router.route(layer, stages[stage][pos], stage))
        for stage in range(2):
            want = _reference(layers, home, num_ranks, stages[stage])
            for g, w in zip(got[stage], want, strict=True):
                assert torch.equal(g, w)


def test_every_pair_goes_to_a_rank_holding_its_expert():
    layers, home = _placement(4, 5)
    router = CopyRouter(layers, home, NUM_EXPERTS, 4, 3, 32, 1, "cpu")
    gen = torch.Generator().manual_seed(7)
    for layer in LAYERS:
        ids = torch.stack(
            [torch.randperm(NUM_EXPERTS, generator=gen)[:3] for _ in range(32)]
        )
        rank_of = router.route(layer, ids, 0)
        for t in range(32):
            for i in range(3):
                assert int(ids[t, i]) in layers[layer][int(rank_of[t, i])]


def test_without_copies_every_pair_goes_home():
    layers = {1: [[0, 2], [1, 3]], 2: [[3], [0, 1, 2]]}
    assert not placement_has_copies(layers)
    router = CopyRouter(layers, None, 4, 2, 2, 4, 1, "cpu")
    ids = torch.tensor([[0, 1], [2, 3], [3, -1]])
    assert router.route(1, ids, 0).tolist() == [[0, 1], [0, 1], [1, -1]]
    assert router.route(2, ids, 0).tolist() == [[1, 1], [1, 0], [0, -1]]


def test_home_must_hold_its_expert():
    with pytest.raises(ValueError, match="home rank"):
        CopyRouter({1: [[0, 1], [1]]}, {1: [1, 0]}, 2, 2, 1, 4, 1, "cpu")


def test_too_many_ranks_are_refused():
    layers = {1: [[e] for e in range(11)]}
    with pytest.raises(ValueError, match="more than"):
        CopyRouter(layers, None, 11, 11, 1, 4, 1, "cpu")


def test_mask_replaces_other_ranks_experts_by_an_absent_one():
    # Not -1: the experts kernel would sum the stale slot of a -1 id in a real
    # row; an expert the rank does not hold is written as zeros.
    ids = torch.tensor([[3, 5, 6], [3, 5, 6], [1, 2, -1]], dtype=torch.int32)
    rank_of = torch.tensor([[0, 1, 0], [0, 1, 0], [2, 2, -1]])
    dest = torch.tensor([0, 1, 2])
    absent = torch.tensor([7, 0, 4])
    mask_other_ranks(ids, rank_of, dest, absent)
    assert ids.tolist() == [[3, 7, 6], [0, 5, 0], [1, 2, -1]]
    assert ids.dtype == torch.int32


def test_absent_experts_are_not_held():
    layers, home = _placement(4, 9)
    router = CopyRouter(layers, home, NUM_EXPERTS, 4, 2, 4, 1, "cpu")
    for layer in LAYERS:
        for rank, e in enumerate(router.absent(layer).tolist()):
            assert e not in layers[layer][rank]
    with pytest.raises(ValueError, match="holds every expert"):
        CopyRouter({1: [[0, 1, 2, 3], [1]]}, None, 4, 2, 1, 4, 1, "cpu")


def test_home_rank_out_of_range_is_refused():
    with pytest.raises(ValueError, match="home rank"):
        CopyRouter({1: [[0, 1], [1]]}, {1: [-1, 0]}, 2, 2, 1, 4, 1, "cpu")
