# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Expert copies in the routed dispatch.

An expert placement file (``--expert-placement-file``) may list an expert on
several FFN ranks (a copy in the FFN ranks' spare memory). Per MoE layer, the
Attention side then decides for every token which FFN ranks it visits and,
for every selected expert, which rank's copy computes it:

1. Ranks: among the sets of FFN ranks holding all the token's experts, the one
   with the fewest ranks plus ``MU`` times the ranks the token did not visit
   at the previous MoE layer of this forward (fewer rows, fewer rows to a rank
   whose state is stale); ties go to the set holding more of the experts on
   their home rank, then to the smaller bit mask.
2. Copy: an expert held by one rank of that set goes there; one held by
   several goes to the rank with the fewest single-choice (token, expert)
   pairs of this stage, then to its home rank, then to the lower rank. All
   pairs choose at once (a few tensor operations). On recorded routings its
   max/mean expert load is 0.01-0.04 above that of the per-pair sequential
   choice it approximates (Plan 16 experiment 6, step 4-3).

The dispatch plan takes these (token, expert) destinations, and every FFN rank
receives the token's row with the experts other ranks compute replaced by an
expert it does not hold at that layer, so a copied expert is computed once.
The replacement goes the way of any other rank's expert in a row: the rank's
expert map turns it into a non-local expert, whose slots the experts kernel
writes as zeros. It is not -1: vLLM's Triton experts kernel leaves the slots
of -1 ids out of its GEMMs but still sums them (stale values), which is safe
only for the fixed-size launch's padding rows, all -1 and never summed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

# Weight of a rank not visited at the previous MoE layer (Plan 16 mu = 1).
MU = 1
# Every set of FFN ranks is scored, 2**F - 1 of them ([tokens, top_k, 2**F - 1]
# temporaries per layer).
MAX_COPY_RANKS = 8


def placement_has_copies(layers: Mapping[int, Sequence[Sequence[int]]]) -> bool:
    for ranks in layers.values():
        listed = [e for experts in ranks for e in experts]
        if len(listed) != len(set(listed)):
            return True
    return False


class CopyRouter:
    """Attention side of a placement with copies: per (token, expert) the FFN
    rank that computes it, and per token the ranks visited at the previous MoE
    layer, per micro-batch stage."""

    def __init__(
        self,
        layers: Mapping[int, Sequence[Sequence[int]]],
        home: Mapping[int, Sequence[int]] | None,
        num_experts: int,
        num_ranks: int,
        top_k: int,
        max_tokens: int,
        num_slots: int,
        device: torch.device | str,
    ) -> None:
        if num_ranks > MAX_COPY_RANKS:
            raise ValueError(
                f"expert copies score every set of FFN ranks; {num_ranks} ranks "
                f"is more than {MAX_COPY_RANKS}"
            )
        self.num_ranks, self.top_k = num_ranks, top_k
        self._mask: dict[int, torch.Tensor] = {}
        self._home: dict[int, torch.Tensor] = {}
        # Per layer and rank, an expert the rank does not hold (the id that
        # replaces the experts other ranks compute in its rows).
        self._absent: dict[int, torch.Tensor] = {}
        for layer, ranks in layers.items():
            mask = torch.zeros(num_experts, dtype=torch.int64)
            for rank, experts in enumerate(ranks):
                idx = torch.as_tensor(list(experts), dtype=torch.int64)
                mask[idx] |= 1 << rank
            if bool((mask == 0).any()):
                raise ValueError(f"expert placement layer {layer}: experts on no rank")
            if home is not None and int(layer) in home:
                home_rank = torch.as_tensor(list(home[int(layer)]), dtype=torch.int64)
                if (
                    home_rank.shape != (num_experts,)
                    or bool(((home_rank < 0) | (home_rank >= num_ranks)).any())
                    or bool((((mask >> home_rank.clamp(0, 62)) & 1) == 0).any())
                ):
                    raise ValueError(
                        f"expert placement layer {layer}: a home rank does not "
                        "hold its expert"
                    )
            else:
                # Lowest rank holding the expert.
                home_rank = _lowest_bit(mask)
            absent = []
            for rank in range(num_ranks):
                missing = (((mask >> rank) & 1) == 0).nonzero().flatten()
                if missing.numel() == 0:
                    raise ValueError(
                        f"expert placement layer {layer}: rank {rank} holds every "
                        "expert, so the experts other ranks compute cannot be left "
                        "out of its rows"
                    )
                absent.append(int(missing[0]))
            self._mask[int(layer)] = mask.to(device)
            self._home[int(layer)] = home_rank.to(device)
            self._absent[int(layer)] = torch.tensor(
                absent, dtype=torch.int64, device=device
            )
        self.first_layer = min(self._mask)
        count = 1 << num_ranks
        self._sets = torch.arange(1, count, dtype=torch.int64, device=device)
        self._full = count - 1
        self._pop = torch.tensor(
            [bin(s).count("1") for s in range(count)], dtype=torch.int64, device=device
        )
        self._low = _lowest_bit(torch.arange(count, dtype=torch.int64)).to(device)
        self._ranks = torch.arange(num_ranks, dtype=torch.int64, device=device)
        self._prev = torch.zeros(
            num_slots, max_tokens, dtype=torch.int64, device=device
        )

    def absent(self, layer_idx: int) -> torch.Tensor:
        """Per FFN rank, an expert it does not hold at ``layer_idx`` (int64)."""
        return self._absent[int(layer_idx)]

    def route(self, layer_idx: int, topk_ids: torch.Tensor, stage: int) -> torch.Tensor:
        """``[tokens, top_k]`` FFN rank of every (token, expert) pair, -1 for
        padding ids; remembers the ranks visited for the next MoE layer."""
        mask_e = self._mask.get(int(layer_idx))
        if mask_e is None:
            raise RuntimeError(f"expert placement has no MoE layer {layer_idx}")
        ids = topk_ids.long()
        num_tokens, k = ids.shape
        valid = ids >= 0
        safe = ids.clamp(min=0)
        mask = mask_e[safe]
        home = self._home[int(layer_idx)][safe]
        prev = self._prev[stage, :num_tokens]
        if int(layer_idx) == self.first_layer:
            prev.zero_()
        sets = self._sets
        # 1. The set of ranks every token visits.
        held = ((mask.unsqueeze(-1) & sets) != 0) | ~valid.unsqueeze(-1)
        cover = held.all(dim=1)
        new = self._pop[sets & (~prev.unsqueeze(-1) & self._full)]
        cost = self._pop[sets] + MU * new
        at_home = (((1 << home).unsqueeze(-1) & sets) != 0) & valid.unsqueeze(-1)
        key = (cost * (k + 1) + (k - at_home.sum(dim=1))) * (self._full + 1) + sets
        key = key.masked_fill(~cover, torch.iinfo(torch.int64).max)
        visit = sets[key.argmin(dim=1)]
        # 2. The rank whose copy computes each pair.
        opts = mask & visit.unsqueeze(1)
        single = (self._pop[opts] == 1) & valid
        first = torch.where(((opts >> home) & 1) == 1, home, self._low[opts])
        fixed = torch.zeros(self.num_ranks, dtype=torch.int64, device=ids.device)
        # Padding pairs (no rank, -1) add nothing.
        fixed.index_add_(0, first.clamp(min=0).reshape(-1), single.reshape(-1).long())
        ranks = self._ranks
        rkey = (fixed * 2 + (ranks != home.unsqueeze(-1))) * self.num_ranks + ranks
        rkey = rkey.masked_fill(((opts.unsqueeze(-1) >> ranks) & 1) == 0, 1 << 62)
        rank_of = torch.where(single, first, rkey.argmin(dim=-1))
        rank_of = rank_of.masked_fill(~valid, -1)
        prev.copy_(visit)
        return rank_of


def mask_other_ranks(
    rows_ids: torch.Tensor,
    rows_rank_of: torch.Tensor,
    rows_dest: torch.Tensor,
    absent: torch.Tensor,
) -> torch.Tensor:
    """Replace, in place, the packed rows' expert ids that other ranks compute
    by an expert the row's rank does not hold (``absent[rank]``): ``rows_*``
    are the plan's packed rows (``[n, top_k]``), ``rows_dest`` the FFN rank
    each row goes to (``[n]``). Padding ids (< 0) stay. Returns ``rows_ids``."""
    other = (rows_rank_of != rows_dest.unsqueeze(1)) & (rows_ids >= 0)
    fill = absent[rows_dest].to(rows_ids.dtype).unsqueeze(1).expand_as(rows_ids)
    return rows_ids.copy_(torch.where(other, fill, rows_ids))


def _lowest_bit(mask: torch.Tensor) -> torch.Tensor:
    """Index of the lowest set bit (-1 for 0)."""
    low = mask & -mask
    out = torch.full_like(mask, -1)
    nz = low > 0
    out[nz] = torch.log2(low[nz].double()).round().long()
    return out
