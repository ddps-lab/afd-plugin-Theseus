# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Compile every specialization of the delta dispatch row kernels for an L4
(sm_89) without a GPU.

The Triton interpreter (TRITON_INTERPRET=1) runs the kernels on the CPU but
does not type-check them like the GPU compiler: a name bound to a scalar in
one branch and to a block in another compiled in the interpreter and failed
on the GPU only. This test runs the real compiler for a fixed target, so it
needs Triton's NVIDIA backend but no device.
"""

from __future__ import annotations

import itertools

import pytest

triton = pytest.importorskip("triton")
compiler = pytest.importorskip("triton.compiler")
backends = pytest.importorskip("triton.backends.compiler")

from afd_plugin import delta_dispatch_triton as kernels  # noqa: E402
from afd_plugin.delta_dispatch import (  # noqa: E402
    CODE_MASK,
    CODE_SHIFT,
    QUANT_GROUP_SIZE,
    RANK_SHIFT,
    TOKEN_MASK,
)

TARGET = backends.GPUTarget("cuda", 89, 32)
HIDDEN = 2048


def _compile(fn, signature: dict[str, str], constexprs: dict[str, object]) -> None:
    names = fn.arg_names
    src = compiler.ASTSource(
        fn=fn,
        signature=signature,
        constexprs={(names.index(k),): v for k, v in constexprs.items()},
    )
    try:
        triton.compile(src, target=TARGET)
    except RuntimeError as e:  # no NVIDIA backend (ptxas) in this environment
        if "ptxas" in str(e) or "backend" in str(e):
            pytest.skip(f"no Triton NVIDIA backend: {e}")
        raise


@pytest.mark.parametrize(
    ("stateful", "fixed", "from_meta"), list(itertools.product([True, False], repeat=3))
)
def test_decode_rows_kernel_compiles(
    stateful: bool, fixed: bool, from_meta: bool
) -> None:
    signature = {
        "data_ptr": "*u8",
        "meta_ptr": "*i32",
        "off_ptr": "*i32" if fixed else "*i64",
        "rows_ptr": "*i32" if from_meta else "*i64",
        "state_ptr": "*fp32" if stateful else "*bf16",
        "state_stride": "i32",
        "out_ptr": "*bf16",
        "out_stride": "i32",
        "row_size": "i32",
        "max_tokens": "i32",
        **{
            k: "constexpr"
            for k in (
                "hidden",
                "group",
                "stateful",
                "code_shift",
                "code_mask",
                "rank_shift",
                "token_mask",
                "fixed",
                "rows_from_meta",
            )
        },
    }
    _compile(
        kernels._decode_rows_kernel,
        signature,
        {
            "hidden": HIDDEN,
            "group": QUANT_GROUP_SIZE,
            "stateful": stateful,
            "code_shift": CODE_SHIFT,
            "code_mask": CODE_MASK,
            "rank_shift": RANK_SHIFT,
            "token_mask": TOKEN_MASK,
            "fixed": fixed,
            "rows_from_meta": from_meta,
        },
    )


@pytest.mark.parametrize(
    ("stateful", "in_kernel"), list(itertools.product([True, False], repeat=2))
)
def test_encode_rows_kernel_compiles(stateful: bool, in_kernel: bool) -> None:
    signature = {
        "src_ptr": "*bf16",
        "src_stride": "i32",
        "state_ptr": "*fp32" if stateful else "*bf16",
        "state_stride": "i32",
        "last_ptr": "*i32" if stateful else "*i64",
        "flat_ptr": "*i64",
        "code_ptr": "*i64" if in_kernel else "*i32",
        "off_ptr": "*i64",
        "out_ptr": "*u8",
        "meta_ptr": "*i32",
        "num_tokens": "i32",
        "rank_word": "i32",
        "layer_pos": "i32",
        "min_range": "fp32",
        "first_code": "i32",
        "rest_code": "i32",
        **{
            k: "constexpr"
            for k in ("hidden", "group", "stateful", "code_shift", "codes_in_kernel")
        },
    }
    _compile(
        kernels._encode_rows_kernel,
        signature,
        {
            "hidden": HIDDEN,
            "group": QUANT_GROUP_SIZE,
            "stateful": stateful,
            "code_shift": CODE_SHIFT,
            "codes_in_kernel": in_kernel,
        },
    )
