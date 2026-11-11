# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Compile every specialization of the delta dispatch kernels (rows and
combine, with and without NVFP4 rows, software and hardware E4M3) for an L4
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
    NVFP4_BLOCK,
    QUANT_GROUP_SIZE,
    RANK_SHIFT,
    TOKEN_MASK,
)

TARGET = backends.GPUTarget("cuda", 89, 32)
HIDDEN = 2048
BLOCKS_P2 = HIDDEN // NVFP4_BLOCK
# (nvfp4, hardware): no NVFP4 rows; NVFP4 with software / hardware E4M3.
NVFP4_VARIANTS = [(False, False), (True, False), (True, True)]
# The PTX conversion of two fp32 to E4M3 (round to nearest even, saturating),
# and the fp16 rounding Triton's own cast would put before it on sm_89.
E4M3_CVT = "cvt.rn.satfinite.e4m3x2.f32"
F16_TRUNCATION = "cvt.rz.f16.f32"


def _compile(fn, signature: dict[str, str], constexprs: dict[str, object]) -> str:
    """Compile for TARGET and return the PTX."""
    names = fn.arg_names
    assert sorted(signature) == sorted(names), set(names) ^ set(signature)
    src = compiler.ASTSource(
        fn=fn,
        signature=signature,
        constexprs={(names.index(k),): v for k, v in constexprs.items()},
    )
    try:
        return triton.compile(src, target=TARGET).asm["ptx"]
    except RuntimeError as e:  # no NVIDIA backend (ptxas) in this environment
        if "ptxas" in str(e) or "backend" in str(e):
            pytest.skip(f"no Triton NVIDIA backend: {e}")
        raise


def _constexprs(names: tuple[str, ...]) -> dict[str, str]:
    return {k: "constexpr" for k in names}


@pytest.mark.parametrize(
    ("stateful", "fixed", "from_meta"), list(itertools.product([True, False], repeat=3))
)
@pytest.mark.parametrize(("nvfp4", "hardware"), NVFP4_VARIANTS)
def test_decode_rows_kernel_compiles(
    stateful: bool, fixed: bool, from_meta: bool, nvfp4: bool, hardware: bool
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
                "nvfp4",
                "hardware",
            )
        },
    }
    ptx = _compile(
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
            "nvfp4": nvfp4,
            "hardware": hardware,
        },
    )
    # E4M3 to fp32 is exact: the hardware path converts with cvt, the
    # software one with bit operations.
    assert ("cvt.rn.f16x2.e4m3x2" in ptx) == hardware


@pytest.mark.parametrize(
    ("stateful", "in_kernel"), list(itertools.product([True, False], repeat=2))
)
@pytest.mark.parametrize(("nvfp4", "hardware"), NVFP4_VARIANTS)
def test_encode_rows_kernel_compiles(
    stateful: bool, in_kernel: bool, nvfp4: bool, hardware: bool
) -> None:
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
        "amax_ptr": "*fp32" if nvfp4 else "*i32",
        "num_tokens": "i32",
        "rank_word": "i32",
        "layer_pos": "i32",
        "min_range": "fp32",
        "first_code": "i32",
        "rest_code": "i32",
        **_constexprs(
            (
                "hidden",
                "group",
                "stateful",
                "code_shift",
                "codes_in_kernel",
                "nvfp4",
                "hardware",
            )
        ),
    }
    ptx = _compile(
        kernels._encode_rows_kernel,
        signature,
        {
            "hidden": HIDDEN,
            "group": QUANT_GROUP_SIZE,
            "stateful": stateful,
            "code_shift": CODE_SHIFT,
            "codes_in_kernel": in_kernel,
            "nvfp4": nvfp4,
            "hardware": hardware,
        },
    )
    assert (E4M3_CVT in ptx) == hardware and F16_TRUNCATION not in ptx


def _row_signature(stateful: bool, in_kernel: bool) -> dict[str, str]:
    return {
        "src_ptr": "*bf16",
        "src_stride": "i32",
        "state_ptr": "*fp32" if stateful else "*bf16",
        "state_stride": "i32",
        "last_ptr": "*i32" if stateful else "*i64",
        "flat_ptr": "*i64",
        "code_ptr": "*i64" if in_kernel else "*i32",
        "num_tokens": "i32",
        "first_code": "i32",
        "rest_code": "i32",
    }


@pytest.mark.parametrize(
    ("stateful", "in_kernel"), list(itertools.product([True, False], repeat=2))
)
def test_encode_rows_amax_kernel_compiles(stateful: bool, in_kernel: bool) -> None:
    signature = {
        **_row_signature(stateful, in_kernel),
        "amax_ptr": "*fp32",
        **_constexprs(("hidden", "hidden_p2", "stateful", "codes_in_kernel")),
    }
    _compile(
        kernels._encode_rows_amax_kernel,
        signature,
        {
            "hidden": HIDDEN,
            "hidden_p2": HIDDEN,
            "stateful": stateful,
            "codes_in_kernel": in_kernel,
        },
    )


@pytest.mark.parametrize(
    ("stateful", "in_kernel", "hardware"),
    list(itertools.product([True, False], repeat=3)),
)
def test_encode_rows_row_kernel_compiles(
    stateful: bool, in_kernel: bool, hardware: bool
) -> None:
    signature = {
        **_row_signature(stateful, in_kernel),
        "off_ptr": "*i64",
        "out_ptr": "*u8",
        "meta_ptr": "*i32",
        "rank_word": "i32",
        "layer_pos": "i32",
        "min_range": "fp32",
        **_constexprs(
            (
                "hidden",
                "group",
                "stateful",
                "code_shift",
                "codes_in_kernel",
                "blocks_p2",
                "hardware",
            )
        ),
    }
    ptx = _compile(
        kernels._encode_rows_row_kernel,
        signature,
        {
            "hidden": HIDDEN,
            "group": QUANT_GROUP_SIZE,
            "stateful": stateful,
            "code_shift": CODE_SHIFT,
            "codes_in_kernel": in_kernel,
            "blocks_p2": BLOCKS_P2,
            "hardware": hardware,
        },
    )
    assert (E4M3_CVT in ptx) == hardware and F16_TRUNCATION not in ptx


@pytest.mark.parametrize("ef", [True, False])
@pytest.mark.parametrize(
    ("bits", "nvfp4", "hardware"),
    [(8, False, False), (4, False, False), (4, True, False), (4, True, True)],
)
def test_combine_kernels_compile(
    ef: bool, bits: int, nvfp4: bool, hardware: bool
) -> None:
    part = {
        "part_ptr": "*bf16",
        "part_stride": "i32",
        "rows_ptr": "*i32",
        "err_ptr": "*fp32",
        "err_stride": "i32",
    }
    ptx = _compile(
        kernels._encode_combine_kernel,
        {
            **part,
            "out_ptr": "*u8",
            "deq_ptr": "*fp32",
            "deq_stride": "i32",
            "amax_ptr": "*fp32",
            "row_bytes": "i32",
            "min_range": "fp32",
            **_constexprs(("hidden", "group", "bits", "ef", "nvfp4", "hardware")),
        },
        {
            "hidden": HIDDEN,
            "group": QUANT_GROUP_SIZE,
            "bits": bits,
            "ef": ef,
            "nvfp4": nvfp4,
            "hardware": hardware,
        },
    )
    assert (E4M3_CVT in ptx) == hardware and F16_TRUNCATION not in ptx
    ptx = _compile(
        kernels._decode_combine_kernel,
        {
            "data_ptr": "*u8",
            "row_bytes": "i32",
            "out_ptr": "*fp32",
            "out_stride": "i32",
            **_constexprs(("hidden", "group", "bits", "nvfp4", "hardware")),
        },
        {
            "hidden": HIDDEN,
            "group": QUANT_GROUP_SIZE,
            "bits": bits,
            "nvfp4": nvfp4,
            "hardware": hardware,
        },
    )
    assert ("cvt.rn.f16x2.e4m3x2" in ptx) == hardware
    if not nvfp4:
        return
    _compile(
        kernels._combine_amax_kernel,
        {**part, "amax_ptr": "*fp32", **_constexprs(("hidden", "hidden_p2", "ef"))},
        {"hidden": HIDDEN, "hidden_p2": HIDDEN, "ef": ef},
    )
    ptx = _compile(
        kernels._encode_combine_row_kernel,
        {
            **part,
            "out_ptr": "*u8",
            "deq_ptr": "*fp32",
            "deq_stride": "i32",
            "row_bytes": "i32",
            **_constexprs(("hidden", "blocks_p2", "ef", "hardware")),
        },
        {"hidden": HIDDEN, "blocks_p2": BLOCKS_P2, "ef": ef, "hardware": hardware},
    )
    assert (E4M3_CVT in ptx) == hardware and F16_TRUNCATION not in ptx
