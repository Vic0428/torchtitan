# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""How tracing handles a buffer whose shape depends on a branch on T.

Target behavior:
    if T > 10: buf = zeros(T, 100)
    else:      buf = zeros(2 * T, 100)

Four ways to write it:
  1. Python ``if`` on the symbol (mark_dynamic / mark_unbacked).
  2. Branch-free ``torch.sym_ite``.
  3. ``torch.cond`` with branches that return different shapes.
  4. ``torch.compile`` with the Python ``if`` (guard -> recompile).

Run from the repo root (1 GPU):
    python -m \
        torchtitan.experiments.graph_trainer.examples.simplefsdp_dynamic_shape.path_dependent_shapes
"""

import torch
from torch._inductor.fx_passes.memory_estimator import build_memory_profile

from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    minimal_fx_tracer,
    run_traced,
)

TRACE_T = 4096
TEST_TS = (5, 20)


def expected_rows(num_tokens):
    return num_tokens if num_tokens > 10 else 2 * num_tokens


def make_input(num_tokens, mark=None):
    x = torch.randn(num_tokens, 100, device="cuda")
    if mark == "dynamic":
        torch._dynamo.mark_dynamic(x, 0)
    elif mark == "unbacked":
        torch._dynamo.decorators.mark_unbacked(x, 0)
    return x


def shape_env_of(traced):
    for node in traced.gm.graph.nodes:
        val = node.meta.get("val")
        if isinstance(val, torch.Tensor):
            for size in val.shape:
                if isinstance(size, torch.SymInt):
                    return size.node.shape_env
    return None


def size_of(nbytes):
    return nbytes.node.expr if isinstance(nbytes, torch.SymInt) else int(nbytes)


def show(traced):
    for node in traced.gm.graph.nodes:
        val = node.meta.get("val")
        if node.op == "call_function" and isinstance(val, torch.Tensor) and val.dim():
            print(
                f"  {node.name}: shape={list(val.shape)} "
                f"bytes={size_of(val.untyped_storage().nbytes())}"
            )
    env = shape_env_of(traced)
    ranges = {str(k): (int(r.lower), str(r.upper)) for k, r in env.var_to_range.items()}
    print(f"  guards={[str(g.expr) for g in env.guards]} ranges={ranges}")


def run_and_check(traced):
    run = run_traced(traced)
    for num_tokens in TEST_TS:
        rows = int(run(torch.randn(num_tokens, 100, device="cuda"))[1])
        status = "OK" if rows == expected_rows(num_tokens) else "WRONG"
        print(
            f"  T={num_tokens}: buf rows at runtime={rows} "
            f"expected={expected_rows(num_tokens)} {status}"
        )


def f_python_if(x_TD):
    num_tokens = x_TD.shape[0]
    if num_tokens > 10:
        buf = torch.zeros(num_tokens, 100, device=x_TD.device)
    else:
        buf = torch.zeros(2 * num_tokens, 100, device=x_TD.device)
    return (buf.sum() + x_TD.sum(), buf.shape[0] * 1.0)


def f_sym_ite(x_TD):
    num_tokens = x_TD.shape[0]
    rows = torch.sym_ite(num_tokens > 10, num_tokens, 2 * num_tokens)
    buf = torch.zeros(rows, 100, device=x_TD.device)
    return (buf.sum() + x_TD.sum(), buf.shape[0] * 1.0)


def f_cond(x_TD):
    def true_fn(x):
        return torch.zeros(x.shape[0], 100, device=x.device) + 1

    def false_fn(x):
        return torch.zeros(2 * x.shape[0], 100, device=x.device) + 1

    buf = torch.cond(x_TD.shape[0] > 10, true_fn, false_fn, (x_TD,))
    return (buf.sum(), buf.shape[0] * 1.0)


def main():
    print("== 1a. Python if + mark_dynamic")
    traced = minimal_fx_tracer(f_python_if)(make_input(TRACE_T, "dynamic"))
    show(traced)
    run_and_check(traced)

    print("\n== 1b. Python if + mark_unbacked")
    try:
        minimal_fx_tracer(f_python_if)(make_input(TRACE_T, "unbacked"))
    except Exception as e:  # noqa: BLE001
        print(f"  {type(e).__name__}: {str(e).splitlines()[0][:120]}")

    print("\n== 2. torch.sym_ite + mark_dynamic")
    traced = minimal_fx_tracer(f_sym_ite)(make_input(TRACE_T, "dynamic"))
    show(traced)
    run_and_check(traced)
    profile = build_memory_profile(
        traced.gm.graph, is_releasable=lambda n: n.op != "placeholder", size_of=size_of
    )
    for num_tokens in TEST_TS:
        vals = [
            v
            if isinstance(v, int)
            else int(v.subs({s: num_tokens for s in v.free_symbols}))
            for v in profile
        ]
        print(f"  T={num_tokens}: estimated peak above inputs={max(vals) - vals[0]} B")

    print("\n== 3. torch.cond + mark_dynamic")
    traced = minimal_fx_tracer(f_cond)(make_input(TRACE_T, "dynamic"))
    show(traced)
    run_and_check(traced)

    print("\n== 4. torch.compile(dynamic=True) with the Python if")
    graphs = []

    def counting_backend(gm, example_inputs):
        graphs.append(gm)
        return gm

    compiled = torch.compile(f_python_if, backend=counting_backend, dynamic=True)
    for num_tokens in (TRACE_T, 20, 5, 7):
        rows = int(compiled(torch.randn(num_tokens, 100, device="cuda"))[1])
        print(
            f"  T={num_tokens}: graphs compiled so far={len(graphs)} "
            f"buf rows={rows} expected={expected_rows(num_tokens)}"
        )


if __name__ == "__main__":
    main()
