# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Show that a custom op's internal workspace is not visible to FX tracing.

The custom op allocates a hidden workspace (4x its input) inside its real
kernel. Its fake impl only describes the output. The FakeTensor-based
memory estimate therefore misses the workspace, and the gap to the CUDA
allocator peak is exactly the workspace size.

Run from the repo root (1 GPU):
    python -m \
        torchtitan.experiments.graph_trainer.examples.simplefsdp_dynamic_shape.custom_op_memory
"""

import torch
from torch._inductor.fx_passes.memory_estimator import build_memory_profile

from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    minimal_fx_tracer,
    run_traced,
)

WORKSPACE_FACTOR = 4
DIM = 4096


@torch.library.custom_op("demo::scale_with_workspace", mutates_args=())
def scale_with_workspace(x: torch.Tensor) -> torch.Tensor:
    # Hidden temporary buffer: allocated and freed inside the real kernel.
    workspace = torch.empty(
        x.numel() * WORKSPACE_FACTOR, dtype=x.dtype, device=x.device
    )
    workspace.fill_(1.0)
    return x * workspace[: x.numel()].view_as(x)


@scale_with_workspace.register_fake
def _(x):
    # The fake impl only describes the output; it never sees the workspace.
    return torch.empty_like(x)


def fn(x_TD):
    y_TD = torch.relu(x_TD)
    z_TD = scale_with_workspace(y_TD)
    return (z_TD.sum(),)


def size_of(nbytes):
    return nbytes.node.expr if isinstance(nbytes, torch.SymInt) else int(nbytes)


def main():
    x = torch.randn(4096, DIM, device="cuda")
    torch._dynamo.mark_dynamic(x, 0)
    traced = minimal_fx_tracer(fn)(x)
    print(traced.gm.code)
    for node in traced.gm.graph.nodes:
        if node.op == "call_function":
            print(f"  {str(node.target):36s} -> {list(node.meta['val'].shape)}")

    profile = build_memory_profile(
        traced.gm.graph, is_releasable=lambda n: n.op != "placeholder", size_of=size_of
    )
    run = run_traced(traced)
    for num_tokens in (1024, 4096, 8192):
        x_TD = torch.randn(num_tokens, DIM, device="cuda")
        vals = [
            e
            if isinstance(e, int)
            else int(e.subs({s: num_tokens for s in e.free_symbols}))
            for e in profile
        ]
        estimate = max(vals) - vals[0]
        run(x_TD)  # warm up
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        run(x_TD)
        torch.cuda.synchronize()
        measured = torch.cuda.max_memory_allocated() - base
        workspace = num_tokens * DIM * x_TD.element_size() * WORKSPACE_FACTOR
        print(
            f"T={num_tokens}: estimate {estimate / 2**20:7.1f} MiB  "
            f"measured {measured / 2**20:7.1f} MiB  "
            f"gap {(measured - estimate) / 2**20:7.1f} MiB  "
            f"hidden workspace {workspace / 2**20:7.1f} MiB"
        )


if __name__ == "__main__":
    main()
