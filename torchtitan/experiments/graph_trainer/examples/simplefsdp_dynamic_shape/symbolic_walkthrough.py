# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Step-by-step walkthrough of symbolic shape propagation in minimal_fx_tracer.

Single GPU, no FSDP. It traces forward + loss of one small block with the
token dim marked dynamic, and prints how the symbol flows through every node.

Run from the repo root:
    python -m \
        torchtitan.experiments.graph_trainer.examples.simplefsdp_dynamic_shape.symbolic_walkthrough

Shape legend: T = number of tokens, D = model dim, F = FFN dim,
H = number of heads, K = head dim, V = vocab size.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    minimal_fx_tracer,
    run_traced,
)

V, D, F_DIM, H = 8192, 1024, 4096, 32
TRACE_T = 4096


class TinyBlockLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_embeddings = nn.Embedding(V, D)
        self.w1 = nn.Linear(D, F_DIM, bias=False)
        self.w2 = nn.Linear(F_DIM, D, bias=False)
        self.output = nn.Linear(D, V, bias=False)

    def forward(self, tokens_T):
        x_TD = self.tok_embeddings(tokens_T)
        h_TF = self.w1(x_TD)
        h_THK = h_TF.view(tokens_T.shape[0], H, -1)
        h_TF = F.silu(h_THK).view(tokens_T.shape[0], -1)
        x_TD = x_TD + self.w2(h_TF)
        return self.output(x_TD)


def fmt(val):
    if isinstance(val, torch.Tensor):
        return f"{str(val.dtype).replace('torch.', '')}{list(val.shape)}"
    if isinstance(val, (list, tuple)):
        return "(" + ", ".join(fmt(v) for v in val) + ")"
    return str(val)


def nbytes(val):
    nb = val.untyped_storage().nbytes()
    return str(nb.node.expr) if isinstance(nb, torch.SymInt) else str(nb)


def main():
    torch.manual_seed(0)
    model = TinyBlockLM().cuda()

    def fwd_loss(tokens_T, labels_T):
        logits_TV = model(tokens_T)
        return F.cross_entropy(logits_TV, labels_T)

    tokens = torch.randint(0, V, (TRACE_T,), device="cuda")
    labels = torch.randint(0, V, (TRACE_T,), device="cuda")

    # Step 1: mark the token dim. This only sets Python attributes on the
    # real tensors; nothing is traced yet.
    torch._dynamo.mark_dynamic(tokens, 0, min=2, max=65536)
    torch._dynamo.mark_dynamic(labels, 0, min=2, max=65536)
    print("Step 1: tokens._dynamo_dynamic_indices =", tokens._dynamo_dynamic_indices)

    # Step 2: trace. _fakeify_input turns each marked dim into a fresh symbol.
    traced = minimal_fx_tracer(fwd_loss, module=model)(tokens, labels)
    gm = traced.gm
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    shape_env = next(
        n.meta["val"].shape[0].node.shape_env
        for n in placeholders
        if isinstance(n.meta["val"].shape[0], torch.SymInt)
    )

    print("\nStep 2: placeholders (fake inputs)")
    for n in placeholders:
        print(f"  {n.name:10s} {fmt(n.meta['val']):22s} bytes={nbytes(n.meta['val'])}")
    print("  symbol sources:")
    for sym, sources in shape_env.var_to_sources.items():
        print(
            f"    {sym} <- {sources[0].name}  hint={shape_env.backed_var_to_val[sym]}"
        )

    # Step 3: walk the graph. Each op's meta kernel computes the output shape
    # from the input shapes, so the symbol flows into every downstream node.
    print("\nStep 3: node by node (op | input shapes -> output shape | bytes)")
    for n in gm.graph.nodes:
        if n.op != "call_function":
            continue
        val = n.meta.get("val")
        ins = [
            fmt(a.meta["val"]) if isinstance(a, torch.fx.Node) else str(a)
            for a in n.args
            if not isinstance(a, torch.fx.Node) or "val" in a.meta
        ]
        out_bytes = nbytes(val) if isinstance(val, torch.Tensor) else "-"
        print(
            f"  {str(n.target):42s} {', '.join(ins)[:70]:70s} -> {fmt(val):22s} {out_bytes}"
        )

    # Step 4: what the ShapeEnv learned while tracing.
    print("\nStep 4: ShapeEnv after tracing")
    print(
        "  var_to_range:",
        {
            str(k): (int(v.lower), int(v.upper))
            for k, v in shape_env.var_to_range.items()
        },
    )
    print(
        "  replacements:", {str(k): str(v) for k, v in shape_env.replacements.items()}
    )
    print("  guards:      ", [str(g.expr) for g in shape_env.guards])

    # Step 5: the generated Python code reads T from the input at runtime.
    print("\nStep 5: generated graph code")
    print(gm.code)

    # Step 6: run the same graph with other token counts.
    print("Step 6: run the traced graph with other T")
    run = run_traced(traced, module=model)
    for num_tokens in (1024, 3000):
        t = torch.randint(0, V, (num_tokens,), device="cuda")
        y = torch.randint(0, V, (num_tokens,), device="cuda")
        with torch.no_grad():
            ref = fwd_loss(t, y)
        got = run(t, y)
        got = got[0] if isinstance(got, (list, tuple)) else got
        mm_node = next(n for n in gm.graph.nodes if n.name == "mm")
        expr = mm_node.meta["val"].untyped_storage().nbytes().node.expr
        mm_bytes = int(expr.subs({s: num_tokens for s in expr.free_symbols}))
        print(
            f"  T={num_tokens}: graph loss={got.item():.6f} eager loss={ref.item():.6f} "
            f"| {mm_node.name} bytes = {expr} -> {mm_bytes}"
        )

    # Step 7: Python control flow on a backed symbol adds a guard.
    print("\nStep 7: branch on the symbol")

    def fwd_branch(tokens_T, labels_T):
        logits_TV = model(tokens_T)
        if tokens_T.shape[0] > 2048:
            logits_TV = logits_TV * 0.5
        return F.cross_entropy(logits_TV, labels_T)

    t = torch.randint(0, V, (TRACE_T,), device="cuda")
    y = torch.randint(0, V, (TRACE_T,), device="cuda")
    torch._dynamo.mark_dynamic(t, 0, min=2, max=65536)
    torch._dynamo.mark_dynamic(y, 0, min=2, max=65536)
    traced_b = minimal_fx_tracer(fwd_branch, module=model)(t, y)
    env_b = next(
        n.meta["val"].shape[0].node.shape_env
        for n in traced_b.gm.graph.nodes
        if n.op == "placeholder" and isinstance(n.meta["val"].shape[0], torch.SymInt)
    )
    print("  mark_dynamic guards:", [str(g.expr) for g in env_b.guards])
    print(
        "  mark_dynamic ranges:",
        {str(k): (int(v.lower), int(v.upper)) for k, v in env_b.var_to_range.items()},
    )
    # The branch is baked in. run_traced does not check guards by default, so
    # a T outside the guarded range runs the wrong branch without an error.
    run_b = run_traced(traced_b, module=model)
    t_small = torch.randint(0, V, (1024,), device="cuda")
    y_small = torch.randint(0, V, (1024,), device="cuda")
    with torch.no_grad():
        ref_small = fwd_branch(t_small, y_small)
    got_small = run_b(t_small, y_small)
    got_small = got_small[0] if isinstance(got_small, (list, tuple)) else got_small
    print(
        f"  T=1024 (violates s79 > 2048): graph loss={got_small.item():.6f} "
        f"eager loss={ref_small.item():.6f}"
    )

    t = torch.randint(0, V, (TRACE_T,), device="cuda")
    y = torch.randint(0, V, (TRACE_T,), device="cuda")
    torch._dynamo.decorators.mark_unbacked(t, 0)
    torch._dynamo.decorators.mark_unbacked(y, 0)
    try:
        minimal_fx_tracer(fwd_branch, module=model)(t, y)
        print("  mark_unbacked: traced (unexpected)")
    except Exception as e:  # noqa: BLE001
        print(f"  mark_unbacked: {type(e).__name__}: {str(e).splitlines()[0][:150]}")


if __name__ == "__main__":
    main()
