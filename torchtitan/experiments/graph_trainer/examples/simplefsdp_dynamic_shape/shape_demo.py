# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SimpleFSDP shape and memory inference demo.

Run from the repo root (needs a PyTorch nightly with graph_trainer support):
    torchrun --nproc_per_node=2 -m \
        torchtitan.experiments.graph_trainer.examples.simplefsdp_dynamic_shape.shape_demo

Parts:
  1. Static trace: token dim is a concrete int in every node.meta["val"].
  2. Static graph called with a different token count -> fails.
  3. Dynamic trace: mark_dynamic(tokens, 0) -> token dim becomes a SymInt (e.g. s69).
     Every node size is a sympy expression of that symbol. One graph runs any T.
  4. Memory estimate from FakeTensor metadata vs. CUDA allocator peak.
  5. Inductor codegen on the dynamic graph: buffer sizes computed from the symbol at runtime.
"""

import dataclasses
import logging
import os

import sympy
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch._inductor.fx_passes.memory_estimator import (
    build_memory_profile,
    GraphAliasTracker,
)
from torch.distributed.device_mesh import init_device_mesh

from torchtitan.experiments.graph_trainer.common_utils import (
    compute_parameter_gradients,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    minimal_fx_tracer,
    run_traced,
)
from torchtitan.experiments.graph_trainer.simple_fsdp import (
    data_parallel,
    MixedPrecisionPolicy,
)

VOCAB, DIM, FFN, NUM_LAYERS = 8192, 1024, 4096, 4
TRACE_T = 4096


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.RMSNorm(DIM)
        self.w1 = nn.Linear(DIM, FFN, bias=False)
        self.w2 = nn.Linear(FFN, DIM, bias=False)

    def forward(self, x_TD):
        # Attention-style head split: view() records the token count. Under
        # static tracing it is baked into the graph as a constant.
        h_TF = self.w1(self.norm(x_TD))
        h_THK = h_TF.view(x_TD.shape[0], 32, -1)
        h_TF = F.silu(h_THK).view(x_TD.shape[0], -1)
        return x_TD + self.w2(h_TF)


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_embeddings = nn.Embedding(VOCAB, DIM)
        self.layers = nn.ModuleList(Block() for _ in range(NUM_LAYERS))
        self.norm = nn.RMSNorm(DIM)
        self.output = nn.Linear(DIM, VOCAB, bias=False)

    def forward(self, tokens_T):
        x_TD = self.tok_embeddings(tokens_T)
        for layer in self.layers:
            x_TD = layer(x_TD)
        return self.output(self.norm(x_TD))


rank = 0


def log(*args):
    if rank == 0:
        print(*args, flush=True)


def fmt_val(val):
    if isinstance(val, torch.Tensor):
        dt = str(val.dtype).replace("torch.", "")
        return f"{dt}{list(val.shape)}"
    if isinstance(val, (list, tuple)):
        return "(" + ", ".join(fmt_val(v) for v in val) + ")"
    return str(val)


def nbytes_expr(val):
    """Byte count of a FakeTensor as int or sympy expression."""
    nb = val.untyped_storage().nbytes()
    return nb.node.expr if isinstance(nb, torch.SymInt) else int(nb)


def show_key_nodes(gm, title):
    log(f"\n--- {title}: key nodes (name | op | shape | storage bytes) ---")
    seen = {}
    for node in gm.graph.nodes:
        val = node.meta.get("val")
        if node.op == "placeholder":
            if isinstance(val, torch.Tensor) and val.dtype == torch.int64:
                log(
                    f"  {node.name:28s} placeholder        {fmt_val(val):24s} "
                    f"{nbytes_expr(val)}"
                )
            continue
        if node.op != "call_function" or not isinstance(val, torch.Tensor):
            continue
        key = str(node.target)
        interesting = (
            "all_gather" in key
            or "reduce_scatter" in key
            or "embedding" in key
            or key in ("aten.mm.default", "aten._fused_rms_norm.default")
        )
        if not interesting or seen.get(key, 0) >= 2:
            continue
        seen[key] = seen.get(key, 0) + 1
        log(
            f"  {node.name:28s} {key.split('.')[-2][:18]:18s} "
            f"{fmt_val(val):24s} {nbytes_expr(val)}"
        )


def estimate_peak(gm, hint=None):
    """Peak bytes over graph execution from FakeTensor metadata.

    Reuses inductor's build_memory_profile: one allocation per fresh storage
    (views share storage), freed after its last use. Placeholders (sharded
    params, tokens) are owned by the caller and never freed. Returns
    (peak_above_inputs, symbolic_expr_at_peak).
    """

    def size_of(nb):
        if isinstance(nb, torch.SymInt):
            return nb.node.expr
        return int(nb)

    sym_profile = build_memory_profile(
        gm.graph, is_releasable=lambda n: n.op != "placeholder", size_of=size_of
    )

    def concrete(e):
        if isinstance(e, int):
            return e
        return int(e.subs({s: hint for s in e.free_symbols}))

    vals = [concrete(e) for e in sym_profile]
    i_peak = max(range(len(vals)), key=vals.__getitem__)
    base = sym_profile[0]
    return vals[i_peak] - vals[0], sympy.expand(sym_profile[i_peak] - base)


def live_at_peak(gm, hint, top=8):
    """List the largest storages alive at the peak step (same rules as above)."""
    nodes = list(gm.graph.nodes)
    alias = GraphAliasTracker(nodes)

    def size(sk):
        nb = sk.storage.nbytes()
        return nb.node.expr if isinstance(nb, torch.SymInt) else int(nb)

    def concrete(e):
        return (
            e if isinstance(e, int) else int(e.subs({s: hint for s in e.free_symbols}))
        )

    live, best, best_set = {}, -1, {}
    for node in nodes:
        if node.op in ("placeholder", "get_attr", "output"):
            continue
        for sk in alias.get_fresh_allocations(node):
            if sk.device.type != "cpu":
                live[sk] = node
        total = sum(concrete(size(sk)) for sk in live)
        if total > best:
            best, best_set, at = total, dict(live), node
        for sk in alias.get_storages_last_used(node):
            live.pop(sk, None)
    rows = sorted(best_set.items(), key=lambda kv: -concrete(size(kv[0])))
    log(
        f"  peak at node {at.name} ({at.target}); {len(rows)} live non-input storages; top {top}:"
    )
    for sk, node in rows[:top]:
        val = node.meta["val"]
        val = val if isinstance(val, torch.Tensor) else None
        log(
            f"    {str(size(sk)):>12s} B  {node.name:32s} {fmt_val(val) if val is not None else ''}"
        )


def measure_peak(run, *args):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out = run(*args)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    del out
    return peak


def mib(x):
    return f"{x / 2**20:8.1f} MiB"


def main():
    global rank
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    mesh = init_device_mesh(
        "cuda", (dist.get_world_size(),), mesh_dim_names=("dp_shard",)
    )

    torch.manual_seed(0)
    model = TinyLM().cuda()
    model = data_parallel(
        model,
        mesh,
        mode="fully_shard",
        mp_policy=MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32
        ),
    )
    log(
        f"world={dist.get_world_size()}  tok_embeddings.weight at rest: "
        f"{type(model.tok_embeddings._parameters['weight'].data).__name__} "
        f"local={list(model.tok_embeddings._parameters['weight'].to_local().shape)}"
    )

    def fwd_bwd(tokens_T, labels_T):
        logits_TV = model(tokens_T)
        loss = F.cross_entropy(logits_TV.float(), labels_T)
        grads = compute_parameter_gradients(loss, model.named_parameters())
        return [loss, *grads]

    def make_batch(num_tokens, seed):
        g = torch.Generator(device="cuda").manual_seed(seed)
        tokens = torch.randint(0, VOCAB, (num_tokens,), device="cuda", generator=g)
        labels = torch.randint(0, VOCAB, (num_tokens,), device="cuda", generator=g)
        return tokens, labels

    # ---- Part 1: static trace -------------------------------------------
    tokens, labels = make_batch(TRACE_T, seed=1)
    static = minimal_fx_tracer(fwd_bwd, module=model)(tokens, labels)
    show_key_nodes(static.gm, f"STATIC trace (T={TRACE_T})")
    free_syms = {
        s
        for n in static.gm.graph.nodes
        if isinstance(n.meta.get("val"), torch.Tensor)
        for s in (
            n.meta["val"].untyped_storage().nbytes().node.expr.free_symbols
            if isinstance(n.meta["val"].untyped_storage().nbytes(), torch.SymInt)
            else ()
        )
    }
    log(f"  free symbols in static graph: {free_syms or 'none'}")
    run_static = run_traced(static, module=model)

    # ---- Part 2: static graph with a different T ------------------------
    log("\n--- Part 2: run STATIC graph with T=2048 ---")
    try:
        run_static(*make_batch(2048, seed=2))
        log("  unexpectedly succeeded")
    except Exception as e:  # noqa: BLE001
        msg = str(e).splitlines()[0][:160]
        log(f"  FAILED as expected: {type(e).__name__}: {msg}")
    baked = [
        n
        for n in static.gm.graph.nodes
        if n.target is torch.ops.aten.view.default and TRACE_T in n.args[1]
    ]
    log(
        f"  view nodes with T={TRACE_T} baked in: {len(baked)}; e.g. "
        f"{baked[0].name}{tuple(baked[0].args[1]) if baked else ''}"
    )

    # ---- Part 3: dynamic trace ------------------------------------------
    tokens_d, labels_d = make_batch(TRACE_T, seed=1)
    torch._dynamo.mark_dynamic(tokens_d, 0, min=2, max=65536)
    torch._dynamo.mark_dynamic(labels_d, 0, min=2, max=65536)
    dynamic = minimal_fx_tracer(fwd_bwd, module=model)(tokens_d, labels_d)
    show_key_nodes(dynamic.gm, "DYNAMIC trace (mark_dynamic tokens dim 0)")
    fake_tok = next(
        n.meta["val"]
        for n in dynamic.gm.graph.nodes
        if n.op == "placeholder"
        and isinstance(n.meta.get("val"), torch.Tensor)
        and n.meta["val"].dtype == torch.int64
    )
    shape_env = fake_tok.shape[0].node.shape_env
    var_to_range = {
        str(k): (int(v.lower), str(v.upper)) for k, v in shape_env.var_to_range.items()
    }
    replacements = {str(k): str(v) for k, v in shape_env.replacements.items()}
    log(f"  ShapeEnv var_to_range: {var_to_range}")
    log(f"  ShapeEnv replacements: {replacements}")
    log(f"  ShapeEnv guards: {[str(g.expr) for g in shape_env.guards]}")
    run_dynamic = run_traced(dynamic, module=model)

    # Same T as the static trace -> same loss bits.
    loss_s = run_static(tokens, labels)[0]
    loss_d = run_dynamic(tokens, labels)[0]
    log(
        f"\n  T={TRACE_T}: static loss={loss_s.item():.6f}  dynamic loss="
        f"{loss_d.item():.6f}  bitwise equal={torch.equal(loss_s, loss_d)}"
    )

    # ---- Part 4: estimate vs. measured ----------------------------------
    log(
        "\n--- Part 4: peak memory above inputs: estimate (FakeTensor) vs CUDA allocator ---"
    )
    est, expr = estimate_peak(static.gm)
    run_static(tokens, labels)  # warm up NCCL / cuBLAS workspaces
    meas = measure_peak(run_static, tokens, labels)
    log(
        f"  STATIC  T={TRACE_T:5d}: estimate {mib(est)}  measured {mib(meas)}  "
        f"ratio {meas / est:.3f}"
    )
    for num_tokens in (1024, 4096, 8192, 16384):
        batch = make_batch(num_tokens, seed=3)
        est, expr = estimate_peak(dynamic.gm, hint=num_tokens)
        run_dynamic(*batch)
        meas = measure_peak(run_dynamic, *batch)
        log(
            f"  DYNAMIC T={num_tokens:5d}: estimate {mib(est)}  measured {mib(meas)}  "
            f"ratio {meas / est:.3f}   peak expr = {expr}"
        )
    live_at_peak(dynamic.gm, hint=TRACE_T)

    # ---- Part 5: inductor on the dynamic graph --------------------------
    if os.environ.get("DEMO_INDUCTOR", "1") == "1":
        from torchtitan.experiments.graph_trainer.inductor_passes import (
            standalone_inductor_compilation_pass,
        )

        out_dir = os.environ.get("DEMO_OUT", ".")
        handler = logging.FileHandler(f"{out_dir}/output_code_rank{rank}.py", mode="w")
        torch._logging.set_logs(output_code=True)
        logging.getLogger("torch._inductor.codecache").addHandler(handler)
        logging.getLogger("torch._inductor.graph").addHandler(handler)
        compiled = standalone_inductor_compilation_pass(
            dynamic.gm, dynamic.example_inputs
        )
        torch._logging.set_logs()
        run_compiled = run_traced(
            dataclasses.replace(dynamic, gm=compiled), module=model
        )
        log("\n--- Part 5: Inductor-compiled DYNAMIC graph ---")
        for num_tokens in (1024, 4096, 8192):
            batch = make_batch(num_tokens, seed=3)
            ref = run_dynamic(*batch)[0]
            got = run_compiled(*batch)[0]
            run_compiled(*batch)
            meas = measure_peak(run_compiled, *batch)
            log(
                f"  T={num_tokens:5d}: eager-graph loss={ref.item():.6f}  "
                f"inductor loss={got.item():.6f}  inductor peak {mib(meas)}"
            )

        compiled_static = standalone_inductor_compilation_pass(
            static.gm, static.example_inputs
        )
        run_compiled_static = run_traced(
            dataclasses.replace(static, gm=compiled_static), module=model
        )
        log("\n--- Part 5b: Inductor-compiled STATIC graph ---")
        run_compiled_static(tokens, labels)
        log(
            f"  T={TRACE_T}: OK, peak {mib(measure_peak(run_compiled_static, tokens, labels))}"
        )
        try:
            run_compiled_static(*make_batch(2048, seed=2))
            log("  T=2048: unexpectedly succeeded")
        except Exception as e:  # noqa: BLE001
            log(
                f"  T=2048: FAILED as expected: {type(e).__name__}: "
                f"{str(e).splitlines()[0][:160]}"
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
