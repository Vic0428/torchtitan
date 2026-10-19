# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Multi-step SimpleFSDP training where a step's T selects a different branch.

The loss adds a z-loss term only when the step has at least
Z_LOSS_MIN_TOKENS tokens. The token count T changes every step. Each step
compares the graph result with eager on the same parameters, then applies the
graph gradients with SGD. Three ways to run the traced graph:

  naive    Trace once with mark_dynamic and reuse the graph (what graph_trainer
           does). The branch taken at trace time is baked in; guards are not
           checked, so a step on the other side of the branch is silently wrong.
  guarded  Keep a list of traced graphs. Before each step, evaluate each
           graph's ShapeEnv guards on the real inputs; on a miss, trace again
           with this step's inputs and add the new graph to the list.
  sym_ite  Write the branch as torch.sym_ite, so one graph covers every T and
           has no guard from the branch.

Run from the repo root (2 GPUs):
    torchrun --nproc_per_node=2 -m \
        torchtitan.experiments.graph_trainer.examples.simplefsdp_dynamic_shape.multi_step_control_flow

Shape legend: T = number of tokens, D = model dim, F = FFN dim, V = vocab size.
"""

import os
import time

import sympy
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch._inductor.fx_passes.memory_estimator import build_memory_profile
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

V, D, F_DIM, NUM_LAYERS = 8192, 1024, 4096, 2
Z_LOSS_MIN_TOKENS = 2048
Z_LOSS_WEIGHT = 1e-4
STEP_TOKENS = [4096, 4096, 1024, 4096, 1024, 8192, 1500, 2048, 2047]
LR = 1e-2

rank = 0


def log(*args):
    if rank == 0:
        print(*args, flush=True)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.RMSNorm(D)
        self.w1 = nn.Linear(D, F_DIM, bias=False)
        self.w2 = nn.Linear(F_DIM, D, bias=False)

    def forward(self, x_TD):
        return x_TD + self.w2(F.silu(self.w1(self.norm(x_TD))))


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_embeddings = nn.Embedding(V, D)
        self.layers = nn.ModuleList(Block() for _ in range(NUM_LAYERS))
        self.norm = nn.RMSNorm(D)
        self.output = nn.Linear(D, V, bias=False)

    def forward(self, tokens_T):
        x_TD = self.tok_embeddings(tokens_T)
        for layer in self.layers:
            x_TD = layer(x_TD)
        return self.output(self.norm(x_TD))


def build_model(mesh):
    torch.manual_seed(0)
    model = TinyLM().cuda()
    return data_parallel(
        model,
        mesh,
        mode="fully_shard",
        mp_policy=MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32
        ),
    )


def make_loss_fns(model):
    def fwd_bwd_if(tokens_T, labels_T):
        logits_TV = model(tokens_T).float()
        loss = F.cross_entropy(logits_TV, labels_T)
        # Python control flow on the dynamic dim.
        if tokens_T.shape[0] >= Z_LOSS_MIN_TOKENS:
            loss = loss + Z_LOSS_WEIGHT * logits_TV.logsumexp(-1).pow(2).mean()
        grads = compute_parameter_gradients(loss, model.named_parameters())
        return [loss, *grads]

    def fwd_bwd_sym_ite(tokens_T, labels_T):
        logits_TV = model(tokens_T).float()
        loss = F.cross_entropy(logits_TV, labels_T)
        # Same math, written as an expression: both terms are in the graph.
        use_z_loss = torch.sym_ite(tokens_T.shape[0] >= Z_LOSS_MIN_TOKENS, 1, 0)
        z_loss = logits_TV.logsumexp(-1).pow(2).mean()
        loss = loss + Z_LOSS_WEIGHT * use_z_loss * z_loss
        grads = compute_parameter_gradients(loss, model.named_parameters())
        return [loss, *grads]

    return fwd_bwd_if, fwd_bwd_sym_ite


def make_batch(num_tokens, step):
    g = torch.Generator(device="cuda").manual_seed(1000 + step)
    tokens = torch.randint(0, V, (num_tokens,), device="cuda", generator=g)
    labels = torch.randint(0, V, (num_tokens,), device="cuda", generator=g)
    return tokens, labels


def trace_dynamic(fn, model, tokens, labels):
    tokens, labels = tokens.clone(), labels.clone()
    torch._dynamo.mark_dynamic(tokens, 0, min=2, max=65536)
    torch._dynamo.mark_dynamic(labels, 0, min=2, max=65536)
    return minimal_fx_tracer(fn, module=model)(tokens, labels)


def shape_env_of(traced):
    for node in traced.gm.graph.nodes:
        val = node.meta.get("val")
        if node.op == "placeholder" and isinstance(val, torch.Tensor):
            for size in val.shape:
                if isinstance(size, torch.SymInt):
                    return size.node.shape_env
    return None


def guard_strs(traced):
    return [str(g.expr) for g in shape_env_of(traced).guards]


def guards_hold(traced, user_args):
    """Evaluate the graph's ShapeEnv guards on this step's real inputs."""
    placeholders = [n for n in traced.gm.graph.nodes if n.op == "placeholder"]
    fakes = [n.meta["val"] for n in placeholders[-len(user_args) :]]
    return shape_env_of(traced).evaluate_guards_for_args(fakes, list(user_args))


def peak_expr(traced):
    def size_of(nb):
        return nb.node.expr if isinstance(nb, torch.SymInt) else int(nb)

    profile = build_memory_profile(
        traced.gm.graph,
        is_releasable=lambda n: n.op != "placeholder",
        size_of=size_of,
    )
    return [sympy.expand(e - profile[0]) for e in profile]


def peak_at(profile, num_tokens):
    return max(
        int(e.subs({s: num_tokens for s in sympy.sympify(e).free_symbols}))
        for e in profile
    )


class GuardedGraphCache:
    """Pick a traced graph whose guards hold; trace a new one on a miss."""

    def __init__(self, fn, model):
        self.fn, self.model, self.entries = fn, model, []

    def __call__(self, tokens, labels):
        for entry in self.entries:
            if guards_hold(entry["traced"], (tokens, labels)):
                entry["steps"] += 1
                return entry["run"](tokens, labels), False
        t0 = time.perf_counter()
        traced = trace_dynamic(self.fn, self.model, tokens, labels)
        self.entries.append(
            {
                "traced": traced,
                "run": run_traced(traced, module=self.model),
                "steps": 1,
                "trace_T": tokens.shape[0],
                "trace_s": time.perf_counter() - t0,
            }
        )
        return self.entries[-1]["run"](tokens, labels), True


def max_grad_diff(grads_a, grads_b):
    return max(
        (a.to_local().float() - b.to_local().float()).abs().max().item()
        for a, b in zip(grads_a, grads_b)
    )


def train(strategy, mesh):
    model = build_model(mesh)
    fwd_bwd_if, fwd_bwd_sym_ite = make_loss_fns(model)
    params = [p for _, p in model.named_parameters()]
    opt = torch.optim.SGD(params, lr=LR)

    tokens, labels = make_batch(STEP_TOKENS[0], step=0)
    if strategy == "naive":
        traced = trace_dynamic(fwd_bwd_if, model, tokens, labels)
        run_naive = run_traced(traced, module=model)
        log(f"  traced once at T={STEP_TOKENS[0]}; guards: {guard_strs(traced)}")

        def run_step(t, y):
            return run_naive(t, y), False

    elif strategy == "guarded":
        run_step = cache = GuardedGraphCache(fwd_bwd_if, model)
    else:
        traced = trace_dynamic(fwd_bwd_sym_ite, model, tokens, labels)
        run_ite = run_traced(traced, module=model)
        log(f"  traced once at T={STEP_TOKENS[0]}; guards: {guard_strs(traced)}")

        def run_step(t, y):
            return run_ite(t, y), False

    log(
        f"  {'step':>4s} {'T':>5s} {'z-loss?':>7s} {'graph loss':>11s} "
        f"{'eager loss':>11s} {'|dloss|':>9s} {'max|dgrad|':>10s}  note"
    )
    for step, num_tokens in enumerate(STEP_TOKENS):
        tokens, labels = make_batch(num_tokens, step)
        loss_ref, *grads_ref = fwd_bwd_if(tokens, labels)  # eager, same params
        (loss, *grads), retraced = run_step(tokens, labels)
        note = "retraced" if retraced else ""
        dloss = abs(loss.item() - loss_ref.item())
        dgrad = max_grad_diff(grads, grads_ref)
        if dloss > 1e-5:
            note += " WRONG"
        log(
            f"  {step:4d} {num_tokens:5d} {str(num_tokens >= Z_LOSS_MIN_TOKENS):>7s} "
            f"{loss.item():11.6f} {loss_ref.item():11.6f} {dloss:9.2e} {dgrad:10.2e}  {note}"
        )
        for p, g in zip(params, grads):
            p.grad = g
        opt.step()
        opt.zero_grad(set_to_none=True)

    if strategy != "guarded":
        profile = peak_expr(traced)
        log(
            f"  estimated peak at T=4096 / T=1024: "
            f"{peak_at(profile, 4096) / 2**20:.1f} / {peak_at(profile, 1024) / 2**20:.1f} MiB"
        )
    if strategy == "guarded":
        log(f"  graphs in cache: {len(cache.entries)}")
        for i, e in enumerate(cache.entries):
            profile = peak_expr(e["traced"])
            log(
                f"    graph {i}: traced at T={e['trace_T']}, used {e['steps']} steps, "
                f"trace {e['trace_s']:.1f}s, guards {guard_strs(e['traced'])}"
            )
            log(
                f"             estimated peak at T=4096 / T=1024: "
                f"{peak_at(profile, 4096) / 2**20:.1f} / {peak_at(profile, 1024) / 2**20:.1f} MiB"
            )


def main():
    global rank
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    mesh = init_device_mesh(
        "cuda", (dist.get_world_size(),), mesh_dim_names=("dp_shard",)
    )
    for strategy in ("naive", "guarded", "sym_ite"):
        log(f"\n=== strategy: {strategy} ===")
        train(strategy, mesh)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
