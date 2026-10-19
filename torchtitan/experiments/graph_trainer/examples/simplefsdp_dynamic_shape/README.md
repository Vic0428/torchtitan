# SimpleFSDP: static vs. dynamic token shapes

This example shows how graph_trainer infers tensor shapes and memory at
trace time, and what changes when the token dimension is dynamic.

graph_trainer traces the full train step (forward, loss and backward) with
`make_fx` under `FakeTensorMode`. Each FX node stores a FakeTensor in
`node.meta["val"]`. The FakeTensor has shape, stride and dtype, so the byte
count of every tensor is known before the graph runs.

## Files

| File | What it does |
|---|---|
| `shape_demo.py` | Standalone 2-GPU demo. It traces a small LM with SimpleFSDP twice: once with static shapes, once with `torch._dynamo.mark_dynamic(tokens, 0)`. It then compares the FakeTensor memory estimate with the CUDA allocator peak, and compiles both graphs with Inductor. |
| `symbolic_walkthrough.py` | Single-GPU, no FSDP. It traces forward + loss of one small block with `mark_dynamic`, then prints the symbol for each input, the shape of each node, the ShapeEnv guards, and the generated code. It also shows that a Python branch on the symbol adds a guard, and that `run_traced` then runs a T outside the guarded range without an error. |
| `custom_op_memory.py` | Single GPU. A custom op allocates a hidden workspace inside its real kernel. The FakeTensor memory estimate misses it, and the gap to the measured peak is exactly the workspace size. |
| `recipe.py` | `simplefsdp_llama3_fsdp4`: the llama3 debug model on graph_trainer with FSDP=4 and flex attention, 10 steps. It is the normal (static-shape) training path. |

## Run

All commands need a PyTorch nightly that has the APIs graph_trainer uses.
Run them from the repo root.

```bash
# Shape and memory demo, 2 GPUs.
PYTORCH_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=2 -m \
    torchtitan.experiments.graph_trainer.examples.simplefsdp_dynamic_shape.shape_demo

# Symbol-by-symbol walkthrough, 1 GPU.
python -m \
    torchtitan.experiments.graph_trainer.examples.simplefsdp_dynamic_shape.symbolic_walkthrough

# Custom op hidden workspace, 1 GPU.
python -m \
    torchtitan.experiments.graph_trainer.examples.simplefsdp_dynamic_shape.custom_op_memory

# Normal training, 4 GPUs.
NGPU=4 \
MODULE=torchtitan.experiments.graph_trainer.examples.simplefsdp_dynamic_shape.recipe \
CONFIG=simplefsdp_llama3_fsdp4 ./run_train.sh
```

Set `DEMO_INDUCTOR=0` to skip the Inductor part. Set `DEMO_OUT=<dir>` to
choose where the generated Inductor code is written.

## What the demo shows

1. **Static trace.** The token count (4096) is a constant in the graph. The
   graph has `view` nodes with 4096 in their shape arguments.
2. **Static graph, other T.** Running it with T=2048 fails. The plain FX
   graph fails in `mm`. The Inductor-compiled graph fails in
   `assert_size_stride`.
3. **Dynamic trace.** With `mark_dynamic`, the token dimension becomes a
   symbol such as `s69`. Activation shapes become `bf16[s69, 4096]` and
   similar. Parameter all-gather and reduce-scatter outputs stay constant.
   The ShapeEnv merges the tokens and labels symbols, because
   `cross_entropy` needs them to be equal.
4. **Memory estimate.** The demo uses upstream
   `torch._inductor.fx_passes.memory_estimator.build_memory_profile` with a
   `size_of` callback that keeps the sympy expression. The estimated peak is
   equal to the measured peak for T = 1024 to 16384. The peak is a linear
   expression in `s69`.
5. **Inductor on the dynamic graph.** The generated code reads the symbol
   from the input size and allocates each buffer from it. One compiled graph
   runs every T.

## Notes

- The default graph_trainer training path does not call `mark_dynamic`. It
  packs tokens to a fixed length and traces once with static shapes.
- `mark_dynamic` works only on plain tensor inputs. graph_trainer rejects
  marks on wrapper subclasses such as DTensor.
- `int()` on a backed `SymInt` adds a guard and specializes the symbol.
  `optimization_hint()` does not.
