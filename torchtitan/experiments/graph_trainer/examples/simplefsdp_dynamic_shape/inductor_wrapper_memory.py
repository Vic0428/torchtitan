# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Estimate peak memory of an Inductor-compiled graph from its wrapper code.

Inductor's own estimate (torch/_inductor/memory.py) uses size hints, so it
gives one number for the traced T. The generated wrapper code keeps the
symbols (for example empty_strided_cuda((s69, 4096), ...)), so walking its
call() body gives the live bytes at every step as an expression of T.

Rules:
- empty_strided_cuda(shape, stride, dtype) allocates a new storage.
- An extern op output (bufN = torch.ops...(...)) allocates a new storage;
  its size comes from the next assert_tensor_metadata(bufN, ...) line.
- reinterpret_tensor(bufM, ...) and "bufN = bufM  # reuse" share bufM's storage.
- "del a, b" drops names; a storage is freed when its last name is dropped.
- Graph inputs (argN_1) belong to the caller and are not counted.

Memory used inside a kernel (for example the cuBLAS workspace of
extern_kernels.mm) is not in the wrapper code, so it is not counted.

Usage (wrapper file path is printed as "Output code written to: ..." when
TORCH_LOGS=output_code is set):
    python -m \
        torchtitan.experiments.graph_trainer.examples.simplefsdp_dynamic_shape.inductor_wrapper_memory \
        <wrapper.py> 1024 4096 8192
"""

import re
import sys

import sympy
import torch

DTYPE_BYTES = {
    "float64": 8,
    "float32": 4,
    "bfloat16": 2,
    "float16": 2,
    "int64": 8,
    "int32": 4,
    "int8": 1,
    "uint8": 1,
    "bool": 1,
}


class _SymbolNamespace(dict):
    """Resolve unknown names (s69, u0, ...) to sympy symbols."""

    def __missing__(self, key):
        return sympy.Symbol(key, integer=True, positive=True)


def _eval_args(text):
    return eval(  # noqa: S307 -- input is Inductor-generated code
        "(" + text + ")", {"__builtins__": {}}, _SymbolNamespace(torch=torch)
    )


def _call_args(line, fn):
    """Return the text inside fn(...) on this line, with nested parens."""
    start = line.index(fn + "(") + len(fn) + 1
    depth, end = 1, start
    while depth:
        depth += {"(": 1, ")": -1}.get(line[end], 0)
        end += 1
    return line[start : end - 1]


def storage_bytes(shape, stride, dtype):
    shape = [sympy.sympify(s) for s in shape]
    stride = [sympy.sympify(s) for s in stride]
    if any(s == 0 for s in shape):
        return sympy.Integer(0)
    span = 1 + sum((s - 1) * st for s, st in zip(shape, stride))
    return sympy.expand(span * DTYPE_BYTES[str(dtype).replace("torch.", "")])


def walk_wrapper(code):
    """Return [(file_line, live_bytes_expr, text)] after each allocation."""
    lines = code.splitlines()
    call_idx = next(
        i for i, line in enumerate(lines) if line.strip() == "def call(self, args):"
    )
    owner = {}  # name -> storage id
    nbytes = {}  # storage id -> bytes expr
    names = {}  # storage id -> live names
    pending = set()  # extern outputs waiting for their metadata line
    live = sympy.Integer(0)
    timeline = []

    def allocate(name, size, file_line, text):
        nonlocal live
        sid = len(nbytes)
        owner[name], nbytes[sid], names[sid] = sid, size, {name}
        live = live + size
        timeline.append((file_line, live, text))

    def share(alias, base):
        if base in owner:
            owner[alias] = owner[base]
            names[owner[base]].add(alias)

    for file_line, raw in enumerate(lines[call_idx + 1 :], start=call_idx + 2):
        if raw.startswith("def ") or raw.startswith("    def "):
            break
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        m = re.match(r"(\w+) = empty_strided_cuda\(", line)
        if m:
            shape, stride, dtype = _eval_args(_call_args(line, "empty_strided_cuda"))
            allocate(m.group(1), storage_bytes(shape, stride, dtype), file_line, line)
        m = re.match(r"(\w+) = reinterpret_tensor\((\w+),", line)
        if m:
            share(*m.groups())
        m = re.match(r"(buf\w*) = (buf\w*)\s*(;|$)", line)
        if m:
            share(m.group(1), m.group(2))
        m = re.match(r"(\w+) = (torch\.ops\.\S+|extern_kernels\.\w+)\(", line)
        if m:
            pending.add(m.group(1))
        m = re.match(r"assert_tensor_metadata\((\w+), ", line)
        if m and m.group(1) in pending:
            args = _call_args(line, "assert_tensor_metadata")
            # args: "bufN, shape, stride, dtype, 'op name'"
            meta = args.split(", ", 1)[1].rsplit(", '", 1)[0]
            shape, stride, dtype = _eval_args(meta)
            pending.discard(m.group(1))
            allocate(m.group(1), storage_bytes(shape, stride, dtype), file_line, line)

        for d in re.finditer(r"\bdel ([\w, ]+)", line):
            for name in (n.strip() for n in d.group(1).split(",")):
                sid = owner.pop(name, None)
                if sid is None:
                    continue
                names[sid].discard(name)
                if not names[sid]:
                    live = live - nbytes[sid]
        if line.startswith("return "):
            break

    if pending:
        print(f"WARNING: no size found for {sorted(pending)}", file=sys.stderr)
    return timeline


def peak_at(timeline, num_tokens):
    best = None
    for file_line, expr, text in timeline:
        value = int(expr.subs({s: num_tokens for s in expr.free_symbols}))
        if best is None or value > best[0]:
            best = (value, file_line, expr, text)
    return best


def main():
    with open(sys.argv[1]) as f:
        timeline = walk_wrapper(f.read())
    symbols = sorted({s for _, e, _ in timeline for s in e.free_symbols}, key=str)
    print(f"{len(timeline)} allocations, symbols: {symbols}")
    for num_tokens in (int(x) for x in sys.argv[2:]):
        value, file_line, expr, text = peak_at(timeline, num_tokens)
        print(
            f"T={num_tokens:6d}: peak {value / 2**20:8.1f} MiB = {expr}\n"
            f"           at line {file_line}: {text[:100]}"
        )


if __name__ == "__main__":
    main()
