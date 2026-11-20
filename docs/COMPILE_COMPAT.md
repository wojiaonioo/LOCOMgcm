# PyTorch compile compatibility

`defgcm.accel.compile_compat.install_compile_compat()` installs process-local
AOTAutograd, scheduling and code generation fixes for PyTorch 2.9.x with Triton 3.5.x. Other version pairs raise
an error before installation. No installed package files are changed.

`runner._configure_step_execution` calls the entry point before either model-step
or tendency compilation. Both ModelRunner and the case generator's worker and
differentiation paths use this context. Before installing anything it calls
`runner.refuse_dynamic_derivative`, which refuses `dynamic=True`
(`DEFGCM_COMPILE_DYNAMIC=1`) when the caller states a vjp or jvp derivative
intent (`derivative_mode`; an unstated intent counts as vjp). ModelRunner.run
calls it at its entry with the intent of `track_gradients`, and the checkpoint
window calls it at its entry (`require_differentiable_window`) for the step it
differentiates, so runner compilation with fully symbolic shapes is forward
only. The case generator's checkpoint-window paths state `derivative_mode="vjp"`
and call the refusal before communication admission and extension preload; its
array holds no compiled differentiated row with `compile_dynamic=1`, and its
runner records such a row as not executable without running it.

ModelRunner.run also calls `runner.refuse_strict_plain_reverse` at its entry.
It refuses `fullgraph=True` (`DEFGCM_COMPILE_FULLGRAPH=1`) with a compile target
when the run is reverse mode (`track_gradients`) on more than one rank, because
ModelRunner.run differentiates without a checkpoint window. Without a window the
communication session runs each halo round, sum and solve eagerly at a graph
exit (`comm_adjoint` `_raw_boundary`), which a compiled step does not admit. Under `error_on_graph_break` such an exit inside an
inlined frame makes Dynamo run the calling frames eagerly with its frame
callback still set, and the first generator frame it meets fails with
"Attempt to trace generator" (case-generator row a0229, 8 ranks, at step 3, the
first traced step; two ranks reproduce it, one rank has no session). Inside a
checkpoint window the session's rounds are an in-graph operator, so windowed
reverse runs and forward runs on several ranks are not refused. The two ways
out are a checkpoint window or `DEFGCM_COMPILE_FULLGRAPH=0`. The case
generator's constraint table carries the same refusal (`run_config`), so its
array holds no such row and its runner records a hand-written one as not
executable without running it. Installation is idempotent and persists
after the context exits: AOTAutograd can compile a backward graph later. Calls to
`torch.compile` made directly outside the runner must first call this entry point.
Eager runner execution does not install the patches.

## Storage of nested graph results

Dynamo requires each tensor returned by `invoke_subgraph` to have independent
storage. In PyTorch 2.9, optimizing a one-layer stack or a redundant copy
can turn that result into a view of an argument or another return value.
The parent may reuse the argument for a different field while a consumer
still reads the result. In the one-layer hydrostatic step, the surface
pressure write then changes the baroclinic pressure used by both gradients.

At `SubgraphLowering.finalize`, the compatibility layer resolves the storage
of each tensor result and checks the layout declared by the calling graph.
It materializes a result when its storage is shared with an input, constant
or previous result, or its layout conflicts with the caller. The copy uses
the caller's strides and storage offset. Independent results with the
declared layout keep their usual lowering.

When upgrading, exercise both consumers of the one-layer center potential.
Write to each public result after the call and check its inputs and siblings.
Include a temporary returned twice through distinct allocating expressions,
a genuine offset view of an internal allocation, dynamic columns and
forward and reverse derivatives.

## Shifted dimensions in a nested graph

During dynamic model-step compilation, a staggered face can have width `s + 1`.
A nested Thomas solve can receive four tensors with that width while the center
tensor owning `s` stays in the caller. PyTorch 2.9's
`SubgraphPythonWrapperCodegen.codegen_input_symbol_assignment` binds unshifted
dimensions. Its inherited input verifier then rejects the face size.

The wrapper checks for an existing scalar, tensor size or tensor stride that
binds the center symbol directly. In their absence it can obtain one symbol
from a tensor size or stride equal to that symbol plus an integer offset.
It emits the inverse integer expression in the nested Python wrapper and
leaves validation of the full shape to Inductor. Upgrade checks should compile
the staggered Thomas solve on two widths and heights, differentiate all floating
inputs and exercise a subgraph with two unresolved symbols in one dimension.

## Symbolic results of a nested graph

Two compilation stages combine in this failure. When AOTAutograd partitions a
nested region (`run_joint_graph_passes_on_hops`), the region's forward also
returns the symbolic sizes its backward needs. Before the outer graph is
partitioned, Inductor's joint-graph pass `constant_fold_uniform_value`
rebuilds each uniform-valued tensor as an `aten.full.default` node. It takes
each symbolic size from the last graph node whose value is that symbol, so a
rebuilt factory after the region uses the region's saved-size output. This
stage only chooses between FX nodes of equal value.

The error occurs later, when Inductor lowers the parent graph. In PyTorch 2.9
the nested graph wraps a symbolic output in `ShapeAsConstantBuffer`
(`ExternKernel.realize_input`), and neither `InvokeSubgraph.create` nor the
`invoke_subgraph` lowering removes the wrapper. The `aten.full` lowering
(`tensor_constructor`) calls `sympy.expand` on each size and raises
`SympifyError` on the wrapper. Elsewhere the parent graph represents symbolic
values as sympy expressions.

In the failing gradient-bearing model step, the hydrostatic
center-geopotential region returns its width `s77` with its tensor. Fourteen
rebuilt `aten.full` nodes, and no slice, use the returned width. They include
the tracer's `zeros_like(state.temp.data)` and the zero bottom row of
`vertical_velocity_from_transport_divergence`, neither of which reads the
region's result. Traced `aten.full` nodes in the same graph keep the parent's
own `s77` input. Taking a size from another tensor in Python therefore does
not change the binding.

The compatibility layer unwraps symbolic results to their existing `expr` at
the `invoke_subgraph` lowering boundary. Tensor and other results retain the
upstream lowering. The region, saved-size dependence and numerical operations
are preserved. This is separate from binding shifted input dimensions above.
Specializing the width would replace this symbolic axis with a shape guard;
changing the region or its partitioning would affect the numerical compilation
boundary and every saved-size consumer; disabling the constant folding would
change every compiled graph. The scalar-result conversion belongs at the
common lowering boundary.

For an upgrade, first review the other compatibility patches. Then work in a
disposable copy of the model tree. The version gate at the start of
`install_compile_compat()` raises before any patch is installed, so in the copy
widen its two version patterns to admit the candidate PyTorch and Triton. The
edit also changes the copy's cache tag.

The following CPU reproduction has the same structure as the failing step. A
region stacks per-layer sums, a `zeros_like` of the same width receives one
layer, and the function is compiled with `dynamic=True` and differentiated.
Run each arm in its own process, since the patches are process-wide: `compat`
installs all patches, `bypass` then removes only this wrapper, once, and
`nofold` also disables `joint_graph_constant_folding`. The script lists each
symbolic region result in the lowered graphs with its consumers, which shows
whether the arm reached this path.

```python
import operator
import sys

import torch
from torch._inductor import config as inductor_config, lowering
from torch._inductor.graph import GraphLowering
from defgcm.accel.compile_compat import install_compile_compat

arm = sys.argv[1]  # compat | bypass | nofold, one process per arm
torch.manual_seed(0)
inductor_config.force_disable_caches = True
install_compile_compat()
if arm in ("bypass", "nofold"):
    op = torch.ops.higher_order.invoke_subgraph
    lowering.lowerings[op] = lowering.lowerings[op].__wrapped__  # once
if arm == "nofold":
    inductor_config.joint_graph_constant_folding = False

shape_results = []
original_init = GraphLowering.__init__


def record(self, gm, *args, **kwargs):
    original_init(self, gm, *args, **kwargs)
    for node in gm.graph.nodes:
        if (node.target is operator.getitem
                and node.args[0].target is torch.ops.higher_order.invoke_subgraph
                and isinstance(node.meta.get("val"), torch.SymInt)):
            shape_results.append((node.name, sorted(str(u.target) for u in node.users)))


GraphLowering.__init__ = record


@torch.compiler.nested_compile_region
def region(upper, lower):
    return torch.stack([upper[:, k] + lower[:, k] for k in range(3)], dim=1)


def step(upper, lower, temp):
    center = region(upper * 2.0, lower * 3.0)
    tendency = torch.zeros_like(temp)
    tendency[:, 0] = center[:, 0] * temp[:, 0]
    return (center * temp).sum() + (tendency * temp).sum()


inputs = [torch.randn(1, 3, 5, dtype=torch.float64, requires_grad=True) for _ in range(3)]
try:
    value = torch.compile(step, dynamic=True)(*inputs)
    grads = torch.autograd.grad(value, inputs)
    reference = step(*inputs)
    expected = torch.autograd.grad(reference, inputs)
    diff = max([float((value - reference).detach().abs())]
               + [float((g - e).abs().max()) for g, e in zip(grads, expected)])
    outcome = f"VJP completed, max |compiled - eager| = {diff:.3e}"
except Exception as error:  # report the compiler error of this arm
    outcome = " ".join(str(error).splitlines()[:2])[:160]
print(arm, outcome)
print("  symbolic region results and their consumers:", shape_results)
```

Save it as `shape_result_repro.py` and run `python shape_result_repro.py compat`,
`bypass` and `nofold` with the model root on `PYTHONPATH`. With PyTorch 2.9.0
each arm lists one symbolic region result. In `compat` and `bypass` its
consumers include `aten.full.default`; `compat` completes the VJP, 3.6e-15 from
eager in float64, and `bypass` fails with the `SympifyError` above. In `nofold`
the factory keeps the parent's size input, the result is consumed only as a
graph output, and the VJP completes without this wrapper.

Production rows run through the case generator. `bin/casegen` starts its
coordinator as `$DEFGCM_PYTHON -m experiments.casegen.cli`, and the
coordinator launches a single-rank row as
`$DEFGCM_PYTHON -m experiments.casegen.run --job <file>` (`_launch_command` in
`experiments/casegen/run.py`; `sys.executable` when the variable is unset).
Point `DEFGCM_PYTHON` at a shell script that runs the driver below with the
to-torch interpreter, and set `UPGRADE_PROBE_MODE` (`bypass` or `record`) and
`UPGRADE_PROBE_DIR`. The coordinator drops `DEFGCM_*` variables outside
`INHERITED_ENV` from the worker environment, so these use another prefix.

```python
import json
import operator
import os
import runpy
import sys

assert sys.argv[1] == "-m", sys.argv
module = sys.argv[2]
sys.path.insert(0, os.getcwd())  # as "python -m" does
if module == "experiments.casegen.run":
    import defgcm.accel.compile_compat as compat
    import defgcm.runner as runner

    original_install = compat.install_compile_compat
    mode = os.environ["UPGRADE_PROBE_MODE"]  # bypass | record
    os.makedirs(os.environ["UPGRADE_PROBE_DIR"], exist_ok=True)
    log = os.path.join(os.environ["UPGRADE_PROBE_DIR"], f"invoke_subgraph.{os.getpid()}.jsonl")
    state = {"done": False}

    def install():
        original_install()
        if state["done"]:
            return  # later runner entries keep the probe lowering and tag
        import torch
        from torch._inductor import ir, lowering
        from torch._inductor.virtualized import V
        from torch.compiler import config
        from defgcm.accel.compile_cache import _configure_compile_worker

        op = torch.ops.higher_order.invoke_subgraph
        lower = lowering.lowerings[op]
        if mode == "bypass":
            lower = lower.__wrapped__  # torch's registered lowering, once

        def recorded(subgraph, identifier, *operands):
            result = lower(subgraph, identifier, *operands)
            consumers = {item.args[1]: sorted(str(user.target) for user in item.users)
                         for item in V.graph.current_node.users if item.target is operator.getitem}
            outputs = [dict(subgraph_output=type(raw).__name__, returned=type(out).__name__,
                            expr=str(raw.expr) if isinstance(raw, ir.ShapeAsConstantBuffer) else None,
                            consumers=consumers.get(index))
                       for index, (raw, out) in enumerate(zip(subgraph.graph.graph_outputs, result))]
            with open(log, "a") as stream:
                stream.write(json.dumps(dict(graph=V.graph.graph_id, backward=V.graph.is_backward,
                                             subgraph=subgraph.name, outputs=outputs)) + "\n")
            return result

        lowering.lowerings[op] = recorded
        config.cache_key_tag = repr((config.cache_key_tag, "upgrade-probe", mode))
        _configure_compile_worker(config.cache_key_tag)
        state["done"] = True

    compat.install_compile_compat = runner.install_compile_compat = install
sys.argv = [module, *sys.argv[3:]]
runpy.run_module(module, run_name="__main__", alter_sys=True)
```

The driver puts the working directory on `sys.path`, as `python -m` does, and
runs every module other than the worker unchanged. In a worker it replaces
`install_compile_compat` in `defgcm.accel.compile_compat` and in
`defgcm.runner`, which bound the name when `defgcm` was imported. The runner
calls the installation each time it enters a compiled execution context, and
after the first call the original returns at once. The replacement therefore
applies its setup once and keeps it: a second `.__wrapped__` would also remove
torch's own registration wrapper. Both modes write one record per
`invoke_subgraph` lowering, with the output types of the nested graph, the types
returned to the parent and the parent's consumers. Both extend the cache tag so
that cached graphs cannot skip the lowering; use a distinct cache pool as well.

This check used to run `a0197` in both modes with its dynamic `model_step` and
`vjp_window` settings. `refuse_dynamic_derivative` now refuses `dynamic=True`
for a differentiated run, and a0197 compiles with static shapes
(`compile_dynamic=0`), so no case-generator row reaches this conversion through
an explicitly dynamic trace. The conversion is still reached with gradients
under torch's automatic-dynamic posture (`DEFGCM_COMPILE_DYNAMIC_AUTO=1`,
`dynamic=None`), which the runner keeps: the first trace is static, and after
an input width changes the recompiled region returns its width as a symbolic
result. To check it, compile the CPU reproduction above with `dynamic=None`,
call it with width 5 and then width 7, and take the VJP after each call. With
PyTorch 2.9.0 the second call returns one symbolic result whose consumers
include `aten.full.default`; with all patches installed the conversion returns
it as a sympy symbol and the VJP completes, and with only this wrapper bypassed
the same consumer raises the `SympifyError` on `ShapeAsConstantBuffer` above.
With `dynamic=False` and a VJP, or with `dynamic=True` without gradients, the
same function lowers no symbolic region result. ModelRunner.run also refuses
forward-mode derivatives whenever a compile target is set (`forward_mode_active`
and `_refuse_unqualified_forward_mode` in `runner.py`), so only reverse mode
reaches this conversion through the runner. Run `a0178` as the CPU static compiled-window
control: its first trace is static, so no nested graph returns a symbolic size,
and it checks installation and the cache identity rather than this conversion.

On the candidate version, a passing `bypass` run counts only if its records
show a symbolic nested-graph output with the original factory consumers;
without such a record the run did not reach this path. Remove this local
wrapper only when those consumers and the complete VJP work without it and the
native reference checks pass. If the saved size still reaches an unsupported
consumer as a buffer wrapper, retain the fix. Other patches need their own
upgrade decisions.

## Storage alignment of concatenation slices

Inductor can write a producer directly into its slice of a concatenated tensor.
In PyTorch 2.9.x, `torch._inductor.ir.is_unaligned` checks a buffer's registered
alignment but misses the backing view of a `NonOwningLayout`. For example, a
one-element offset in float32 places the producer four bytes past the aligned
base. A byte view of that producer can then fail the generated 16-byte alignment
assertion during packed halo exchange.

The wrapper follows the backing view and applies the existing alignment check.
When upgrading, test `aten.view.dtype` on a producer shared by concatenation
and byte packing, with one-element and four-element padding in float32. Check
the packed bytes and the gradient of the floating-point output in each case.

## Triton constant index broadcast

Upstream function: `torch._inductor.codegen.triton.TritonKernel.indexing`.
In PyTorch 2.9.0, its `sympy.Integer` case emits `tl.full`, but other expressions
that need a dense index emit `tl.broadcast_to`. A reversed 26-layer column fused
with concatenation and a surface select can retain the constant expression
`25 - Identity(0)`, printed as `25 + (-1)*(0)`. Triton 3.5 treats it as a constexpr,
which lacks the block type needed by `broadcast_to`.

The wrapper changes only `IndexingOptions` whose generated expression is a
`tl.broadcast_to` of an integer index with no free symbols. It emits `tl.full`
with the same expression, broadcast shape and kernel index dtype. Masks and all
other indexing metadata are retained. Existing integer `tl.full`, symbolic
indices, block pointers and unbroadcast indices pass through unchanged.
The numerical operator keeps its two flips and the original cumulative sum.

## Triton store placement in pointwise kernels

Upstream functions: `torch._inductor.codegen.triton.TritonKernel.load`,
`TritonKernel.store` and `TritonKernel.codegen_body`. The `load` wrapper
records the element type of each memory load and skips scalar arguments. The
`store` wrapper records the element type of each store and the output pointer
of each plain store line. The `codegen_body` wrapper moves the stores after
all loads, computation and stores of the kernel have been generated, when the
dependencies of each store are known. The defect is in Triton 3.5's
`TritonGPURemoveLayoutConversions` pass (triton-lang/triton#8257, fixed by
triton-lang/triton#9020 after the 3.5 releases). The pass records each layout
conversion it keeps as a reusable rematerialization of its source value.
While building a slice it reuses only records that dominate the use, but
`rewriteSlice` substitutes any recorded value and inserts the cloned operations
before the originals. An earlier operation can then use a conversion that is
defined later, and the verifier stops `make_ttgir` with `operand #0 does not
dominate this use` (`PassManager::run failed`).

Inductor emits every store of a pointwise kernel after all computation. In a
one-dimensional pointwise kernel all loads share one layout, because Coalesce
gives a load the widest per-thread vector of the kernel's memory operations.
A store of a different width keeps a conversion directly before it, at the end
of the kernel. If the stored value also feeds later computation, that
conversion is recorded and reused for the earlier users. The f32 KPP buoyancy
graph with a VJP has this form: its float64 island loads f32 fields and stores
many float64 intermediates for the backward pass. In `b0279`
(`scs_paper_h160`, f32, `model_step`, `vjp_window`), compiled then with
dynamic shapes, the forward partition of the gradient step failed in this pass.
The row now compiles with static shapes (`compile_dynamic=0`), since the runner
refuses a dynamically compiled derivative; the reproductions below compile the
graphs directly with `torch.compile` and keep the dynamic form. Its pointwise kernel
`triton_poi_fused__to_copy_add_clamp_div_expand_gt_mul_ones_like_rsub_slice_sqrt_sub_view_where_4`
combines `kpp_statekpp` with `kpp_surface_buoyancy_divided_difference`.
The 1024-element, four-warp configuration failed, while the 512-element,
eight-warp configuration compiled.

The wrapper selects the kernel family reproduced here; the condition is not a
necessary condition of the upstream defect. With the reproduction below,
loading f16, f32 or i32 and storing f32, f64 or i64 fails. Equal load and store
widths compile, including strided stores, bool masks that are stored, reused
or loaded, and int32 index loads in floating kernels, widened to int64 or not.
These variants do not cover reductions, multidimensional layouts, in-place
mutation, buffer aliases or arbitrary access orders. The upstream report
includes a reduction kernel.

Element sizes are compared separately for floating and integer accesses; bool
accesses do not count. In a one-dimensional pointwise `TritonKernel` that loads
a narrower floating element than it stores, or a narrower integer element than
it stores, the wrapper moves each store directly after the last computation
line that defines a name the store reads. For a dense store this is the
definition of its value, and a conversion kept for the store then dominates
every other user of the value. A store that reads only loads, arguments and
indices moves to the start of the computation. Only store lines move and every
other line keeps its order. Triton may then choose different layouts and
contract different multiply-add pairs, so results can differ by rounding; the
arithmetic itself is unchanged. A store stays last if it reads a name defined
in an indented block, or if its pointer is stored more than once or appears in
a later computation line. All stores stay last if the kernel has a
barrier-guarded, atomic or cooperative store. Other kernels, including
reductions, tiled and template kernels, keep their generated source unchanged.
If the same defect occurs in such a kernel, the MLIR verifier stops the
compilation in this pass.

Each kernel whose generated line order differs from upstream's, because a
store moved ahead of computation lines or of other stores, increments
`torch._dynamo.utils.counters["inductor"]["defgcm_moved_store_kernels"]` in
the process that generates it. The counter does not change generated code.
An FX graph cache hit does not regenerate kernels and does not count. Processes
that write Inductor's counters, such as the compile-group cache statistics of
`bin/verify`, record it. When the wrapper still installs but its condition no
longer matches, upstream order is kept without an error; the counter shows it.

The existing version gate accepts PyTorch 2.9.x with Triton 3.5.x. Triton 3.6
or later raises before any patch is installed; this wrapper is not switched off
on its own. A Triton 3.5.x release with a backport of triton-lang/triton#9020
still installs the wrapper. Remove it only after the trailing-store
reproduction below compiles on the candidate version.

A local sm_89 GPU compiled four graphs with and without the wrapper, each in
its own process with empty caches, and recorded the configuration each kernel
ran with. With the default configurations, the f32 dynamic STATEKPP VJP, alone
and with the divided-difference surface buoyancy, failed in this pass without
the wrapper and compiled with it; two of the wrapped kernels ran with the
1024-element, four-warp configuration. The f32 static forward graph gave
byte-identical outputs with one kernel changed. The f64 dynamic STATEKPP VJP
generated the same 15 kernel sources, and the counter stayed at zero.
Autotuning chose different configurations for two of its kernels, one of them
a reduction, and 9198 values of the temperature gradient differed by up to
1.1e-16. With every kernel restricted to its first configuration in both
builds, all its outputs and gradients were byte-identical. In a controlled
experiment, each kernel with an eight-warp configuration was restricted to it,
which compiles the failing kernels without the wrapper. Both f32 VJPs then ran
the same configurations with and without the wrapper and gave byte-identical
outputs and gradients. All 86 configurations of the 43 distinct pointwise
sources compiled to `cuda:80` TTGIR on a CPU. Stores moved in 17 of these
sources, each with f32 and f64 pointers.

This reproduction runs on a CPU, without a GPU and without Inductor. It compiles
a Triton kernel to TTGIR for `cuda:80`:

```python
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource, make_backend
from triton._C.libtriton import ir


@triton.jit
def kernel(a_ptr, b_ptr, s2_ptr, root_ptr, out_ptr, n, XBLOCK: tl.constexpr,
           EARLY: tl.constexpr):
    x = tl.arange(0, XBLOCK)
    m = x < n
    a = tl.load(a_ptr + x, m)
    b = tl.load(b_ptr + x, m)
    s2 = a.to(tl.float64) * a.to(tl.float64)
    if EARLY:
        tl.store(s2_ptr + x, s2, m)
    root = tl.sqrt(s2)
    if EARLY:
        tl.store(root_ptr + x, root, m)
    out = b.to(tl.float64) * b.to(tl.float64) + root
    if not EARLY:
        tl.store(s2_ptr + x, s2, m)
        tl.store(root_ptr + x, root, m)
    tl.store(out_ptr + x, out, m)


for early in (False, True):
    signature = {"a_ptr": "*fp32", "b_ptr": "*fp32", "s2_ptr": "*fp64", "root_ptr": "*fp64",
                 "out_ptr": "*fp64", "n": "i32", "XBLOCK": "constexpr", "EARLY": "constexpr"}
    attrs = {(i,): [["tt.divisibility", 16]] for i in range(5)}
    src = ASTSource(kernel, signature, {"XBLOCK": 1024, "EARLY": early}, attrs)
    target = GPUTarget("cuda", 80, 32)
    backend = make_backend(target)
    options = backend.parse_options({"num_warps": 4, "num_stages": 1})
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    module = src.make_ir(target, options, backend.get_codegen_implementation(options),
                         backend.get_module_map(), context)
    stages = {}
    backend.add_stages(stages, options, src.language)
    try:
        stages["ttgir"](stages["ttir"](module, {}), {})
        print("early" if early else "trailing", "stores: TTGIR compiled")
    except RuntimeError as error:
        print("early" if early else "trailing", "stores:", error)
```

With Triton 3.5.0, trailing stores print `PassManager::run failed` after the
dominance diagnostic, while early stores compile. Remove the wrapper only
after the trailing form compiles on the candidate Triton version. On a GPU,
also compile the f32 `kpp_statekpp` with `dynamic=True` and a VJP without the
wrapper, and compare its outputs and gradients with the wrapped build. Record
the configuration each kernel ran with in both builds. Judge a difference by
its source and structure: a changed layout or configuration can change
multiply-add contraction and reduction order. `TRITON_DEFAULT_FP_FUSION=0` is
only a diagnostic for contraction.

## C++ nested masked values

Upstream function: `torch._inductor.codegen.cpp.CppVecOverrides.masked`.
`CppKernel.masked` combines the explicit mask with the enclosing load mask.
When the enclosing mask is vector-valued, the generated result is a vector even
if the explicit mask and body are scalar. Upstream propagates only the explicit
mask and body into the result's `CppCSEVariable` type. Consequently a later
`where` may emit a scalar conditional with a vector and a scalar branch, which
C++ rejects. This affects both halo corner assignment and its generated reverse.

The wrapper restores the missing vector metadata only when an enclosing vector
mask produced a result still marked scalar. The existing `CppVecOverrides`
operand promotion and broadcast logic then broadcasts scalar operands and emits
the normal lane selection. There is no new value arithmetic, dtype conversion
policy, mask rule, or vector-width setting. Scalar contexts and correctly marked
vector results are unchanged. Original halo slice assignments are retained.

## Dependencies after CPU loop-body rebuild

Upstream function: `torch._inductor.scheduler.SchedulerNode.recompute_size_and_body`.
When `CppScheduling.fuse` merges nodes with compatible but different iteration
ranges, it rebuilds the loop body. `_compute_attrs` then replaces the node's
dependencies with the reads and writes extracted from the new body. The ordering
dependencies added by `compute_dependencies` for mutations (`StarDep`, `WeakDep`)
and the node's `mutation_renames` are lost. The scheduler can then place a
buffer's initialization after a fused kernel that writes it. The generated
wrapper passes the buffer before it is allocated (UnboundLocalError on a
`bufN`). The cubed-sphere first-order advection with face-strip exchange and
`index_copy_` reproduces it on one 16-by-16 local face.

The wrapper keeps the existing `StarDep` and `WeakDep` across the rebuild, merges
them with the new reads and writes, and restores the mutation renames. This is
the rule upstream already uses in `refresh_dependencies`. The custom op
`defgcm::halo_p2p` keeps its declaration.

## Reduction stores and fused reads

Upstream function: `torch._inductor.scheduler.Scheduler.fusable_read_and_write`.
Its `MemoryDep` branch compares buffer name, index and size only. A store with a
reduction mode such as `atomic_add` therefore matches a later plain read of the
same location, and the two nodes fuse into one loop. Code generation forwards
only plain stores through the store cache, so the fused loop reads the old value
before the accumulation. Gradients of indexed updates (`index_copy_`,
accumulating `index_put_`, scatter) can lose a contribution: a two-update graph
returns half of one input's gradient. The GAD index updates of the cubed sphere
reach this decision.

The wrapper refuses value-dependency fusion between a plain `MemoryDep` read and
a store whose mode is not None. Ordering dependencies between reduction stores of
the same mode, and plain read-after-write fusion, are unchanged.

## AOT returned inputs after runtime mutation outputs

Upstream function:
`torch._functorch._aot_autograd.graph_capture_wrappers.create_functionalized_fn`,
also imported by `graph_capture`. Its nested `_functionalized_f_helper` replaces
functionalized outputs with their original inputs when those inputs were mutated
inside the graph. In PyTorch 2.9.0 it writes `flat_outs[i]` using a user-output
index. The flattened forward outputs start with the inputs whose mutations must
be replayed by the runtime, so the correct slot is
`meta.num_mutated_inp_runtime_indices + i`.

This requires both a nonempty runtime-mutation prefix and a returned input whose
mutation stays inside the graph, with `keep_inference_input_mutations` enabled.
For example, scatter into a gradient-bearing U input and a non-gradient V input,
then return `(salt_zero, temp_zero, V, U, eta_zero)`. The prefix has length one;
the V replacement writes slot 2, which holds temp, instead of slot 3. Different
shapes cause a later shape error; equal shapes silently return V values as temp.
Fresh outputs need not be zeros or duplicates. A metadata mutation can also
create the prefix in a forward-only graph. Changing the output order may hide
the mistake when the overwritten output is itself reconstructed from an input;
placing V first instead corrupts the runtime mutation output.

The wrapper gives this helper a shallow metadata copy with non-alias entries
prepended to `output_info`. This list is used only by its final alias-substitution
loop, so the existing loop now writes the correct slots. The wrapped `fn`, graph
descriptors, partitioner and runtime keep the original metadata. The mutation
prefix, backward outputs and intermediate bases are unchanged. Both imported
references are patched. No tensor operations or model code are replaced.
Without the triggering combination the original function receives the original
metadata object. The patch uses the existing version gate and cache identity.

On upgrade, inspect the two mutation-output builders
(`fn_prepped_for_autograd` and `fn_input_mutations_to_outputs`) and every use of
`meta.output_info` inside `create_functionalized_fn`. If upstream fixes the offset
or uses that list elsewhere, remove or revise this local index view before
enabling the new version. Confirm the previously failing output order without
the wrapper, including equal-shaped fields and more than one prefix output.

## Cache identity

Installation extends `torch.compiler.config.cache_key_tag` with the caller's
existing tag, the `defgcm.compile_compat.v1` marker, and the complete UTF-8
sources of `compile_compat.py` and `dynamics/_rounded_product.py`. The latter
registers the separately rounded product used by MDJWF, KPP enhancement and
the Smagorinsky strain norm. NVIDIA kernels issue a round-to-nearest multiply
for matching binary32 or binary64 operands at those sites.
PyTorch 2.9 includes the tag in `FxGraphHashDetails`.
`AOTAutogradCacheDetails` inherits those fields, including for saved backward
graphs. Both local and remote graph caches therefore distinguish an unpatched
process, the installed patch set and subsequent edits. Source paths and file
timestamps are excluded, so copies of the same source use the same identity.
Editing either file changes the tag automatically. Source must be available
at installation. Even comment-only edits conservatively cause misses.

The autotune key includes this tag as well as the generated kernel source.
Two revisions can therefore share a physical pool and identical generated
code while retaining different last launch choices under different keys.
A bitwise comparison must check those actual choices. A diagnostic can reuse
an already measured choice for the same kernel and configuration space to
isolate reduction order while keeping both graph identities intact. In a0197,
two revisions that differed only in this file kept different choices for the
U transport reduction, and the surface height differed at one fixed wet point
in each of five steps. Appending the other revision's measured choice for that
one kernel removed every difference in the state and masks.

The pressure supplied to the EOS has a separate storage boundary. With nested
regions enabled, `equation_of_state.py` publishes the one-dimensional static
reference pressure or reference geopotential column before its consumer runs.
The dynamic branch adds the previous total hydrostatic potential to the
rounded reference geopotential, then multiplies by the reference density.

The source is passed as text. LOCOMgcm adds no digest calculation or file
manifest: PyTorch's existing cache-key hashing consumes the tag. This uses the
dedicated compiler key extension rather than adding a no-op graph pass or
marking functions as unsafe-cacheable. Installation is idempotent and leaves
the tag active for deferred backward compilation. Install before compilation;
changing a disk-cache key does not replace already-returned compiled callables.

Per-kernel local and remote autotune keys also include `cache_key_tag`. The
Triton worker task explicitly carries the submitting process's tag, including
when casegen has already started its worker pool. It applies that tag while
loading and compiling the generated kernel; workers do not install scheduling
patches. A new process which only imports LOCOMgcm or uses the eager runner
keeps its original tag. Direct `torch.compile` users who omit installation keep
the unpatched namespace and its results.

The casegen compile pool has no independent graph key: its publication wrappers
use PyTorch's target names and Triton's cache manager keys. Thus FX/AOT entries
and autotune snapshots are published under the extended keys in the same shared
root. Python/C++ code, Triton binaries and template timing choices already use
generated-source keys (plus compiler/device/options where applicable). Identical
generated kernels can still be reused; changed scheduling or code generation
cannot recover old graph code through these source-keyed layers. Bundled
autotune caches key on the generated wrapper and backend, and contain the
tag-specific individual entry names; an old bundle cannot satisfy a lookup for
a different tag. Existing shared entries remain available to their original
identity, and need no deletion or migration. Both casegen run and build workers
reach installation through the runner, regardless of their cache-root setup.

## Compiled checkpoint continuation

The compiled-resource conventions restore observed geometry caches and regional
publication slots. They do not persist Inductor's pointwise launch selections.
The same generated kernel can acquire different launch configurations within a
shared pool, with different floating-point results. Strict bitwise continuation
checks must control the actual choices in both legs as well as source, state
and configuration. The verification pin helper covers reductions and scans;
persisting and restoring every pointwise choice across a cold compilation is
not implemented.

The graph partition of a continued run's first step also depends on which step
constants exist when it is traced. The implicit free-surface stepper builds them
in its eager prepare (`steppers/regional_cold_start.py`
`_prewarm_step_constants`) for every run except one with an explicit carry or an
active nonlinear free surface (`_constant_prewarm_applies`). A Stevens-boundary
run used to be excluded too, together with the cross-step seed, which it still
skips: its sponge tables and gates, the surface tracer area and the assembly
split memo were built in its first traced step, so a continuation compiled that
cold variant at its first step, a different partition from the uninterrupted
run's later steps, and its float32 rounding differed (case-generator row b0279).
Runs that the prewarm still excludes build their constants in the first traced
step and keep that difference.

## Host sampling in compiled steps

The implicit free-surface stepper samples its clocks and boundary records in
the eager `prepare_step_host`. AB2, SSPRK3 and split-explicit sample inside
their own `step`, which is the traced kernel under `compile=model_step`. Three
host operations there are opaque to Dynamo: `TorchRegionalOceanModel.prepare_step_time`
and `_compute_balanced_targets` of the momentum sponge and of
`Prescribed3DBoundaryCondition` are `@torch._dynamo.disable`, and
`SplitExplicitStepper._step_impl` is `@torch.compiler.disable(recursive=False)`,
so the tensor functions it calls still compile as their own frames. Traced, the
samplers below `prepare_step_time` were guarded on the per-call Python clock
(`BoundaryBundleSource._time_bracket`, `ERA5WindSource.sample`), the balanced
target memo on the `id()` of each new sample, and the split step on its clock
under static shapes, so each new clock recompiled those frames until the
recompile limit (case-generator rows b0033 and b0319). After the change the
recompiles are confined to the first steps: the initial state is an ordinary
tensor and later states are inference tensors, which changes the dispatch keys
once.

The ERA5 clock slot (`io/era5.py` `_ModelTimeSlot.sample`) uploads a float clock
through two persistent buffers written in place, `fill_` on a pinned host
scalar and then `copy_` to the device. A Dynamo capture with an AOT backend
(aot_eager or Inductor) on torch 2.9 drops both writes without an error, with
or without the patches in this file, so a traced sampler read the slot's
previous clock: an SSPRK3 ERA5-wind row compiled with `model_step` sampled
three of its four stage clocks at the step start. While compiling, the slot now
returns `torch.as_tensor(value)` and writes nothing, the same split as
`seaice/dynamics/types.py` `scalar`. AB2, and split-explicit with an AB2 slow
stepper, evaluate tendencies only at the step start, whose sample the eager
path had written, so their states were not affected; the implicit free-surface
steppers sample in eager prepare and were not affected either.

## Upgrade and verification

Before changing the supported versions, inspect the wrapped upstream functions and
`CppCSEVariable.update_on_args`. Check whether upstream now normalizes nested
constant indices and propagates the merged mask's vector type. Remove a wrapper
only after reproducing its previously failing expression without it.

Verification must cover:

1. Original f64 15-layer, 32-by-32 halo strip/corner assignments with one and two
   local faces, plus compiled VJP for both input tensors. Compare output and VJP
   bytes against eager execution.
2. The real Triton indexing generator with `25 - Identity(0)`, dense and copied
   shapes, plus ordinary integer and symbolic controls. CPU text generation
   checks this branch without executing a CUDA kernel; CUDA integration still
   needs the original single-column run.
3. Fresh code generation before and after installation for representative
   pointwise selection, flip/concatenation and vertical continuity graphs.
   Save the full generated sources. Their AOT ID comments change on each
   compilation; compare the remaining text and output bytes exactly.
4. Unsupported-version refusal, no partial installation, repeated installation,
   and installation before both runner compile targets.
5. The one-face 16-by-16 cube advection with face-strip exchange: no buffer used
   before allocation, forward and VJP bytes equal to eager.
6. The two-update indexed graph with an empty and a non-empty index, and its
   extracted reverse subgraph as a forward graph: compiled gradients equal eager.
   Plain read-after-write fusion stays allowed.
7. In one temporary cache root, first populate the forward and backward caches
   without compatibility installation, then install and compile the same graphs
   in another process. Require FX and AOT misses and correct results. Repeat the
   installed set in another process and require hits with identical output/VJP
   bytes. Edit and remove a patch in isolated source copies to verify automatic
   key changes; an identical copy at another path must retain the key.
8. Check `cache_key_tag` in `FxGraphHashDetails`, AOT inheritance, local/remote
   autotune keys and sidecar task serialization after upgrading PyTorch. Verify
   tag forwarding with a worker started before installation, preservation of a
   caller-supplied tag, and no tag changes on unsupported-version refusal.
9. Run the AOT mutation-prefix reproduction of the appendix with the patches
   installed, on the `inductor` and `aot_eager` backends; every case must
   report OK. It covers mixed-gradient mutations, functional scatter controls,
   reordered outputs, equal-shaped fields, a two-output mutation prefix, direct
   AOT metadata mutations and a no-grad warmup followed by grad-mode
   compilation. It compares output and mutated-input bytes, input/output
   identity and VJP bytes against eager, materializing unused gradients as
   zeros in both arms. Run the extra cases, and `base` with dynamic shapes and
   the warmup, for the extended checks. The original production rows also need
   rerunning on their CPU and CUDA distributed execution paths.
10. When upgrading PyTorch or Triton, or editing the store-placement wrapper,
    compile the f32 dynamic STATEKPP VJP with and without the wrapper.
    `defgcm_moved_store_kernels` must be positive with the wrapper. Its
    float64-island `_to_copy` kernels must have stores before computation
    lines, while the unwrapped sources keep all stores last. The f64 dynamic
    STATEKPP VJP must generate the same kernel sources with and without the
    wrapper. When upgrading Triton, also run the store-placement reproduction
    above without the wrapper, and rerun `b0279` on its CUDA path; it now
    compiles with static shapes, so only the direct compilation above covers
    the dynamic graph in which the failure was found.

These checks cover local code generation and the tested inputs. The full CUDA
column and CPU distributed cube cases must also run on their production devices.

## Appendix: AOT mutation-prefix reproduction

This CPU reproduction checks the AOT returned-input patch above. Each case
builds the same tuple of fresh and mutated fields from seeded float64 inputs,
including a negative zero, and compares the compiled call with eager. The core
below holds the cases, inputs, assembled function and checks:

```python
from contextlib import nullcontext

import torch

NAMES = ("salt", "temp", "v", "u", "eta")
BASIC = {
    "base": {},
    "temp_nograd": {"temp_grad": False},
    "out_u": {"mut_u": False},
    "out_v": {"mut_v": False},
    "out_both": {"mut_u": False, "mut_v": False},
    "only_v": {"skip_u": True},
    "all_grad": {"grad_v": True},
    "no_grad_inputs": {"grad_u": False, "temp_grad": False},
    "natural_order": {"order": ("eta", "u", "v", "temp", "salt")},
    "same_shape": {"same_shape": True},
}
EXTRA = {
    "salt_target": {"order": ("salt", "v", "temp", "u", "eta")},
    "v_first": {"order": ("v", "salt", "temp", "u", "eta")},
    "v_grad": {"grad_u": False, "grad_v": True},
    "prefix_two": {"mut_temp": True},
    "metadata_prefix": {"metadata_u": True, "grad_u": False, "temp_grad": False},
    "no_return_v": {"order": ("salt", "temp", "u", "eta")},
    "view_v": {"view_v": True},
    "no_grad_context": {"no_grad": True},
    "distinct_fresh": {"fresh_values": True},
}


def make_inputs(options):
    g = torch.Generator().manual_seed(0)
    leaves = []

    def floating(shape, grad):
        value = torch.randn(shape, dtype=torch.float64, generator=g)
        value.flatten()[0] = -0.0  # torch.equal alone would miss a sign change
        value.requires_grad_(grad)
        if grad:
            leaves.append(value)
        return value.clone() if grad else value

    nz, ny, nx = 4, 3, 5
    su = (1, nz, ny, nx if options.get("same_shape") else nx + 1)
    sv = (1, nz, ny if options.get("same_shape") else ny + 1, nx)
    gu, gv = options.get("grad_u", True), options.get("grad_v", False)
    eta = floating((1, ny, nx), False)
    u, ub = floating(su, gu), floating((1, 1, *su[-2:]), gu)
    v, vb = floating(sv, gv), floating((1, 1, *sv[-2:]), gv)
    kw = torch.randint(0, nz, ub.shape, generator=g)
    ks = torch.randint(0, nz, vb.shape, generator=g)
    temp = floating((1, nz, ny, nx), options.get("temp_grad", True))
    salt = floating((1, nz, ny, nx), False)
    return (eta, u, kw, ub, v, ks, vb, temp, salt), leaves


def make_assemble(options):
    def assemble(eta, u_fill, kw, ub, v_fill, ks, vb, temp, salt):
        e = torch.zeros_like(eta)
        if options.get("metadata_u"):
            u = u_fill.transpose_(-1, -2)
        elif options.get("skip_u"):
            u = u_fill
        elif options.get("mut_u", True):
            u = u_fill.scatter_(-3, kw.expand(ub.shape), ub)
        else:
            u = u_fill.scatter(-3, kw.expand(ub.shape), ub)
        if options.get("mut_v", True):
            v = v_fill.scatter_(-3, ks.expand(vb.shape), vb)
        else:
            v = v_fill.scatter(-3, ks.expand(vb.shape), vb)
        if options.get("view_v"):
            v = v.view(v.shape)
        if options.get("mut_temp"):
            temp.add_(1.0)
        t, s = torch.zeros_like(temp), torch.zeros_like(salt)
        if options.get("fresh_values"):
            t, s = t + 2.0, s + 3.0
        values = {"salt": s, "temp": t, "v": v, "u": u, "eta": e}
        return tuple(values[n] for n in options.get("order", NAMES))
    return assemble


def same_bytes(a, b):
    if a is None or b is None:
        return a is b
    return (a.shape == b.shape and a.dtype == b.dtype
            and torch.equal(a.detach().contiguous().view(torch.uint8),
                            b.detach().contiguous().view(torch.uint8)))


def vjp(outputs, leaves):
    differentiable = [(i, value) for i, value in enumerate(outputs) if value.requires_grad]
    if not differentiable:
        return tuple(torch.zeros_like(leaf) for leaf in leaves)
    return torch.autograd.grad(
        [v for _, v in differentiable], leaves,
        [torch.full_like(v, 2.0 ** i) for i, v in differentiable],
        allow_unused=True, materialize_grads=True)


def run(options, backend, dynamic):
    torch._dynamo.reset()
    assemble = make_assemble(options)
    reference_inputs, reference_leaves = make_inputs(options)
    compiled_inputs, compiled_leaves = make_inputs(options)
    compiled = torch.compile(assemble, backend=backend, fullgraph=True, dynamic=dynamic)
    with torch.no_grad() if options.get("no_grad") else nullcontext():
        reference = assemble(*reference_inputs)
        actual = compiled(*compiled_inputs)
    outputs = all(same_bytes(a, b) for a, b in zip(reference, actual))
    mutations = all(same_bytes(a, b) for a, b in zip(reference_inputs, compiled_inputs))
    aliases = all((ref is ri) == (out is ci)
                  for ref, out in zip(reference, actual)
                  for ri, ci in zip(reference_inputs, compiled_inputs))
    gradients = all(same_bytes(a, b) for a, b in zip(
        vjp(reference, reference_leaves), vjp(actual, compiled_leaves)))
    return dict(outputs=outputs, mutations=mutations, aliases=aliases, vjp=gradients)
```

The driver around it is a command line
`repro_aot.py plain|compat inductor|aot_eager [--group basic|extra] [--case NAME] [--dynamic] [--warmup]`.
`compat` calls `install_compile_compat()` before compiling, and `plain` does
not. The driver sets `torch._inductor.config.compile_threads = 1`, arms a
55-second `signal.alarm`, runs the selected cases (`BASIC` by default, `EXTRA`
with `--group extra`, one case with `--case`), prints for each the four checks
returned by `run`, any exception, and exits with status 1 if any case fails.
`--dynamic` passes `dynamic=True` to `torch.compile`. `--warmup` first calls
the eager and compiled functions under `torch.no_grad()` on fresh inputs and
requires equal bytes. Dynamo rejects `transpose_` on an input, so
`metadata_prefix` compiles `assemble` with `aot_function` and
`keep_inference_input_mutations=True` instead of `torch.compile`: on `inductor`
it uses `compile_fx_inner` as the forward compiler with `select_decomp_table()`
decompositions, and on `aot_eager` a boxed call of the graph module's forward.
With PyTorch 2.9.0 and `plain inductor`, the basic cases `base`, `temp_nograd`
and `same_shape` and the extra cases `salt_target`, `prefix_two` and
`distinct_fresh` return wrong output bytes, and `v_first` fails with a shape
error. With `compat inductor` all of them report OK. These two readings come
from the core with `torch.compile`; `metadata_prefix` runs only through the
`aot_function` path.
