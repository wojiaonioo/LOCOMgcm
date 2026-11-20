"""PyTorch 2.9 compiler compatibility; installed before runner compilation.

See docs/COMPILE_COMPAT.md for upstream functions, scope, and upgrade checks.
The patches live for the process lifetime, including deferred AOT backward.
"""

from copy import copy
from functools import wraps
from importlib.metadata import version
from pathlib import Path
import re
from threading import Lock

_lock = Lock()
_installed = False


def install_compile_compat() -> None:
    """Install the reviewed torch 2.9.x / Triton 3.5.x compiler fixes.

    Unsupported versions fail before any patch is installed. Eager execution
    does not need this entry point. Direct users of torch.compile outside the
    runner must call it before compiling LOCOMgcm operations.
    """
    global _installed
    with _lock:
        if _installed:
            return

        import torch

        triton_version = version("triton")
        if not (
            re.fullmatch(r"2\.9\.\d+(?:\+[^\s]+)?", torch.__version__)
            and re.fullmatch(r"3\.5\.\d+(?:\+[^\s]+)?", triton_version)
        ):
            raise RuntimeError(
                "LOCOMgcm compile compatibility requires torch 2.9.x and "
                f"Triton 3.5.x; found torch {torch.__version__}, "
                f"Triton {triton_version}. Review docs/COMPILE_COMPAT.md "
                "before upgrading or removing the patches."
            )

        import sympy
        from torch._dynamo.utils import counters
        from torch._inductor.codegen.wrapper import SubgraphPythonWrapperCodegen
        from torch._functorch._aot_autograd import graph_capture, graph_capture_wrappers
        from torch._functorch._aot_autograd.schemas import (
            MutationType, OutputAliasInfo, OutputType,
        )
        from torch._inductor import ir, lowering
        from torch._inductor.graph import SubgraphLowering
        from torch._inductor.codegen.cpp import CppVecOverrides
        from torch._inductor.codegen.triton import IndexingOptions, TritonKernel
        from torch._inductor.codegen.triton_utils import should_unwrap_unspec_arg
        from torch._inductor.dependencies import MemoryDep, StarDep, WeakDep
        from torch._inductor.scheduler import Scheduler, SchedulerNode
        from torch._inductor.utils import DeferredLineBase
        from torch._inductor.virtualized import V
        from torch.utils._ordered_set import OrderedSet
        from torch.compiler import config as compiler_config

        from defgcm.dynamics import _rounded_product
        from .compile_cache import _configure_compile_worker

        # Use torch's shared FX/AOT/autotune key extension. Include this whole
        # module so adding/removing a patch or editing its helpers changes the
        # identity automatically. Keep the caller's tag and exclude file paths;
        # only torch's existing cache machinery hashes this source text.
        cache_key_tag = repr((
            "defgcm.compile_compat.v1",
            compiler_config.cache_key_tag,
            Path(__file__).read_text(encoding="utf-8"),
            Path(_rounded_product.__file__).read_text(encoding="utf-8"),
        ))

        upstream_subgraph_input = SubgraphPythonWrapperCodegen.codegen_input_symbol_assignment
        upstream_invoke_subgraph = lowering.lowerings[torch.ops.higher_order.invoke_subgraph]
        upstream_finalize = SubgraphLowering.finalize
        upstream_is_unaligned = ir.is_unaligned
        upstream_indexing = TritonKernel.indexing
        upstream_load = TritonKernel.load
        upstream_store = TritonKernel.store
        upstream_codegen_body = TritonKernel.codegen_body
        upstream_masked = CppVecOverrides.masked
        upstream_recompute = SchedulerNode.recompute_size_and_body
        upstream_fusable = Scheduler.fusable_read_and_write
        upstream_functionalized = graph_capture_wrappers.create_functionalized_fn

        @wraps(upstream_subgraph_input)
        def codegen_subgraph_input(self, name, value, bound_vars):
            upstream_subgraph_input(self, name, value, bound_vars)
            if not isinstance(value, ir.TensorBox) or self.partition_signatures is not None:
                return
            for accessor, dimensions in (("size", value.get_size()), ("stride", value.get_stride())):
                for axis, expr in enumerate(dimensions):
                    if not isinstance(expr, sympy.Add):
                        continue
                    offset, symbol = expr.as_coeff_Add()
                    if not isinstance(offset, sympy.Integer) or not isinstance(symbol, sympy.Symbol) or symbol in bound_vars:
                        continue
                    # Let upstream use an unshifted extent or scalar whenever
                    # the subgraph has one, even on a later tensor argument.
                    for argument in self.get_graph_inputs().values():
                        if isinstance(argument, ir.TensorBox):
                            if symbol in argument.get_size() or symbol in argument.get_stride():
                                break
                        elif argument == symbol:
                            break
                    else:
                        # The observed metadata is exactly symbol + integer.
                        # Its inverse is valid for every permitted invocation.
                        self.prefix.writeline(f"{symbol} = {name}.{accessor}({axis}) - ({offset})")
                        bound_vars.add(symbol)

        @wraps(upstream_invoke_subgraph)
        def invoke_subgraph(*args, **kwargs):
            outputs = upstream_invoke_subgraph(*args, **kwargs)
            # AOT forward regions can also return sizes saved for backward.
            # Their parent consumes symbolic expressions, not buffer wrappers.
            return [output.expr if isinstance(output, ir.ShapeAsConstantBuffer) else output
                    for output in outputs]

        @wraps(upstream_finalize)
        def finalize_subgraph(self):
            node = self.parent.current_node
            if node is not None and node.target is torch.ops.higher_order.invoke_subgraph:
                # Dynamo requires distinct tensor storage at this boundary.
                # An inner optimization can turn a fresh result into an input
                # or sibling view, even when the frontend saw no alias.
                reserved = set(self.graph_input_names).union(self.constants)
                expected_outputs = node.meta["val"]
                if not isinstance(expected_outputs, (tuple, list)):
                    expected_outputs = (expected_outputs,)
                for index, (output, expected) in enumerate(zip(self.graph_outputs, expected_outputs, strict=True)):
                    if isinstance(output, (ir.NoneAsConstantBuffer, ir.ShapeAsConstantBuffer)):
                        continue
                    aliases = set()
                    pending = [output.get_name()]
                    while pending:
                        name = pending.pop()
                        if name not in aliases:
                            aliases.add(name)
                            buffer = self.try_get_buffer(name)
                            if buffer is not None:
                                pending.extend(buffer.get_inputs_that_alias_output())
                    # The caller still has the declared layout. Inner view
                    # folding may already have changed the result's offset.
                    sizes = output.get_size()
                    strides = [s.node.expr if isinstance(s, torch.SymInt) else s for s in expected.stride()]
                    offset = expected.storage_offset()
                    if isinstance(offset, torch.SymInt):
                        offset = offset.node.expr
                    if reserved.isdisjoint(aliases) and output.get_layout().offset == offset and list(output.get_stride()) == strides:
                        reserved.update(aliases)
                        continue
                    layout = ir.FixedLayout(output.get_device(), output.get_dtype(), sizes, strides, offset)
                    storage = lowering.empty_strided(
                        [layout.storage_size()], [1], dtype=layout.dtype, device=layout.device
                    )
                    view = lowering.as_strided(storage, sizes, strides, offset)
                    lowering.copy_(view, ir.TensorBox(output))
                    self.graph_outputs[index] = ir.ExternKernel.realize_input(view)
            return upstream_finalize(self)

        @wraps(upstream_is_unaligned)
        def is_unaligned(node):
            # ConcatKernel can place a producer directly into a slice of its
            # output. Fallbacks such as aten.view.dtype inherit that address.
            return upstream_is_unaligned(node) or (
                isinstance(node, ir.Buffer)
                and isinstance(node.layout, ir.NonOwningLayout)
                and is_unaligned(node.layout.view)
            )

        @wraps(upstream_indexing)
        def indexing(self, index, **kwargs):
            options = upstream_indexing(self, index, **kwargs)
            if (
                isinstance(options, IndexingOptions)
                and not options.index.free_symbols
                and options.index.is_integer is True
                and options.index_str.startswith("tl.broadcast_to(")
            ):
                # Nested Identity constants survive prepare_indexing. Keep
                # the original masks/shape, but give Triton a block tensor.
                copy_shape = kwargs.get("copy_shape")
                shape = f"{copy_shape}.shape" if copy_shape else self.dense_size_str()
                options.index_str = (
                    f"tl.full({shape}, {self.index_to_str(options.index)}, "
                    f"{self.index_dtype})"
                )
            return options

        def record_size(kernel, access, name):
            # Element sizes of floating and integer (not bool) memory accesses.
            dtype = V.graph.get_dtype(name)
            if dtype != torch.bool:
                sizes = kernel.__dict__.setdefault("_defgcm_access_sizes", {})
                sizes.setdefault((access, dtype.is_floating_point), set()).add(dtype.itemsize)

        @wraps(upstream_load)
        def load(self, name, index):
            result = upstream_load(self, name, index)
            if not should_unwrap_unspec_arg(name):  # a memory load, not a scalar argument
                record_size(self, "load", name)
            return result

        @wraps(upstream_store)
        def store(self, name, index, value, mode=None):
            start = len(self.stores._lines)
            upstream_store(self, name, index, value, mode)
            added = self.stores._lines[start:]
            record_size(self, "store", name)
            # Remember the output pointer of a single plain store line;
            # barrier-guarded, atomic and cooperative stores are not recorded.
            plain = self.__dict__.setdefault("_defgcm_plain_stores", {})
            if mode is None and len(added) == 1:
                plain[id(added[0])] = self.args.output(name)

        @wraps(upstream_codegen_body)
        def codegen_body(self):
            plain = self.__dict__.pop("_defgcm_plain_stores", {})
            sizes = self.__dict__.pop("_defgcm_access_sizes", {})
            widened = any(
                min(sizes[("load", floating)]) < max(sizes[("store", floating)])
                for floating in (True, False)
                if ("load", floating) in sizes and ("store", floating) in sizes
            )
            stores, compute = self.stores._lines, self.compute._lines
            if (
                type(self) is TritonKernel
                and not self.features.is_reduction()
                and [tree.prefix for tree in self.range_trees if not tree.is_reduction] == ["x"]
                and widened
                and stores
                and all(id(line) in plain for line in stores)
            ):
                # Triton 3.5 remove-layout-conversions can reuse the layout
                # conversion in front of a trailing store for an earlier user
                # of the stored value, which it does not dominate. The
                # reproductions fail only when the kernel loads a narrower
                # floating (integer) element than it stores (f16/f32/i32 to
                # f32/f64/i64); equal widths, bool accesses and integer index
                # loads in floating kernels compile. Place each store directly
                # after the last definition it reads (normally its value)
                # instead. Only the store moves; it stays last if its pointer
                # is used again later in the body or is stored more than once,
                # or if it reads an indented definition.
                texts = [line if isinstance(line, str) else getattr(line, "line", None)
                         for line in compute]
                defined = {}
                for position, text in enumerate(texts):
                    if isinstance(text, str) and " = " in text:
                        targets = text.split(" = ", 1)[0]
                        indented = targets != targets.lstrip()
                        for target in targets.strip().split(", "):
                            defined[target] = None if indented else position
                pointers = [plain[id(line)] for line in stores]
                after, kept = {}, []
                for line, pointer in zip(stores, pointers):
                    text = line if isinstance(line, str) else line.line
                    reads = [defined[name] for name in re.findall(r"[A-Za-z_]\w*", text)
                             if name in defined]
                    # -1: everything it reads is defined before the compute lines.
                    position = -1 if not reads or None in reads else max(reads)
                    if None in reads or pointers.count(pointer) > 1 or any(
                        isinstance(later, str) and re.search(rf"\b{pointer}\b", later)
                        for later in texts[position + 1:]
                    ):
                        kept.append(line)
                    else:
                        after.setdefault(position, []).append(line)
                reordered = after.pop(-1, [])
                for position, line in enumerate(compute):
                    reordered.append(line)
                    reordered.extend(after.get(position, ()))
                # Readout for upgrade checks: kernels generated in this process
                # whose emitted line order differs from upstream's. A deferred
                # line of a removed buffer emits nothing and is not compared.
                def emitted(lines):
                    return [id(line) for line in lines
                            if not isinstance(line, DeferredLineBase) or line() is not None]

                if emitted(reordered + kept) != emitted(compute + stores):
                    counters["inductor"]["defgcm_moved_store_kernels"] += 1
                compute[:] = reordered
                stores[:] = kept
            return upstream_codegen_body(self)

        @wraps(upstream_masked)
        def masked(mask, body, other):
            outer_mask = V.kernel._load_mask
            result = upstream_masked(mask, body, other)
            # masked() emits vectors under a merged vector mask, even when
            # its explicit mask and body are scalar. Upstream only propagates
            # those explicit inputs. Correct the type so the existing where
            # wrapper broadcasts scalar operands before selecting lanes.
            if outer_mask is not None and outer_mask.is_vec and not result.is_vec:
                result.update_on_args("masked", (outer_mask,), {})
            return result

        @wraps(upstream_recompute)
        def recompute_size_and_body(self, *args, **kwargs):
            # CPU compatible-range fusion and loop splitting rebuild the body.
            # _compute_attrs() drops mutation ordering and reads the original
            # buffer names. Preserve both, as refresh_dependencies() does:
            # otherwise a fused writer can precede its target's allocation or
            # overwrite storage before an earlier reader has consumed it.
            ordering = OrderedSet(
                dep for dep in self.read_writes.reads
                if isinstance(dep, (StarDep, WeakDep))
            )
            result = upstream_recompute(self, *args, **kwargs)
            self.set_read_writes(
                self.read_writes.with_read(ordering).rename(self.mutation_renames)
            )
            return result

        @wraps(upstream_fusable)
        def fusable_read_and_write(self, read, write):
            # A reduction store does not produce the final value of a buffer
            # for a fused load: codegen cannot forward it via the store cache.
            # Matching indices alone would let CPU loads precede the update
            # (and cannot synchronize a GPU scatter with its consumers).
            # Keep same-mode StarDep ordering between reduction writers;
            # only a value-reading consumer must run after the reduction.
            if isinstance(read, MemoryDep) and write.mode is not None:
                return False
            return upstream_fusable(self, read, write)

        @wraps(upstream_functionalized)
        def create_functionalized_fn(fn, args, args_descs, *, meta, aot_config, **kwargs):
            prefix = meta.num_mutated_inp_runtime_indices
            if (
                aot_config.keep_inference_input_mutations
                and prefix
                and any(
                    info.output_type == OutputType.is_input
                    and meta.input_info[info.base_idx].mutation_type
                    == MutationType.MUTATED_IN_GRAPH
                    for info in meta.output_info
                )
            ):
                # This helper alone indexes the flattened forward outputs with
                # user-output indices, forgetting the runtime-mutation prefix.
                # Give its alias-substitution loop a private, padded index view.
                # It only reads output_info in that loop; fn and all downstream
                # consumers retain the original metadata and output descriptors.
                meta = copy(meta)
                skip = OutputAliasInfo(OutputType.non_alias, torch.Tensor, None, None, False)
                meta.output_info = [skip] * prefix + meta.output_info
            return upstream_functionalized(
                fn, args, args_descs, meta=meta, aot_config=aot_config, **kwargs
            )

        _rounded_product.install_product_lowering()
        _configure_compile_worker(cache_key_tag)
        compiler_config.cache_key_tag = cache_key_tag
        SubgraphPythonWrapperCodegen.codegen_input_symbol_assignment = codegen_subgraph_input
        lowering.lowerings[torch.ops.higher_order.invoke_subgraph] = invoke_subgraph
        SubgraphLowering.finalize = finalize_subgraph
        ir.is_unaligned = is_unaligned
        TritonKernel.indexing = indexing
        TritonKernel.load = load
        TritonKernel.store = store
        TritonKernel.codegen_body = codegen_body
        CppVecOverrides.masked = staticmethod(masked)
        SchedulerNode.recompute_size_and_body = recompute_size_and_body
        Scheduler.fusable_read_and_write = fusable_read_and_write
        graph_capture_wrappers.create_functionalized_fn = create_functionalized_fn
        graph_capture.create_functionalized_fn = create_functionalized_fn
        _installed = True
