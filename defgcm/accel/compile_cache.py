"""Publish shared compiler artifacts once, without replacing visible names.

The worker adds this directory to PYTHONPATH and imports ``compile_cache``
directly; compiler subprocesses never initialize the defgcm package.
"""

import copy
import functools
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import uuid

from triton.runtime.cache import FileCacheManager


def publish_file(staged, target) -> bool:
    """Link a closed, complete file into place; an existing name wins.

    Both paths must be on the same filesystem. Unsupported hard links and all
    errors other than EEXIST propagate; replacing the target is never a fallback.
    """
    try:
        os.link(staged, target)
    except FileExistsError:
        return False
    return True


def write_once(path_, content, make_dirs=False, encode_utf_8=False) -> bool:
    """Match Inductor's serialization and permissions, with first-writer wins."""
    assert isinstance(content, (str, bytes)), "Only strings and bytes can be cached"
    path = Path(path_)
    if make_dirs:
        path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.parent / f".tmp.{os.getpid()}.{uuid.uuid4().hex}"
    try:
        with staged.open("x" if isinstance(content, str) else "xb",
                         encoding="utf-8" if encode_utf_8 else None) as stream:
            stream.write(content)
        return publish_file(staged, path)
    finally:
        try:
            staged.unlink(missing_ok=True)
        except OSError as exc:
            # Cleanup cannot invalidate a completed publication or mask its error.
            print(f"[compile-cache] temporary cleanup {staged}: {exc}",
                  file=sys.stderr, flush=True)


class PublishedFileCacheManager(FileCacheManager):
    """Keep Triton's layout and publish every artifact/group without overwrite."""

    def put(self, data, filename, binary=True):
        if not self.cache_dir:
            raise RuntimeError("Could not create or locate cache dir")
        assert self.lock_path is not None
        # FileCacheManager chooses serialization by data type, not binary.
        binary = isinstance(data, bytes)
        if not binary:
            data = str(data)
        target = self._make_path(filename)
        write_once(target, data)
        return target


def _in_compile_pool(path) -> bool:
    root = os.environ.get("DEFGCM_COMPILE_CACHE_ROOT")
    path = Path(path)
    # Publication changes the directory entry, not a symlink's destination.
    return bool(root) and (path.parent.resolve() / path.name).is_relative_to(Path(root).resolve())


def _build_once(builder, target) -> None:
    """Stage JIT C++ outputs too: a killed linker must not leave a visible .so."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    directory = target.parent / f".tmp.build.{os.getpid()}.{uuid.uuid4().hex}"
    directory.mkdir()
    try:
        staged = directory / target.name
        local = copy.copy(builder)
        if local._use_relative_path or not local._output.endswith(local._target_file):
            raise RuntimeError("unsupported CppBuilder output layout for shared compile pool")
        local._output = local._output[:-len(local._target_file)] + str(staged)
        local._target_file = str(staged)
        local._output_dir = str(directory)
        local._name = target.stem
        local.build()
        publish_file(staged, target)
    finally:
        try:
            shutil.rmtree(directory)
        except OSError as exc:
            print(f"[compile-cache] temporary cleanup {directory}: {exc}",
                  file=sys.stderr, flush=True)


def _merge_timings(target, source) -> None:
    # CacheBase is a growing op/input/precision/choice tree, not an immutable
    # artifact. Merge complete immutable snapshots instead of freezing its first
    # version or repeatedly overwriting its system-wide index.
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge_timings(target[key], value)
        else:
            target[key] = value


def _snapshot_path(path) -> Path:
    return Path(str(path) + ".entries") / f"{time.time_ns():020d}.{uuid.uuid4().hex}.json"


def _worker_compile_triton(load_kernel, extra_env, extra_config, *, cache_key_tag):
    # Inductor's default sidecar is a fresh interpreter, not a fork of LOCOMgcm.
    # This top-level function is pickleable without importing defgcm there.
    os.environ.update(extra_env)
    from torch.compiler import config as compiler_config
    from torch._inductor.runtime.compile_tasks import _worker_compile_triton as compile_kernel

    # A prewarmed sidecar may predate compatibility installation. Its autotune
    # keys must use the submitting process's tag, not its startup environment.
    with compiler_config.patch(cache_key_tag=cache_key_tag):
        configure_compile_cache()
        return compile_kernel(load_kernel, extra_env, extra_config)


def _configure_compile_worker(cache_key_tag: str) -> None:
    """Bind the compiler identity into each pickleable Triton worker task."""
    from torch._inductor import async_compile

    # Keep the subprocess entry import independent of defgcm.__init__.
    directory = str(Path(__file__).resolve().parent)
    if directory not in sys.path:
        sys.path.insert(0, directory)
    pythonpath = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    if directory not in pythonpath:
        os.environ["PYTHONPATH"] = os.pathsep.join([directory, *filter(None, pythonpath)])
    import compile_cache

    async_compile._worker_compile_triton = functools.partial(
        compile_cache._worker_compile_triton, cache_key_tag=cache_key_tag
    )


def configure_compile_cache() -> None:
    """Install the torch 2.9 pool policy before compiling in this process.

    DEFGCM_COMPILE_CACHE_ROOT is explicit: writes outside that directory
    retain torch's implementation. Call at eager execution preparation, before
    entering compiled code. Compiler subprocesses call the same entry point.
    """
    if not os.environ.get("DEFGCM_COMPILE_CACHE_ROOT"):
        return
    import torch

    if not re.match(r"^2\.9\.\d+(?:[+.-]|$)", str(torch.__version__)):
        raise RuntimeError(
            "shared compile-cache publication requires torch 2.9.x; "
            f"found {torch.__version__}"
        )
    from torch._inductor import codecache

    if getattr(codecache.write_atomic, "_defgcm_publish_once", False):
        return
    original = codecache.write_atomic

    @functools.wraps(original)
    def write_atomic(path_, content, make_dirs=False, encode_utf_8=False):
        if _in_compile_pool(path_):
            write_once(path_, content, make_dirs, encode_utf_8)
        else:
            return original(path_, content, make_dirs, encode_utf_8)

    write_atomic._defgcm_publish_once = True
    codecache.write_atomic = write_atomic
    # These torch 2.9 modules bind write_atomic with a from-import. Future
    # imports get the patched function; already imported aliases need rebinding.
    for name in ("torch._functorch._aot_autograd.autograd_cache",
                 "torch._inductor.codegen.triton", "torch._inductor.codegen.multi_kernel"):
        module = sys.modules.get(name)
        if module is not None and getattr(module, "write_atomic", None) is original:
            module.write_atomic = write_atomic

    original_compile_cpp = codecache._worker_compile_cpp

    @functools.wraps(original_compile_cpp)
    def compile_cpp(lock_path, cpp_builders):
        if not any(_in_compile_pool(builder.get_target_file_path()) for builder in cpp_builders):
            return original_compile_cpp(lock_path, cpp_builders)
        from torch.utils._filelock import FileLock

        with FileLock(lock_path, timeout=codecache.LOCK_TIMEOUT):
            for builder in cpp_builders:
                target = builder.get_target_file_path()
                if not os.path.exists(target):
                    if _in_compile_pool(target):
                        _build_once(builder, target)
                    else:
                        builder.build()

    codecache._worker_compile_cpp = compile_cpp

    original_get = codecache.CacheBase.get_local_cache
    original_update = codecache.CacheBase.update_local_cache

    def get_local_cache(self):
        path = self.get_local_cache_path()
        result = original_get(self)
        if _in_compile_pool(path):
            # The old single file remains a read-only seed. No shared name is
            # changed, including when several processes add different choices.
            entries = Path(str(path) + ".entries")
            for entry in sorted(entries.glob("*.json")):
                with entry.open() as stream:
                    _merge_timings(result, json.load(stream)["cache"])
        return result

    def update_local_cache(self, local_cache):
        path = self.get_local_cache_path()
        if not _in_compile_pool(path):
            return original_update(self, local_cache)
        entry = _snapshot_path(path)
        write_once(entry, json.dumps({"system": self.system, "cache": local_cache}, indent=4),
                   make_dirs=True)

    codecache.CacheBase.get_local_cache = get_local_cache
    codecache.CacheBase.update_local_cache = update_local_cache

    # A .best_config can be refined by coordinate descent or invalidated by a
    # new configs_hash. Retaining only its first value would lose those updates.
    from torch._inductor.runtime.autotune_cache import _LocalAutotuneCacheBackend

    original_tune_get = _LocalAutotuneCacheBackend._get
    original_tune_put = _LocalAutotuneCacheBackend._put

    def tune_get(self, key):
        if _in_compile_pool(key):
            entries = sorted(Path(str(key) + ".entries").glob("*.json"))
            if entries:
                return entries[-1].read_bytes()
        return original_tune_get(self, key)

    def tune_put(self, key, data):
        if not _in_compile_pool(key):
            return original_tune_put(self, key, data)
        write_once(_snapshot_path(key), data, make_dirs=True)

    _LocalAutotuneCacheBackend._get = tune_get
    _LocalAutotuneCacheBackend._put = tune_put

    from torch.compiler import config as compiler_config

    _configure_compile_worker(compiler_config.cache_key_tag)
    import compile_cache
    os.environ["TRITON_CACHE_MANAGER"] = "compile_cache:PublishedFileCacheManager"
    # Triton may already have cached the environment-derived manager class.
    from triton import knobs

    knobs.cache.manager_class = compile_cache.PublishedFileCacheManager
