"""The single place a fused PCG package reads its gate and writes its ledger.

The shared acceleration dispatcher provides one decision point per key; this is
it for every PCG client. The gate is read ONCE per solve (a snapshot lookup
inside a 50-iteration loop is Python on the critical path) and the ledger is
written once per solve, one row PER KERNEL carrying that kernel's launch count.

Per-kernel rows include launches before the first completed iteration, such
as the pre-loop operator and preconditioner. This preserves launch counts
even when a solve breaks down at pAp near zero. The update kernel count
records the number of iterations served.

Nothing here knows a grid family or a rank count.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable

from locomgcm.accel import current_rank, dispatch_scope, gate, record_dispatch


class FusedIterationLedger:
    """Launch counter + ledger entry for one solve served by fused kernels.

    Subclasses add the kernel methods and call :meth:`count` in each.  Used as
    a context manager around the host loop::

        with bundle:
            pcg_iterate(kernels=bundle, ...)

    ``__exit__`` writes one ``hit`` row per kernel that launched, or one
    ``error`` row if the loop raised.  A solve admitted that launched nothing
    records nothing, which is the truth.
    """

    def __init__(self, *, key: str, site: str, variant: str, backend: str = "cuda") -> None:
        self._key = key
        self._site = site
        self._backend = backend
        self._variant = variant
        self._scope: Any = None
        self._launches: Counter[str] = Counter()
        self.errored = False
        self.iterations = 0

    # -- accounting ---------------------------------------------------------

    @property
    def variant(self) -> str:
        return self._variant

    @property
    def launches(self) -> dict[str, int]:
        """Launch count per kernel so far, the ledger's own source."""
        return dict(self._launches)

    def count(self, kernel: str) -> None:
        self._launches[kernel] += 1

    def _record(self, outcome: str, *, variant: str, reason: str | None = None,
                count: int = 1) -> None:
        record_dispatch(
            self._key,
            outcome=outcome,
            variant=variant,
            site=self._site,
            backend=self._backend,
            rank=current_rank(),
            reason=reason,
            count=count,
        )

    def __enter__(self) -> "FusedIterationLedger":
        # The scope spans the whole solve, not one launch.  Wider than the
        # sibling dispatchers hold it, and intentional: the recursion guard's
        # job here is "the eager reference the caller falls back to must not
        # re-enter this dispatcher", and that fallback happens after the loop.
        self._scope = dispatch_scope(self._key)
        self._scope.__enter__()
        return self

    @staticmethod
    def _reason_for(exc: BaseException) -> str:
        """The ledger reason for a failed solve.

        A per-launch refusal raises ``RuntimeError: native... is inapplicable:
        <token>``, and ``<token>`` is the whole diagnostic value -- recording
        the bare exception CLASS turned every per-launch refusal into an
        indistinguishable ``RuntimeError`` row, i.e. a silent
        no-acceleration whose cause the ledger could not name.
        Anything else keeps its class name.
        """
        text = str(exc)
        marker = "is inapplicable: "
        if isinstance(exc, RuntimeError) and marker in text:
            return text.rsplit(marker, 1)[1].strip()[:64] or type(exc).__name__
        return type(exc).__name__

    def record_error(self, exc: BaseException) -> None:
        """Record one ``error`` row for this solve, at most once.

        Called by ``__exit__`` when the loop raises, and by the call site when
        something outside the ``with`` block does.  Idempotent, so the two
        cannot double-count one failed solve.
        """
        if self.errored:
            return
        self.errored = True
        self._record("error", variant=self._variant, reason=self._reason_for(exc))

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        try:
            if exc_type is not None:
                self.record_error(
                    exc_value if exc_value is not None else exc_type()
                )
            else:
                for kernel, count in sorted(self._launches.items()):
                    self._record(
                        "hit", variant=f"{self._variant}_{kernel}", count=count
                    )
        finally:
            scope, self._scope = self._scope, None
            if scope is not None:
                scope.__exit__(exc_type, exc_value, traceback)
        return False


def open_iteration(
    *,
    key: str,
    site: str,
    variant: str,
    decline: Callable[[], str | None],
    build: Callable[[], Any],
    backend: str = "cuda",
) -> Any | None:
    """Read the gate once; return the bundle, or ``None`` with a recorded reason.

    ``None`` means the caller must run its eager reference.  Nothing is recorded
    when the gate is off -- recording a disabled key is an error by construction
    (``accel/runtime.py`` raises).
    """
    if not gate(key).effective_value:
        return None
    with dispatch_scope(key):
        reason = decline()
        if reason is not None:
            record_dispatch(
                key,
                outcome="inapplicable",
                variant=variant,
                site=site,
                backend=backend,
                rank=current_rank(),
                reason=reason,
            )
            return None
    return build()


__all__ = ["FusedIterationLedger", "open_iteration"]
