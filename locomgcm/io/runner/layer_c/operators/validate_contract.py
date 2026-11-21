"""Pass-through operator that enforces the physical-plausibility data contract.

Placed as the LAST operator of the state / boundary prep pipelines (after
``vertical_interp``, before the zarr consumer) so that physically-implausible
fields fail the run loud at write time — schema-on-write — instead of silently
flowing into a model blow-up.  See :mod:`locomgcm.io.runner.layer_c.data_contract`.

YAML usage::

    - op: validate_contract        # default: fail-loud on HARD violations
                                   # (soft ones are logged, not raised)
    - op: validate_contract        # raise_on: warn = STRICT — blocks on ANY
      raise_on: warn               # violation (incl. soft), not "warn-only"

Note: ``raise_on: warn`` is stricter than the default, not more lenient — it
raises on any violation. See ``ValidateContractOperator`` below for the exact
severity semantics.

The operator is pure (returns the bundle unchanged); its only effect is raising
:class:`ContractViolationError` on a violation at or above ``raise_on``.
"""

from __future__ import annotations

import logging
from typing import Any

from locomgcm.io.runner.layer_a import DataBundle, LocationKind

from .._registry import register_operator
from .._types import OperatorCapabilities
from ..data_contract import OCEAN_IC_CONTRACT, validate_contract

_LOGGER = logging.getLogger("locomgcm.data_contract")


@register_operator("validate_contract")
class ValidateContractOperator:
    """Enforce a declarative physical-plausibility contract on the bundle.

    Parameters
    ----------
    contract : dict, optional
        Declarative per-variable check spec (see ``data_contract.OCEAN_IC_CONTRACT``).
        Defaults to the built-in ocean IC/boundary contract.
    raise_on : {"fail", "warn"}, default "fail"
        Minimum severity that aborts the run.  ``"fail"`` blocks on hard
        violations (logging soft ones); ``"warn"`` blocks on anything (strict);
        set per-case in YAML to relax a noisy gate without editing code.
    """

    capabilities = OperatorCapabilities(
        accepts=(LocationKind.GRIDDED_3D_OCEAN, LocationKind.GRIDDED_2D_SURFACE,
                 LocationKind.BOUNDARY_TIMESERIES),
        produces=(LocationKind.GRIDDED_3D_OCEAN, LocationKind.GRIDDED_2D_SURFACE,
                  LocationKind.BOUNDARY_TIMESERIES),
        convention_in=None,
        convention_out=None,
        preserves_variables=True,
    )

    def __init__(
        self,
        *,
        contract: dict[str, list[dict[str, Any]]] | None = None,
        raise_on: str = "fail",
        **_: Any,
    ) -> None:
        self._contract = contract if contract is not None else OCEAN_IC_CONTRACT
        self._raise_on = str(raise_on)

    def __call__(self, bundle: DataBundle) -> DataBundle:
        validate_contract(
            bundle.data,
            self._contract,
            raise_on=self._raise_on,
            logger=_LOGGER,
        )
        return bundle
