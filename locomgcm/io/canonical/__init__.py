"""Canonical model state schema — single source of truth for the IO pipeline.

This module defines the contract between L3 (Canonical State Builder) and
downstream layers (L4 Format Adapters, L5 Model Runners, L6 Diagnostics):
the objects re-exported below are the canonical single-source-of-truth (SSOT)
interface between these layers. See ``state_schema.py`` for the field-level
definitions and validation rules.
"""

from .state_schema import (
    CanonicalModelInput,
    L3ProvenanceStep,
    SignConvention,
    StagedField,
    StagedLocation,
    validate_canonical,
)

__all__ = [
    "CanonicalModelInput",
    "L3ProvenanceStep",
    "SignConvention",
    "StagedField",
    "StagedLocation",
    "validate_canonical",
]
