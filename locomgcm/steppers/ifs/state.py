"""Restart codecs for the completed-step counter and elevation histories.

The live counter is a 0-d int64 CPU tensor; the shared restart boundary writes
an exact int. Prescribed elevation remains a scalar or an upward-positive
cell-centre tensor in m, retaining its dtype.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

def encode_step_index(value):
    return {} if value is None else {"_step_index": value}


def decode_step_index(payload):
    from locomgcm.checkpoint.state import live_step_index

    return live_step_index(payload["_step_index"]) if "_step_index" in payload else None


def encode_boundary_eta(value):
    return {} if value is None else {"_obc_eta": value}


def decode_boundary_eta(payload):
    return payload.get("_obc_eta")


@dataclass(frozen=True)
class PreviousEtaHistory:
    eta: Any
    present: bool


def encode_previous_eta(history: PreviousEtaHistory):
    return {"_eta_prev_step": history.eta} if history.present else {}


def decode_previous_eta(payload):
    return PreviousEtaHistory(payload.get("_eta_prev_step"), "_eta_prev_step" in payload)
