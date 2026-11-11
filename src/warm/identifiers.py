"""Stable identifiers matching captions/method.py (WARM-ID-V1)."""
from __future__ import annotations

import base64
import hashlib
import json


def token(domain: str, payload) -> str:
    canonical = json.dumps([domain, payload], ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(("WARM-ID-V1|" + canonical).encode("utf-8")).digest()
    return base64.b32encode(digest).decode("ascii")[:10]


def source_id(original_number: int) -> str:
    return "V_E_" + token("experiment:source", int(original_number))


def predictor_video_id(kind: str, original_number: int) -> str:
    if kind not in {"train", "test"}:
        raise ValueError(f"Invalid prediction partition {kind!r}")
    return "V_T_" + token("prediction:" + kind, int(original_number))


def replica_id(video_id: str, height, width, fps, gop, codec, params) -> str:
    # The order and scalar JSON types must match the CSV -> JSON conversion script.
    values = [video_id, height, width, fps, gop, codec, params]
    return "C_E_" + video_id.removeprefix("V_E_") + "_" + token("replica:configuration", values)
