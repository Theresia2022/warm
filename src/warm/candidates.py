from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CandidateConfig

EPS = 1e-9


def _in_numeric_pool(series: pd.Series, pool: tuple[float, ...], tol: float = 1e-6) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(float)
    if not pool:
        return pd.Series(True, index=series.index)
    p = np.asarray(pool, dtype=float)
    ok = np.any(np.isclose(values[:, None], p[None, :], rtol=0.0, atol=tol), axis=1)
    return pd.Series(ok, index=series.index)


def build_candidate_catalog(
    originals: pd.DataFrame,
    copies: pd.DataFrame,
    cfg: CandidateConfig | None = None,
) -> pd.DataFrame:
    """Build the materializable candidate catalog from measured Copies_Info.

    WARM creates *lower/equal logical-specification replicas* from the source.
    The source itself remains the always-available fallback.  This intentionally
    fixes the old CAVM implementation that filtered widths/frame-rates in the
    opposite direction.
    """
    cfg = cfg or CandidateConfig()
    src = originals[["VID", "weight_source", "height_source", "frame_rate_source"]].drop_duplicates("VID")
    cand = copies.merge(src, on="VID", how="inner", suffixes=("", "_src"))

    if cfg.logical_only_not_richer_than_source:
        mask = (
            (cand["weight_replica"] <= cand["weight_source_src"] + EPS)
            & (cand["height_replica"] <= cand["height_source_src"] + EPS)
            & (cand["frame_rate_replica"] <= cand["frame_rate_source_src"] + EPS)
        )
        cand = cand[mask].copy()

    # Restrict to declared system-supported knobs while allowing exact source
    # logical values even if they are not explicitly in the pool.
    width_ok = _in_numeric_pool(cand["weight_replica"], tuple(float(x) for x in cfg.widths)) | np.isclose(
        cand["weight_replica"].to_numpy(float), cand["weight_source_src"].to_numpy(float), atol=1e-6
    )
    fps_ok = _in_numeric_pool(cand["frame_rate_replica"], cfg.frame_rates) | np.isclose(
        cand["frame_rate_replica"].to_numpy(float), cand["frame_rate_source_src"].to_numpy(float), atol=1e-6
    )
    quality_ok = _in_numeric_pool(cand["gop_length_replica"], cfg.qualities)
    codec_ok = cand["encoding_protocol_replica"].isin([str(x).lower() for x in cfg.codecs])
    encoding_params_ok = cand["encoding_params"].isin(cfg.params)
    cand = cand[width_ok & fps_ok & quality_ok & codec_ok & encoding_params_ok].copy()

    cand = cand.drop(columns=[c for c in ["weight_source_src", "height_source_src", "frame_rate_source_src"] if c in cand.columns])
    cand = cand.drop_duplicates(["VID", "CID"]).reset_index(drop=True)
    cand["cand_idx"] = np.arange(len(cand), dtype=np.int64)
    return cand


def specification_distance(cand_row, req_row) -> float:
    """Tie-break distance among already-compatible representations."""
    cw, ch, cf = float(cand_row["weight_replica"]), float(cand_row["height_replica"]), float(cand_row["frame_rate_replica"])
    rw, rh, rf = float(req_row["req_w"]), float(req_row["req_h"]), float(req_row["req_f"])
    area_ratio = max((cw * ch) / max(rw * rh, 1.0), 1.0)
    fps_ratio = max(cf / max(rf, 1e-9), 1.0)
    return float(np.log(area_ratio) + np.log(fps_ratio))
