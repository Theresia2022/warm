from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from .config import OptimizerConfig, WorkloadConfig
from .data import KB_TO_MB, US_TO_S

EPS = 1e-12


@dataclass
class CompatibilityIndex:
    coverage: Dict[int, np.ndarray]
    coverage_bits: Dict[int, int]
    request_count: int


@dataclass
class OptimizationInstance:
    candidates: pd.DataFrame
    requests: pd.DataFrame
    compatibility: CompatibilityIndex
    budget_mb: float
    storage_col: str = "robust_storage_mb"
    decode_col: str = "robust_decode_s"
    enforce_true_budget_at_materialization: bool = True


def add_physical_costs(candidates: pd.DataFrame, opt_cfg: OptimizerConfig | None = None) -> pd.DataFrame:
    opt_cfg = opt_cfg or OptimizerConfig()
    out = candidates.copy()
    if "serving_cost_source" not in out.columns:
        raise ValueError("candidate table must contain serving_cost_source after data normalization")
    source_decode_s = out["serving_cost_source"].astype(float) * US_TO_S
    dsv_true = np.maximum(out["serving_speedup_value"].astype(float), EPS)

    out["true_storage_mb"] = out["storage_cost_source"].astype(float) * out["storage_overhead_radio"].astype(float) * KB_TO_MB
    # serving_speedup_value is a decode-speed ratio: serving_speedup_value = T_src / T_rep.
    # Therefore T_rep = T_src / serving_speedup_value.
    out["true_decode_s"] = source_decode_s / dsv_true

    if "storage_overhead_radio_pred" not in out.columns or "serving_speedup_value_pred" not in out.columns:
        raise ValueError("candidate table must contain storage_overhead_radio_pred and serving_speedup_value_pred")

    dsv_pred = np.maximum(out["serving_speedup_value_pred"].astype(float), EPS)
    out["pred_storage_mb"] = out["storage_cost_source"].astype(float) * out["storage_overhead_radio_pred"].astype(float) * KB_TO_MB
    out["pred_decode_s"] = source_decode_s / dsv_pred

    storage_std_ratio = out["storage_overhead_radio_std"].astype(float) if "storage_overhead_radio_std" in out.columns else 0.0
    decode_std_ratio = out["serving_speedup_value_std"].astype(float) if "serving_speedup_value_std" in out.columns else 0.0

    out["pred_storage_std_mb"] = out["storage_cost_source"].astype(float) * storage_std_ratio * KB_TO_MB
    # Delta-method diagnostic only: std(T_src / serving_speedup_value) ~= T_src * std(serving_speedup_value) / serving_speedup_value^2.
    out["pred_decode_std_s"] = source_decode_s * decode_std_ratio / np.maximum(dsv_pred ** 2, EPS)

    out["robust_storage_mb"] = np.maximum(
        out["pred_storage_mb"] + opt_cfg.beta_storage * out["pred_storage_std_mb"], EPS
    )

    # For serving_speedup_value = T_src/T_rep, a *smaller* serving_speedup_value means a *larger* replica decode
    # time. Hence a conservative decode cost must use a lower confidence value
    # of serving_speedup_value rather than serving_speedup_value + beta*std.
    robust_dsv_lower = np.maximum(
        dsv_pred - opt_cfg.beta_decode * decode_std_ratio,
        EPS,
    )
    out["robust_decode_s"] = np.maximum(source_decode_s / robust_dsv_lower, EPS)
    return out


def build_compatibility(
    candidates: pd.DataFrame,
    requests: pd.DataFrame,
    workload_cfg: WorkloadConfig | None = None,
) -> CompatibilityIndex:
    cfg = workload_cfg or WorkloadConfig()
    if "cand_idx" not in candidates.columns:
        raise ValueError("candidates must contain cand_idx")
    req = requests.sort_values("request_idx").reset_index(drop=True)
    if not np.array_equal(req["request_idx"].to_numpy(), np.arange(len(req))):
        req = req.copy()
        req["request_idx"] = np.arange(len(req), dtype=np.int64)
    coverage: Dict[int, np.ndarray] = {}
    bits: Dict[int, int] = {}
    req_by_vid = {video_id: g for video_id, g in req.groupby("VID", sort=False)}
    for video_id, cg in candidates.groupby("VID", sort=False):
        rg = req_by_vid.get(video_id)
        if rg is None or len(rg) == 0:
            for cidx in cg["cand_idx"]:
                coverage[int(cidx)] = np.empty(0, dtype=np.int64)
                bits[int(cidx)] = 0
            continue
        rw = rg["req_w"].to_numpy(float)[None, :]
        rh = rg["req_h"].to_numpy(float)[None, :]
        rf = rg["req_f"].to_numpy(float)[None, :]
        cw = cg["weight_replica"].to_numpy(float)[:, None]
        ch = cg["height_replica"].to_numpy(float)[:, None]
        cf = cg["frame_rate_replica"].to_numpy(float)[:, None]
        mat = (cw + 1e-9 >= rw) & (ch + 1e-9 >= rh) & (cf + 1e-9 >= rf)
        if not cfg.logical_coverage_only:
            if cfg.strict_codec and "req_proto" in rg.columns:
                cp = cg["encoding_protocol_replica"].astype(str).to_numpy()[:, None]
                rp = rg["req_proto"].astype(str).to_numpy()[None, :]
                mat &= cp == rp
            if cfg.quality_constraint and "req_g" in rg.columns:
                cq = cg["gop_length_replica"].to_numpy(float)[:, None]
                rq = rg["req_g"].to_numpy(float)[None, :]
                mat &= cq + 1e-9 >= rq
        # Coverage arrays retain GLOBAL request_idx values because routing and
        # service evaluation index the full request vector.  coverage_bits,
        # however, are used only for same-video dominance/overlap checks.
        # Using global request positions made every bit mask ~46k bits wide for
        # the 1500-video experiment.  A per-video local mask is semantically
        # identical for same-video subset tests and is orders of magnitude
        # smaller (typically <= max_requests_per_video bits).
        rpos = rg["request_idx"].to_numpy(np.int32)
        for row_i, cidx in enumerate(cg["cand_idx"].to_numpy(np.int64)):
            local_hits = np.flatnonzero(mat[row_i])
            idx = rpos[local_hits]
            coverage[int(cidx)] = idx
            b = 0
            for local_j in local_hits:
                b |= 1 << int(local_j)
            bits[int(cidx)] = b
    return CompatibilityIndex(coverage=coverage, coverage_bits=bits, request_count=len(req))


def attach_singleton_gain(
    candidates: pd.DataFrame,
    requests: pd.DataFrame,
    compatibility: CompatibilityIndex,
    storage_col: str = "robust_storage_mb",
    decode_col: str = "robust_decode_s",
) -> pd.DataFrame:
    out = candidates.copy()
    req = requests.sort_values("request_idx").reset_index(drop=True)
    weights = req["weight"].to_numpy(float)
    src = req["source_decode_cost"].to_numpy(float)
    gains = []
    densities = []
    for row in out.itertuples(index=False):
        cidx = int(row.cand_idx)
        ridx = compatibility.coverage.get(cidx, np.empty(0, dtype=np.int64))
        tau = float(getattr(row, decode_col))
        gain = float(np.dot(weights[ridx], np.maximum(src[ridx] - tau, 0.0))) if len(ridx) else 0.0
        storage = float(getattr(row, storage_col))
        gains.append(gain)
        densities.append(gain / max(storage, EPS))
    out["singleton_gain"] = gains
    out["singleton_density"] = densities
    return out


def prune_dominated_candidates(
    candidates: pd.DataFrame,
    compatibility: CompatibilityIndex,
    storage_col: str = "robust_storage_mb",
    decode_col: str = "robust_decode_s",
) -> tuple[pd.DataFrame, CompatibilityIndex, int]:
    """Safe same-video dominance pruning from main.pdf Eq. 28-29."""
    keep: set[int] = set(int(x) for x in candidates["cand_idx"])
    removed = 0
    for _vid, group in candidates.groupby("VID", sort=False):
        records = list(group[["cand_idx", storage_col, decode_col]].itertuples(index=False, name=None))
        # Sort promising dominators first. This does not change correctness.
        records.sort(key=lambda x: (float(x[1]), float(x[2]), -int(compatibility.coverage_bits.get(int(x[0]), 0).bit_count())))
        for i, (a_idx, a_s, a_t) in enumerate(records):
            a_idx = int(a_idx)
            if a_idx not in keep:
                continue
            a_bits = compatibility.coverage_bits.get(a_idx, 0)
            for b_idx, b_s, b_t in records[i + 1:]:
                b_idx = int(b_idx)
                if b_idx not in keep:
                    continue
                b_bits = compatibility.coverage_bits.get(b_idx, 0)
                covers_superset = (a_bits | b_bits) == a_bits
                if float(a_s) <= float(b_s) + EPS and float(a_t) <= float(b_t) + EPS and covers_superset:
                    keep.remove(b_idx)
                    removed += 1
    pruned = candidates[candidates["cand_idx"].isin(keep)].copy().reset_index(drop=True)
    cov = {int(k): v for k, v in compatibility.coverage.items() if int(k) in keep}
    bits = {int(k): v for k, v in compatibility.coverage_bits.items() if int(k) in keep}
    return pruned, CompatibilityIndex(cov, bits, compatibility.request_count), removed


def build_instance(
    candidates: pd.DataFrame,
    requests: pd.DataFrame,
    budget_mb: float,
    workload_cfg: WorkloadConfig | None = None,
    optimizer_cfg: OptimizerConfig | None = None,
    use_true_costs: bool = False,
) -> OptimizationInstance:
    optimizer_cfg = optimizer_cfg or OptimizerConfig()
    cand = add_physical_costs(candidates, optimizer_cfg)
    comp = build_compatibility(cand, requests, workload_cfg)
    storage_col = "true_storage_mb" if use_true_costs else "robust_storage_mb"
    decode_col = "true_decode_s" if use_true_costs else "robust_decode_s"
    cand = attach_singleton_gain(cand, requests, comp, storage_col=storage_col, decode_col=decode_col)
    if optimizer_cfg.enable_dominance_pruning:
        cand, comp, _ = prune_dominated_candidates(cand, comp, storage_col=storage_col, decode_col=decode_col)
    return OptimizationInstance(
        cand,
        requests.sort_values("request_idx").reset_index(drop=True),
        comp,
        float(budget_mb),
        storage_col,
        decode_col,
        bool(optimizer_cfg.enforce_true_budget_at_materialization),
    )
