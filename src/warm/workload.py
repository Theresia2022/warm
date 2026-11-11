from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .config import WorkloadConfig

EPS = 1e-12


def _height(width: float, source_w: float, source_h: float) -> int:
    if source_w <= 0:
        return int(round(source_h))
    return max(4, int((width / source_w) * source_h / 4.0) * 4)


def _le_source(source: float, values: tuple[float, ...]) -> list[float]:
    return sorted({float(source)} | {float(x) for x in values if float(x) <= float(source) + 1e-9})


def _reindex_requests(df: pd.DataFrame) -> pd.DataFrame:
    out = df.reset_index(drop=True).copy()
    out["request_idx"] = np.arange(len(out), dtype=np.int64)
    counters: dict[object, int] = {}
    rids = []
    for video_id in out["VID"]:
        j = counters.get(video_id, 0)
        rids.append(f"{video_id}:{j}")
        counters[video_id] = j + 1
    out["rid"] = rids
    return out


def build_request_catalog(
    originals: pd.DataFrame,
    cfg: WorkloadConfig | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    cfg = cfg or WorkloadConfig()
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for src in originals.itertuples(index=False):
        widths = _le_source(float(src.weight_source), tuple(float(x) for x in cfg.request_widths))
        fps_values = _le_source(float(src.frame_rate_source), cfg.request_frame_rates)
        local: list[dict] = []
        for w in widths:
            h = _height(w, float(src.weight_source), float(src.height_source))
            for fps in fps_values:
                local.append({
                    "VID": src.VID,
                    "req_w": int(round(w)),
                    "req_h": int(h),
                    "req_f": float(fps),
                    "req_proto": str(src.encoding_protocol_source).lower(),
                    "req_g": float(src.gop_length_source),
                    "source_decode_cost": float(src.source_decode_s),
                })
        local_df = pd.DataFrame(local).drop_duplicates(["req_w", "req_h", "req_f"])
        if cfg.max_requests_per_video and len(local_df) > cfg.max_requests_per_video:
            # Deterministically retain low/mid/high logical specs. The request
            # catalog is fixed across statistical workload seeds; only weights
            # and future arrivals change later.
            order = local_df.sort_values(["req_w", "req_f"]).index.to_numpy()
            positions = np.linspace(0, len(order) - 1, cfg.max_requests_per_video).round().astype(int)
            keep = np.unique(order[positions])
            if len(keep) < cfg.max_requests_per_video:
                remaining = np.setdiff1d(order, keep)
                rng.shuffle(remaining)
                keep = np.concatenate([keep, remaining[: cfg.max_requests_per_video - len(keep)]])
            local_df = local_df.loc[keep].copy()
        rows.extend(local_df.to_dict("records"))
    return _reindex_requests(pd.DataFrame(rows))


def build_controlled_overlap_catalog(
    originals: pd.DataFrame,
    cfg: WorkloadConfig,
    mode: str,
    requests_per_video: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Construct low/natural/high overlap request catalogs on all videos.

    ``high`` selects a monotone chain in (resolution, fps), causing richer
    replicas to cover nested request sets. ``low`` selects an anti-diagonal of
    the grid, making requests more mutually incomparable. ``natural`` uses the
    standard catalog sampler. All modes keep requests no richer than source.
    """
    mode = mode.lower()
    if mode == "natural":
        return build_request_catalog(
            originals,
            replace(cfg, max_requests_per_video=requests_per_video),
            seed=seed,
        )
    if mode not in {"low", "high"}:
        raise ValueError(f"unknown overlap mode: {mode}")

    full = build_request_catalog(originals, replace(cfg, max_requests_per_video=0), seed=seed)
    out_groups = []
    for _vid, g in full.groupby("VID", sort=False):
        widths = sorted(g["req_w"].unique())
        fpss = sorted(g["req_f"].unique())
        if not widths or not fpss:
            continue
        k = max(1, min(int(requests_per_video), max(len(widths), len(fpss))))
        qs = np.linspace(0.0, 1.0, k)
        chosen = []
        used = set()
        for q in qs:
            wi = int(round(q * (len(widths) - 1)))
            if mode == "high":
                fi = int(round(q * (len(fpss) - 1)))
            else:
                fi = int(round((1.0 - q) * (len(fpss) - 1)))
            w, f = widths[wi], fpss[fi]
            candidates = g[(g["req_w"] == w) & np.isclose(g["req_f"], f, atol=1e-9)]
            if len(candidates):
                idx = int(candidates.index[0])
                if idx not in used:
                    chosen.append(idx)
                    used.add(idx)
        # Fill if quantile rounding produced duplicates.
        if len(chosen) < min(requests_per_video, len(g)):
            remaining = [int(i) for i in g.index if int(i) not in used]
            # For high overlap prefer low->high complexity; for low overlap prefer
            # points far from the diagonal (incomparable trade-offs).
            if mode == "high":
                remaining.sort(key=lambda i: (float(g.loc[i, "req_w"]), float(g.loc[i, "req_f"])))
            else:
                w_rank = {w: j for j, w in enumerate(widths)}
                f_rank = {f: j for j, f in enumerate(fpss)}
                remaining.sort(
                    key=lambda i: -abs(
                        (w_rank[g.loc[i, "req_w"]] / max(len(widths) - 1, 1))
                        - (f_rank[g.loc[i, "req_f"]] / max(len(fpss) - 1, 1))
                    )
                )
            need = min(requests_per_video, len(g)) - len(chosen)
            chosen.extend(remaining[:need])
        out_groups.append(g.loc[chosen])
    if not out_groups:
        return full.iloc[0:0].copy()
    return _reindex_requests(pd.concat(out_groups, ignore_index=True))


def _zipf_probs(n: int, alpha: float) -> np.ndarray:
    if n <= 0:
        return np.empty(0, dtype=float)
    ranks = np.arange(1, n + 1, dtype=float)
    p = ranks ** (-float(alpha))
    return p / p.sum()


def attach_synthetic_workload_weights(
    requests: pd.DataFrame,
    cfg: WorkloadConfig | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    cfg = cfg or WorkloadConfig()
    rng = np.random.default_rng(seed)
    out = requests.copy()
    vids = np.asarray(sorted(out["VID"].unique()))
    rng.shuffle(vids)
    if cfg.video_distribution == "uniform":
        vp = np.full(len(vids), 1.0 / max(len(vids), 1))
    elif cfg.video_distribution == "zipf":
        vp = _zipf_probs(len(vids), cfg.video_alpha)
    else:
        raise ValueError(f"unknown video distribution: {cfg.video_distribution}")
    video_prob = dict(zip(vids, vp))
    out["video_access_prob"] = out["VID"].map(video_prob).astype(float)
    out["request_prob_given_video"] = 0.0
    for _vid, group in out.groupby("VID", sort=False):
        idx = group.index.to_numpy()
        shuffled = idx.copy()
        rng.shuffle(shuffled)
        if cfg.request_distribution == "uniform":
            rp = np.full(len(idx), 1.0 / max(len(idx), 1))
        elif cfg.request_distribution == "zipf":
            rp = _zipf_probs(len(idx), cfg.request_alpha)
        else:
            raise ValueError(f"unknown request distribution: {cfg.request_distribution}")
        out.loc[shuffled, "request_prob_given_video"] = rp
    out["weight"] = out["video_access_prob"] * out["request_prob_given_video"]
    out["weight"] = out["weight"] / max(float(out["weight"].sum()), EPS)
    out["access_freq"] = out["weight"] * float(cfg.total_historical_queries)
    return out


def workload_from_log(
    requests: pd.DataFrame,
    log_path: str | Path,
    total_historical_queries: int | None = None,
) -> pd.DataFrame:
    log_path = Path(log_path)
    if log_path.suffix.lower() != ".json":
        raise ValueError("historical log must be a JSON array of request objects")
    log = pd.read_json(log_path, orient="records")
    if "VID" in log and "VID" not in log:
        log = log.rename(columns={"VID": "VID"})
    required = ["VID", "req_w", "req_h", "req_f"]
    missing = [c for c in required if c not in log.columns]
    if missing:
        raise ValueError(f"query log missing columns: {missing}")
    counts = log.groupby(required, as_index=False).size().rename(columns={"size": "count"})
    out = requests.merge(counts, on=required, how="left")
    out["count"] = out["count"].fillna(0.0)
    total = float(out["count"].sum())
    if total <= 0:
        raise ValueError("historical query log has no requests matching the catalog")
    out["weight"] = out["count"] / total
    scale = total_historical_queries if total_historical_queries is not None else total
    out["access_freq"] = out["weight"] * float(scale)
    video_counts = out.groupby("VID")["count"].transform("sum")
    out["video_access_prob"] = video_counts / total
    out["request_prob_given_video"] = np.where(video_counts > 0, out["count"] / video_counts, 0.0)
    return out.drop(columns=["count"])


def build_workload(
    originals: pd.DataFrame,
    cfg: WorkloadConfig | None = None,
    seed: int = 42,
    historical_log: str | Path | None = None,
) -> pd.DataFrame:
    cfg = cfg or WorkloadConfig()
    requests = build_request_catalog(originals, cfg=cfg, seed=seed)
    if historical_log is not None:
        return workload_from_log(requests, historical_log, cfg.total_historical_queries)
    return attach_synthetic_workload_weights(requests, cfg=cfg, seed=seed)


def make_future_trace(
    requests: pd.DataFrame,
    query_count: int,
    arrival_rate: float,
    seed: int = 42,
    process: str = "poisson",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    probs = requests["weight"].to_numpy(float)
    probs = probs / probs.sum()
    sampled_pos = rng.choice(np.arange(len(requests)), size=int(query_count), replace=True, p=probs)
    if process == "poisson":
        inter = rng.exponential(1.0 / max(float(arrival_rate), EPS), size=query_count)
    elif process == "fixed":
        inter = np.full(query_count, 1.0 / max(float(arrival_rate), EPS))
    elif process == "bursty":
        state = 0
        inter = np.empty(query_count, dtype=float)
        for i in range(query_count):
            if rng.random() < 0.04:
                state = 1 - state
            multiplier = 4.0 if state else 0.35
            inter[i] = rng.exponential(1.0 / max(arrival_rate * multiplier, EPS))
    else:
        raise ValueError(f"unknown arrival process: {process}")
    arrival = np.cumsum(inter)
    trace = requests.iloc[sampled_pos].copy().reset_index(drop=True)
    trace["trace_id"] = np.arange(query_count, dtype=np.int64)
    trace["arrival_time"] = arrival
    return trace
