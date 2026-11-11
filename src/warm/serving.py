from __future__ import annotations

from dataclasses import dataclass
import heapq

import numpy as np
import pandas as pd

from .coupling import OptimizationInstance
from .candidates import specification_distance

EPS = 1e-12


@dataclass
class ServingMetrics:
    query_count: int
    workers: int
    arrival_rate: float
    achieved_throughput_qps: float
    mean_latency_s: float
    p50_latency_s: float
    p95_latency_s: float
    p99_latency_s: float
    mean_queue_s: float
    p95_queue_s: float
    sla_miss_rate: float
    utilization: float
    makespan_s: float


def route_requests_true(
    instance: OptimizationInstance,
    plan: pd.DataFrame,
    routing_policy: str = "min_cost",
) -> pd.DataFrame:
    """Route requests to feasible representations.

    ``min_cost`` implements main.pdf Eq. (7): source is the fallback and the
    feasible path with minimum true service cost wins; specification distance
    is only a tie-break.

    ``closest_feasible`` first minimizes resolution/FPS overshoot among
    compatible materialized replicas, then uses service cost as the tie-break.
    If no compatible replica exists, it falls back to the source.

    Formal Chapter-3/4 optimization experiments should keep ``min_cost`` because
    the submodular objective and approximation analysis are written for that
    rule.
    """
    if routing_policy not in {"min_cost", "closest_feasible"}:
        raise ValueError(f"unknown routing_policy: {routing_policy}")

    req = instance.requests.sort_values("request_idx").reset_index(drop=True).copy()
    cand = instance.candidates.set_index("cand_idx", drop=False)
    selected = set() if plan is None else set(
        int(x) for x in plan.get("cand_idx", pd.Series(dtype=int))
    )

    service = req["source_decode_cost"].to_numpy(float).copy()
    served_by = np.asarray(["source"] * len(req), dtype=object)
    overshoot = np.full(len(req), np.inf, dtype=float)

    if routing_policy == "min_cost":
        for cidx in selected:
            if cidx not in cand.index:
                continue
            row = cand.loc[cidx]
            tau = float(row["true_decode_s"])
            ridx = instance.compatibility.coverage.get(
                cidx, np.empty(0, dtype=np.int64)
            )
            for r in ridx:
                rr = req.iloc[int(r)]
                dist = specification_distance(row, rr)
                if tau < service[r] - 1e-12 or (
                    abs(tau - service[r]) <= 1e-12 and dist < overshoot[r]
                ):
                    service[r] = tau
                    served_by[r] = f"replica:{row['CID']}"
                    overshoot[r] = dist
    else:
        best_dist = np.full(len(req), np.inf, dtype=float)
        best_tau = np.full(len(req), np.inf, dtype=float)
        for cidx in selected:
            if cidx not in cand.index:
                continue
            row = cand.loc[cidx]
            tau = float(row["true_decode_s"])
            ridx = instance.compatibility.coverage.get(
                cidx, np.empty(0, dtype=np.int64)
            )
            for r in ridx:
                rr = req.iloc[int(r)]
                dist = specification_distance(row, rr)
                if dist < best_dist[r] - 1e-12 or (
                    abs(dist - best_dist[r]) <= 1e-12
                    and tau < best_tau[r] - 1e-12
                ):
                    best_dist[r] = dist
                    best_tau[r] = tau
                    service[r] = tau
                    served_by[r] = f"replica:{row['CID']}"
                    overshoot[r] = dist

    req["served_by"] = served_by
    req["service_cost_s"] = service
    req["spec_overshoot"] = np.where(np.isfinite(overshoot), overshoot, 0.0)
    return req


def weighted_service_metrics(
    instance: OptimizationInstance,
    plan: pd.DataFrame,
    routing_policy: str = "min_cost",
) -> dict:
    route = route_requests_true(instance, plan, routing_policy=routing_policy)
    w = route["weight"].to_numpy(float)
    src = route["source_decode_cost"].to_numpy(float)
    svc = route["service_cost_s"].to_numpy(float)
    source_cost = float(np.dot(w, src))
    plan_cost = float(np.dot(w, svc))
    saving = source_cost - plan_cost
    true_storage = float(plan["true_storage_mb"].sum()) if plan is not None and len(plan) else 0.0
    return {
        "true_weighted_service_s": plan_cost,
        "source_weighted_service_s": source_cost,
        "true_weighted_saving_s": saving,
        "service_speedup": source_cost / max(plan_cost, EPS),
        "true_storage_mb": true_storage,
        "budget_mb": float(instance.budget_mb),
        "budget_utilization": true_storage / max(instance.budget_mb, EPS),
        "budget_violation": bool(true_storage > instance.budget_mb + 1e-9),
        "selected_count": 0 if plan is None else int(len(plan)),
        "replica_served_fraction": float(np.mean(route["served_by"].astype(str).str.startswith("replica:"))) if len(route) else 0.0,
    }


def source_capacity_qps(requests: pd.DataFrame, workers: int) -> float:
    p = requests["weight"].to_numpy(float)
    mean_service = float(np.dot(p, requests["source_decode_cost"].to_numpy(float)))
    return float(workers / max(mean_service, EPS))


def simulate_worker_pool(
    trace: pd.DataFrame,
    route_table: pd.DataFrame,
    workers: int,
    sla_seconds: float = 0.2,
) -> tuple[pd.DataFrame, ServingMetrics]:
    if workers < 1:
        raise ValueError("workers must be >= 1")
    route_map = route_table.set_index("request_idx")[["service_cost_s", "served_by"]]
    q = trace.join(route_map, on="request_idx", rsuffix="_route").copy()
    if q["service_cost_s"].isna().any():
        raise ValueError("trace contains request_idx not present in route table")
    heap = [(0.0, wid) for wid in range(workers)]
    heapq.heapify(heap)
    starts = np.empty(len(q), dtype=float)
    finishes = np.empty(len(q), dtype=float)
    worker_ids = np.empty(len(q), dtype=int)
    busy = np.zeros(workers, dtype=float)
    for i, row in enumerate(q.itertuples(index=False)):
        avail, wid = heapq.heappop(heap)
        arrival = float(row.arrival_time)
        service = max(float(row.service_cost_s), 0.0)
        start = max(arrival, avail)
        finish = start + service
        starts[i], finishes[i], worker_ids[i] = start, finish, wid
        busy[wid] += service
        heapq.heappush(heap, (finish, wid))
    q["worker_id"] = worker_ids
    q["start_time"] = starts
    q["finish_time"] = finishes
    q["queue_delay_s"] = starts - q["arrival_time"].to_numpy(float)
    q["latency_s"] = finishes - q["arrival_time"].to_numpy(float)
    first_arrival = float(q["arrival_time"].min()) if len(q) else 0.0
    last_finish = float(q["finish_time"].max()) if len(q) else first_arrival
    makespan = max(last_finish - first_arrival, EPS)
    inter_span = max(float(q["arrival_time"].max()) - first_arrival, EPS) if len(q) > 1 else makespan
    arrival_rate = (len(q) - 1) / inter_span if len(q) > 1 else 0.0
    lat = q["latency_s"].to_numpy(float)
    wait = q["queue_delay_s"].to_numpy(float)
    metrics = ServingMetrics(
        query_count=len(q),
        workers=workers,
        arrival_rate=float(arrival_rate),
        achieved_throughput_qps=float(len(q) / makespan) if len(q) else 0.0,
        mean_latency_s=float(np.mean(lat)) if len(lat) else 0.0,
        p50_latency_s=float(np.quantile(lat, 0.50)) if len(lat) else 0.0,
        p95_latency_s=float(np.quantile(lat, 0.95)) if len(lat) else 0.0,
        p99_latency_s=float(np.quantile(lat, 0.99)) if len(lat) else 0.0,
        mean_queue_s=float(np.mean(wait)) if len(wait) else 0.0,
        p95_queue_s=float(np.quantile(wait, 0.95)) if len(wait) else 0.0,
        sla_miss_rate=float(np.mean(lat > float(sla_seconds))) if len(lat) else 0.0,
        utilization=float(busy.sum() / max(workers * makespan, EPS)),
        makespan_s=float(makespan),
    )
    return q, metrics
