from __future__ import annotations

import hashlib
import heapq
import time
from bisect import bisect_right
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import numpy as np
import pandas as pd

from .coupling import OptimizationInstance
from .optimization import PlanResult
from .mvs_greedy_mk import MVSGreedyMKPlanner

EPS = 1e-12


def _logical_key(row) -> tuple[int, int, float]:
    return int(round(float(row.weight_replica))), int(round(float(row.height_replica))), round(float(row.frame_rate_replica), 6)


def _request_key(row) -> tuple[int, int, float]:
    return int(round(float(row.req_w))), int(round(float(row.req_h))), round(float(row.req_f), 6)


def _uniform_requests(instance: OptimizationInstance) -> pd.DataFrame:
    req = instance.requests.copy()
    req["weight"] = 1.0 / max(len(req), 1)
    return req


def _service_costs(instance: OptimizationInstance, plan_ids: set[int]) -> np.ndarray:
    req = instance.requests
    current = req["source_decode_cost"].to_numpy(float).copy()
    # Planning objective uses estimated costs; final evaluation separately uses true costs.
    cost_map = instance.candidates.set_index("cand_idx")[instance.decode_col].astype(float).to_dict()
    for cidx in plan_ids:
        ridx = instance.compatibility.coverage.get(int(cidx), np.empty(0, dtype=np.int64))
        if len(ridx) and cidx in cost_map:
            current[ridx] = np.minimum(current[ridx], cost_map[cidx])
    return current


def _repair_to_budget_unweighted(instance: OptimizationInstance, plan_ids: set[int], budget: float) -> set[int]:
    """Fast exact whole-replica budget repair for the static VStore adaptation.

    The original implementation recomputed the complete service-cost vector once
    for *every candidate* at *every removal step*.  That is cubic-like in the
    number of selected replicas.  Here we maintain, for each request, the active
    providers sorted by the planner's selected decode cost and the exact marginal degradation caused
    by removing its current unique best provider.  After a removal only requests
    covered by that replica need to be updated.

    The score is unchanged from the original implementation:

        uniform mean service degradation / MB freed.

    No WARM workload frequencies are used.
    """
    current_ids = set(int(c) for c in plan_ids)
    if not current_ids:
        return current_ids

    cand = instance.candidates
    storage_map = cand.set_index("cand_idx")[instance.storage_col].astype(float).to_dict()
    decode_map = cand.set_index("cand_idx")[instance.decode_col].astype(float).to_dict()
    used = float(sum(storage_map.get(c, 0.0) for c in current_ids))
    if used <= budget + EPS:
        return current_ids

    source = instance.requests["source_decode_cost"].to_numpy(float)
    request_count = len(source)

    # Pre-build the provider list for every request.  Lists remain sorted and
    # deletions are handled lazily through the `active` set.
    providers: list[list[tuple[float, int]]] = [[] for _ in range(request_count)]
    for cidx in current_ids:
        tau = float(decode_map[cidx])
        for ridx in instance.compatibility.coverage.get(cidx, np.empty(0, dtype=np.int64)):
            providers[int(ridx)].append((tau, cidx))
    for p in providers:
        p.sort(key=lambda x: (x[0], x[1]))

    active = current_ids
    owner = np.full(request_count, -1, dtype=np.int64)
    contribution = np.zeros(request_count, dtype=float)
    penalty_sum = {c: 0.0 for c in active}

    def request_penalty_owner(ridx: int) -> tuple[int, float]:
        """Return candidate whose removal raises this request's cost, and by how much."""
        first_c = -1
        first_tau = float("inf")
        second_tau = float("inf")
        for tau, cidx in providers[ridx]:
            if cidx not in active:
                continue
            if first_c < 0:
                first_c = cidx
                first_tau = tau
            else:
                second_tau = tau
                break

        src = float(source[ridx])
        if first_c < 0 or first_tau >= src:
            return -1, 0.0

        next_cost = min(src, second_tau)
        delta = max(next_cost - first_tau, 0.0)
        if delta <= 0.0:
            return -1, 0.0
        return first_c, delta

    for ridx in range(request_count):
        cidx, delta = request_penalty_owner(ridx)
        owner[ridx] = cidx
        contribution[ridx] = delta
        if cidx >= 0:
            penalty_sum[cidx] += delta

    # Heap with versioned lazy updates.  Division by request_count is common to
    # all candidates, so it is omitted without changing the ordering.
    version = {c: 0 for c in active}
    heap: list[tuple[float, int, int]] = []

    def push_score(cidx: int) -> None:
        if cidx not in active:
            return
        score = penalty_sum.get(cidx, 0.0) / max(float(storage_map.get(cidx, 0.0)), EPS)
        heapq.heappush(heap, (score, version[cidx], cidx))

    for cidx in active:
        push_score(cidx)

    while active and used > budget + EPS:
        best_remove = None
        while heap:
            _score, ver, cidx = heapq.heappop(heap)
            if cidx in active and version.get(cidx, -1) == ver:
                best_remove = cidx
                break
        if best_remove is None:
            break

        active.remove(best_remove)
        used -= float(storage_map.get(best_remove, 0.0))

        touched: set[int] = set()
        for ridx_raw in instance.compatibility.coverage.get(best_remove, np.empty(0, dtype=np.int64)):
            ridx = int(ridx_raw)
            old_owner = int(owner[ridx])
            old_delta = float(contribution[ridx])
            if old_owner >= 0:
                penalty_sum[old_owner] -= old_delta
                # Numerical cleanup only; score semantics are unchanged.
                if abs(penalty_sum[old_owner]) < 1e-15:
                    penalty_sum[old_owner] = 0.0
                if old_owner in active:
                    touched.add(old_owner)

            new_owner, new_delta = request_penalty_owner(ridx)
            owner[ridx] = new_owner
            contribution[ridx] = new_delta
            if new_owner >= 0:
                penalty_sum[new_owner] += new_delta
                if new_owner in active:
                    touched.add(new_owner)

        for cidx in touched:
            version[cidx] += 1
            push_score(cidx)

    return set(active)


def _result(instance: OptimizationInstance, ids: Iterable[int], method: str, start: float, metadata: dict | None = None) -> PlanResult:
    ids = set(int(x) for x in ids)
    plan = instance.candidates[instance.candidates["cand_idx"].isin(ids)].copy()
    req = instance.requests
    source = req["source_decode_cost"].to_numpy(float)
    service = _service_costs(instance, ids)
    obj = float(np.dot(req["weight"].to_numpy(float), np.maximum(source - service, 0.0)))
    used = float(plan[instance.storage_col].sum()) if len(plan) else 0.0
    return PlanResult(method, plan.reset_index(drop=True), obj, used, time.perf_counter() - start, metadata=metadata or {})


class VStoreGoldenPlanner:
    """VStore-like `1 -> N` / golden-format baseline adapted to WARM."""
    def optimize(self, instance: OptimizationInstance, method: str = "vstore_golden") -> PlanResult:
        start = time.perf_counter()
        selected = []
        profiled = 0
        for _vid, group in instance.candidates.groupby("VID", sort=False):
            pixels = (group["weight_replica"].to_numpy(float) * group["height_replica"].to_numpy(float))
            max_pixels = pixels.max()
            richest_mask = pixels == max_pixels
            richest = group.loc[richest_mask]
            max_fps = richest["frame_rate_replica"].max()
            richest = richest[richest["frame_rate_replica"] == max_fps]
            profiled += len(richest)
            row = richest.nsmallest(1, [instance.decode_col, instance.storage_col]).iloc[0]
            selected.append(int(row["cand_idx"]))
        storage = instance.candidates.set_index("cand_idx")[instance.storage_col].to_dict()
        admitted, used = [], 0.0
        for c in sorted(selected, key=lambda x: (storage[x], x)):
            if used + storage[c] <= instance.budget_mb + EPS:
                admitted.append(c)
                used += storage[c]
        return _result(instance, admitted, method, start, {"profiled_candidates": profiled, "vstore_variant": "golden_1_to_N"})


class VStoreNToNPlanner:
    """VStore-like one-storage-format-per-consumer baseline."""
    def optimize(self, instance: OptimizationInstance, method: str = "vstore_n2n") -> PlanResult:
        start = time.perf_counter()
        selected: set[int] = set()
        profiled: set[int] = set()
        cand_by_vid = {video_id: g for video_id, g in instance.candidates.groupby("VID", sort=False)}

        # Reverse compatibility index removes the original O(requests*candidates)
        # search when an exact logical format is absent.
        request_to_candidates: list[list[int]] = [[] for _ in range(len(instance.requests))]
        for cidx, ridx in instance.compatibility.coverage.items():
            for r in ridx:
                request_to_candidates[int(r)].append(int(cidx))

        for row in instance.requests.itertuples(index=False):
            group = cand_by_vid.get(row.VID)
            if group is None or len(group) == 0:
                continue
            w = group["weight_replica"].to_numpy(float)
            h = group["height_replica"].to_numpy(float)
            f = group["frame_rate_replica"].to_numpy(float)
            exact_mask = (
                (np.rint(w).astype(np.int64) == int(round(row.req_w)))
                & (np.rint(h).astype(np.int64) == int(round(row.req_h)))
                & np.isclose(f, float(row.req_f), atol=1e-6)
            )
            if exact_mask.any():
                pool = group.loc[exact_mask]
            else:
                cids = request_to_candidates[int(row.request_idx)]
                if not cids:
                    continue
                pool = group[group["cand_idx"].isin(cids)]
            if len(pool) == 0:
                continue
            profiled.update(int(x) for x in pool["cand_idx"])
            best = pool.nsmallest(1, [instance.decode_col, instance.storage_col]).iloc[0]
            selected.add(int(best["cand_idx"]))
        repaired = _repair_to_budget_unweighted(instance, selected, instance.budget_mb)
        return _result(instance, repaired, method, start, {
            "profiled_candidates": len(profiled),
            "pre_erosion_count": len(selected),
            "vstore_variant": "N_to_N_consumers",
        })


@dataclass
class _FormatState:
    cand_idx: int
    consumer_bits: int
    threshold_s: float


@dataclass
class _PreparedCoalescing:
    selected: set[int]
    profiled_candidates: int
    merge_count: int
    build_seconds: float


class VStoreCoalescingPlanner:
    """Efficient VStore-style storage-format coalescing baseline.

    Semantics are the same as the original adaptation: per-consumer initial
    formats + golden format, repeated storage-nonincreasing pairwise merges,
    then workload-agnostic whole-replica repair to the shared budget.

    Efficiency changes:
      * local request bitmasks replace Python set conversions in inner loops;
      * per-video candidate arrays/frontiers replace repeated pandas filtering;
      * merge queries are memoized;
      * the budget repair is incremental rather than full recomputation;
      * budget-independent coalescing is cached and reused across budget sweeps.
    """
    def __init__(self, free_merge_only: bool = False):
        self.free_merge_only = free_merge_only
        self._prepared_key: bytes | None = None
        self._prepared: _PreparedCoalescing | None = None

    @staticmethod
    def _signature(instance: OptimizationInstance) -> bytes:
        h = hashlib.blake2b(digest_size=16)
        ccols = ["cand_idx", "VID", "weight_replica", "height_replica", "frame_rate_replica", instance.storage_col, instance.decode_col]
        rcols = ["request_idx", "VID", "req_w", "req_h", "req_f", "source_decode_cost"]
        ch = pd.util.hash_pandas_object(instance.candidates[ccols], index=False).to_numpy(np.uint64, copy=False)
        rh = pd.util.hash_pandas_object(instance.requests[rcols], index=False).to_numpy(np.uint64, copy=False)
        h.update(ch.tobytes())
        h.update(rh.tobytes())
        return h.digest()

    def _prepare(self, instance: OptimizationInstance) -> _PreparedCoalescing:
        prep_start = time.perf_counter()
        all_selected: set[int] = set()
        profiled: set[int] = set()
        merge_count = 0
        request_groups = {video_id: g for video_id, g in instance.requests.groupby("VID", sort=False)}

        for video_id, cand in instance.candidates.groupby("VID", sort=False):
            req = request_groups.get(video_id)
            if req is None or len(req) == 0:
                continue

            cids = cand["cand_idx"].to_numpy(np.int64)
            stor = cand[instance.storage_col].to_numpy(float)
            dec = cand[instance.decode_col].to_numpy(float)
            w = cand["weight_replica"].to_numpy(float)
            hh = cand["height_replica"].to_numpy(float)
            fps = cand["frame_rate_replica"].to_numpy(float)
            n = len(cids)
            pos_by_cid = {int(c): i for i, c in enumerate(cids)}

            req_indices = req["request_idx"].to_numpy(np.int64)
            local_pos = {int(r): j for j, r in enumerate(req_indices)}
            local_cov = np.zeros(n, dtype=object)
            for k, cidx in enumerate(cids):
                mask = 0
                for ridx in instance.compatibility.coverage.get(int(cidx), np.empty(0, dtype=np.int64)):
                    p = local_pos.get(int(ridx))
                    if p is not None:
                        mask |= 1 << p
                local_cov[k] = mask

            # Candidate indices grouped by local coverage mask.  Merge queries use
            # a small Pareto frontier (decode asc, storage strictly decreasing).
            by_mask: dict[int, list[int]] = {}
            for k, mask_obj in enumerate(local_cov):
                by_mask.setdefault(int(mask_obj), []).append(k)

            frontiers: dict[int, tuple[list[float], list[int], list[int]]] = {}
            # tuple: frontier decode values, frontier candidate positions, all
            # candidate positions sorted by decode (for exact profiling metadata)
            for mask, positions in by_mask.items():
                ordered = sorted(positions, key=lambda k: (dec[k], stor[k], int(cids[k])))
                frontier_pos: list[int] = []
                frontier_dec: list[float] = []
                best_storage = float("inf")
                for k in ordered:
                    if stor[k] < best_storage - EPS:
                        frontier_pos.append(k)
                        frontier_dec.append(float(dec[k]))
                        best_storage = float(stor[k])
                frontiers[mask] = (frontier_dec, frontier_pos, ordered)

            # Initial N->N states.
            states: list[_FormatState] = []
            for local_r, rr in enumerate(req.itertuples(index=False)):
                bit = 1 << local_r
                compatible_pos = [k for k in range(n) if int(local_cov[k]) & bit]
                if not compatible_pos:
                    continue
                exact_pos = [
                    k for k in compatible_pos
                    if int(round(w[k])) == int(round(rr.req_w))
                    and int(round(hh[k])) == int(round(rr.req_h))
                    and np.isclose(fps[k], float(rr.req_f), atol=1e-6)
                ]
                pool_pos = exact_pos if exact_pos else compatible_pos
                profiled.update(int(cids[k]) for k in pool_pos)
                best_k = min(pool_pos, key=lambda k: (dec[k], stor[k], int(cids[k])))
                states.append(_FormatState(int(cids[best_k]), bit, float(dec[best_k])))

            # Dedupe initial states that selected the same physical format.
            merged_by_id: dict[int, _FormatState] = {}
            for s in states:
                prev = merged_by_id.get(s.cand_idx)
                if prev is None:
                    merged_by_id[s.cand_idx] = _FormatState(s.cand_idx, s.consumer_bits, s.threshold_s)
                else:
                    prev.consumer_bits |= s.consumer_bits
                    prev.threshold_s = min(prev.threshold_s, s.threshold_s)
            states = list(merged_by_id.values())

            # Golden richest logical format.
            pixels = w * hh
            max_pixels = pixels.max()
            richest_pos = np.flatnonzero(pixels == max_pixels)
            max_fps = fps[richest_pos].max()
            golden_pos = [int(k) for k in richest_pos if fps[int(k)] == max_fps]
            profiled.update(int(cids[k]) for k in golden_pos)
            golden_k = min(golden_pos, key=lambda k: (dec[k], stor[k], int(cids[k])))
            golden_cid = int(cids[golden_k])
            if golden_cid not in {s.cand_idx for s in states}:
                states.append(_FormatState(golden_cid, 0, float(dec[golden_k])))

            merge_cache: dict[tuple[int, float], int | None] = {}

            def best_merge_candidate(union_bits: int, threshold: float) -> int | None:
                key = (union_bits, threshold)
                if key in merge_cache:
                    return merge_cache[key]
                best_k: int | None = None
                for cov_mask, (front_dec, front_pos, ordered) in frontiers.items():
                    if (cov_mask & union_bits) != union_bits:
                        continue
                    # All actually feasible candidates count as profiled, matching
                    # the original metadata meaning without dataframe scans.
                    ordered_dec = [dec[k] for k in ordered]
                    cutoff = bisect_right(ordered_dec, threshold + EPS)
                    if cutoff:
                        profiled.update(int(cids[k]) for k in ordered[:cutoff])
                    fi = bisect_right(front_dec, threshold + EPS) - 1
                    if fi < 0:
                        continue
                    k = front_pos[fi]
                    if best_k is None or (stor[k], dec[k], int(cids[k])) < (stor[best_k], dec[best_k], int(cids[best_k])):
                        best_k = k
                ans = None if best_k is None else int(cids[best_k])
                merge_cache[key] = ans
                return ans

            # Greedy pairwise free coalescing.
            while len(states) >= 2:
                best_merge = None
                best_delta = float("inf")
                for i in range(len(states)):
                    for j in range(i + 1, len(states)):
                        a, b = states[i], states[j]
                        union_bits = a.consumer_bits | b.consumer_bits
                        if union_bits == 0:
                            continue
                        threshold = min(a.threshold_s, b.threshold_s)
                        x_cid = best_merge_candidate(union_bits, threshold)
                        if x_cid is None:
                            continue
                        xk = pos_by_cid[x_cid]
                        old_s = stor[pos_by_cid[a.cand_idx]] + stor[pos_by_cid[b.cand_idx]]
                        delta = float(stor[xk] - old_s)
                        if self.free_merge_only and delta > EPS:
                            continue
                        # The original implementation only accepts non-increasing
                        # storage merges even when free_merge_only=False.
                        if delta <= EPS and delta < best_delta:
                            best_delta = delta
                            best_merge = (i, j, _FormatState(x_cid, union_bits, threshold))
                if best_merge is None:
                    break
                i, j, new_state = best_merge
                states = [s for k, s in enumerate(states) if k not in {i, j}] + [new_state]
                merge_count += 1

            all_selected.update(s.cand_idx for s in states)

        return _PreparedCoalescing(
            selected=all_selected,
            profiled_candidates=len(profiled),
            merge_count=merge_count,
            build_seconds=time.perf_counter() - prep_start,
        )

    def optimize(self, instance: OptimizationInstance, method: str = "vstore_coalesce") -> PlanResult:
        start = time.perf_counter()
        key = self._signature(instance)
        cache_hit = self._prepared is not None and self._prepared_key == key
        if not cache_hit:
            self._prepared = self._prepare(instance)
            self._prepared_key = key
        prepared = self._prepared
        assert prepared is not None

        repaired = _repair_to_budget_unweighted(instance, set(prepared.selected), instance.budget_mb)
        return _result(instance, repaired, method, start, {
            "profiled_candidates": prepared.profiled_candidates,
            "coalescing_merges": prepared.merge_count,
            "pre_erosion_count": len(prepared.selected),
            "coalescing_build_s": prepared.build_seconds,
            "coalescing_cache_hit": bool(cache_hit),
            "vstore_variant": "profiled_CF_to_SF_coalescing",
            "selection_cost_basis": f"{instance.storage_col}+{instance.decode_col}",
        })


@lru_cache(maxsize=3)
def vstore_planner(method: str):
    # Reuse planner objects inside one process so the coalescing planner can
    # reuse its budget-independent pre-budget configuration across budget sweeps.
    if method == "vstore_golden":
        return VStoreGoldenPlanner()
    if method == "vstore_n2n":
        return VStoreNToNPlanner()
    if method == "vstore_coalesce":
        return VStoreCoalescingPlanner()
    raise KeyError(method)


class PriorMVSGreedyPlanner(MVSGreedyMKPlanner):
    """Backward-compatible method name for VLDB 2000 Greedy(2,k)-inspired MVS.

    The implementation lives in mvs_greedy_mk.py and is independent of the
    WARM LazyBudgetedGreedy class.
    """


class PriorZhaoVersionPlanner:
    """Popularity-aware proposal of per-request versions plus global admission.

    A per-request best-serving version is proposed using the instance's
    predicted costs. Proposals are admitted by their static popularity-weighted
    singleton savings / predicted storage. This is a VoD pre-storage adaptation,
    not a reproduction of the original segment/transcoding graph or its
    divide-and-conquer storage/transcoding optimization.
    """
    def optimize(self, instance: OptimizationInstance, method: str = "prior_zhao_version") -> PlanResult:
        t0 = time.perf_counter()
        if instance.storage_col == "true_storage_mb" or instance.decode_col == "true_decode_s":
            raise ValueError("VACTS adapter requires predicted planning costs")
        req = instance.requests
        cand = instance.candidates
        n = len(req)
        src = req["source_decode_cost"].to_numpy(float)
        choice = np.full(n, -1, dtype=np.int64)
        best = src.copy()
        best_storage = np.full(n, np.inf, dtype=float)
        cidxs = cand["cand_idx"].to_numpy(np.int64)
        stor = cand[instance.storage_col].to_numpy(float)
        decode = cand[instance.decode_col].to_numpy(float)
        for ix, cidx in enumerate(cidxs):
            if not np.isfinite(stor[ix]) or stor[ix] <= 0 or stor[ix] > instance.budget_mb + EPS:
                continue
            covered = instance.compatibility.coverage.get(int(cidx))
            if covered is None or len(covered) == 0:
                continue
            rr = np.asarray(covered, dtype=np.int64)
            better = decode[ix] < best[rr] - EPS
            equal = (np.abs(decode[ix] - best[rr]) <= EPS) & (stor[ix] < best_storage[rr] - EPS)
            replace_mask = better | equal
            if np.any(replace_mask):
                selected_requests = rr[replace_mask]
                best[selected_requests] = decode[ix]
                best_storage[selected_requests] = stor[ix]
                choice[selected_requests] = cidx
        proposals = set(int(x) for x in choice if int(x) >= 0)
        proposals_df = cand[cand["cand_idx"].isin(proposals)].copy()
        if len(proposals_df):
            proposals_df["_score"] = (
                proposals_df["singleton_gain"].to_numpy(float)
                / np.maximum(proposals_df[instance.storage_col].to_numpy(float), EPS)
            )
            proposals_df = proposals_df.sort_values(
                ["_score", "singleton_gain", "cand_idx"],
                ascending=[False, False, True], kind="stable",
            )
        remaining = float(instance.budget_mb)
        admitted = []
        for row in proposals_df.itertuples(index=False):
            s = float(getattr(row, instance.storage_col))
            if s <= remaining + EPS and float(row.singleton_gain) > EPS:
                admitted.append(int(row.cand_idx))
                remaining -= s
        result = _result(instance, admitted, method, t0, {
            "baseline_family": "Zhao-style popularity-aware multi-version pre-storage",
            "proposed_versions": len(proposals),
            "candidate_pruning": False,
            "selection_cost_basis": f"{instance.storage_col}+{instance.decode_col}",
        })
        if len(result.plan):
            result.plan["selected_order"] = result.plan["cand_idx"].map({x:i for i,x in enumerate(admitted)})
            result.plan.sort_values("selected_order", inplace=True)
            result.plan.reset_index(drop=True, inplace=True)
        return result


_PRIOR_MVS = PriorMVSGreedyPlanner()
_PRIOR_MVS_CONFIG_KEY = (12000, 2, 1, 24, 24)
_PRIOR_ZHAO = PriorZhaoVersionPlanner()


def prior_work_planner(method: str, mvs_config=None):
    """Share a cache-bearing planner across budgets, reconfigure on YAML change."""
    global _PRIOR_MVS, _PRIOR_MVS_CONFIG_KEY
    if method == "prior_mvs_greedy":
        if mvs_config is not None:
            key = (
                int(mvs_config.pool_cap), int(mvs_config.per_video_density),
                int(mvs_config.per_video_gain), int(mvs_config.pair_top_density),
                int(mvs_config.pair_top_gain),
            )
            if key != _PRIOR_MVS_CONFIG_KEY:
                _PRIOR_MVS = PriorMVSGreedyPlanner(
                    pool_cap=key[0], per_video_density=key[1],
                    per_video_gain=key[2], pair_top_density=key[3],
                    pair_top_gain=key[4],
                )
                _PRIOR_MVS_CONFIG_KEY = key
        return _PRIOR_MVS
    if method == "prior_zhao_version":
        return _PRIOR_ZHAO
    raise KeyError(method)
