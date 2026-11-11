from __future__ import annotations

"""Adaptations of original *video* storage algorithms to the four-CSV protocol.

VSA (Song et al., ETRI Journal 2009):
    Multiple-choice knapsack dynamic programming over feasible VERSION SETS
    for each video. Original versions remain available without consuming the
    additional replica budget. Candidate-version cardinality and capacity
    precision are bounded explicitly to handle a large library.

VSS (Haynes et al., SIGMOD 2021):
    Online read-driven representation caching, with per-video budgets,
    an additional fair global budget, LRU_VSS-style redundancy-aware
    eviction, and permanently retained original video. Because there are
    no GOP positions, frames, quality/PSNR samples or cross-camera overlap
    data, this is an explicitly named WHOLE-VIDEO adaptation: neither
    temporal position offsets nor PSNR protection nor joint compression are
    simulated or invented.

All planning uses instance.storage_col and instance.decode_col (predictions).
No observed candidate cost is read here, even though the formal candidates
carry such columns for the external evaluation and true budget guard.
"""

from dataclasses import dataclass
from collections import defaultdict
from heapq import heappush, heappop, heapify
import hashlib
import math
import time
import weakref

import numpy as np
import pandas as pd

from .coupling import OptimizationInstance
from .optimization import PlanResult, evaluate_objective

EPS = 1e-12


@dataclass(frozen=True)
class _VersionSet:
    ids: tuple[int, ...]
    storage_mb: float
    benefit_s: float


def _final_plan(instance: OptimizationInstance, ids: list[int], method: str,
                started: float, metadata: dict) -> PlanResult:
    if len(ids) != len(set(ids)):
        raise AssertionError("A physical representation cannot be selected twice")
    order = {c: i for i, c in enumerate(ids)}
    plan = instance.candidates[instance.candidates["cand_idx"].isin(order)].copy()
    if len(plan):
        plan["selected_order"] = plan["cand_idx"].map(order)
        plan = plan.sort_values("selected_order").reset_index(drop=True)
    predicted_storage = float(plan[instance.storage_col].sum()) if len(plan) else 0.0
    if predicted_storage > instance.budget_mb + 1e-8:
        raise AssertionError("A prior-work planner exceeded its predicted budget")
    objective = evaluate_objective(instance, plan)
    return PlanResult(
        method=method,
        plan=plan,
        predicted_objective=float(objective),
        storage_used=float(predicted_storage),
        runtime_s=time.perf_counter() - started,
        metadata={
            **metadata,
            "selection_cost_basis": f"{instance.storage_col}+{instance.decode_col}",
            "candidate_pruning": False,
        },
    )


class PriorVSAPlanner:
    """VSA-style per-video feasible version sets + multiple-choice knapsack DP.

    The original algorithm enumerates all possible versions of each video
    (its input has a small number of discrete quality levels). Large CSV
    repositories can have hundreds of variants per video, hence we first
    choose a documented, diverse set of <=max_versions representative
    versions. Every subset of those versions is evaluated exactly under the
    current *predicted* min-cost service model. The resulting grouped options
    are optimized by a global multiple-choice knapsack DP.

    Storage is conservatively rounded UP to DP capacity units. Thus the DP
    approximation never authorizes a plan over the predicted budget.
    """

    def __init__(self, *, max_versions: int = 4, capacity_bins: int = 2048):
        if max_versions < 1 or max_versions > 10:
            raise ValueError("VSA max_versions must be in [1,10]")
        if capacity_bins < 1:
            raise ValueError("VSA capacity_bins must be positive")
        self.max_versions = int(max_versions)
        self.capacity_bins = int(capacity_bins)
        self._cached_candidate_ref = None
        self._cached_request_ref = None
        self._cached_version_sets: tuple[list[list[_VersionSet]], int] | None = None

    def _representatives(self, group: pd.DataFrame, instance: OptimizationInstance) -> list[int]:
        cand = group.loc[
            (group["singleton_gain"] > EPS)
            & np.isfinite(group[instance.storage_col])
            & (group[instance.storage_col] > EPS)
            & np.isfinite(group[instance.decode_col])
        ]
        if cand.empty:
            return []

        # Reflect multiple-quality version combinations rather than merely
        # choosing the WARM density-sorted pool. Include versions preferred
        # by distinct video requests, cheap versions, and fast versions.
        chosen: list[int] = []
        seen: set[int] = set()

        def add(candidate_id: int) -> None:
            if candidate_id not in seen and len(chosen) < self.max_versions:
                chosen.append(candidate_id)
                seen.add(candidate_id)

        # Priority: average request/popularity exposure, then coverage of
        # requested specifications; subsequently resource diversity.
        weighted = cand.sort_values(
            ["singleton_gain", "cand_idx"], ascending=[False, True], kind="stable"
        )
        density = cand.sort_values(
            ["singleton_density", "cand_idx"], ascending=[False, True], kind="stable"
        )
        compact = cand.sort_values(
            [instance.storage_col, "cand_idx"], ascending=[True, True], kind="stable"
        )
        fastest = cand.sort_values(
            [instance.decode_col, "cand_idx"], ascending=[True, True], kind="stable"
        )

        for ranked in (weighted, density, compact, fastest):
            add(int(ranked.iloc[0]["cand_idx"]))

        if len(chosen) < self.max_versions:
            for candidate_id in density["cand_idx"].to_numpy(np.int64):
                add(int(candidate_id))
                if len(chosen) == self.max_versions:
                    break
        return chosen

    def _version_sets(self, instance: OptimizationInstance) -> tuple[list[list[_VersionSet]], int]:
        c = instance.candidates
        req = instance.requests
        src = req["source_decode_cost"].to_numpy(float)
        weights = req["weight"].to_numpy(float)
        decode = c.set_index("cand_idx")[instance.decode_col].to_dict()
        storage = c.set_index("cand_idx")[instance.storage_col].to_dict()
        positions = {
            video_id: gr["request_idx"].to_numpy(np.int64)
            for video_id, gr in req.groupby("VID", sort=False)
        }

        groups: list[list[_VersionSet]] = []
        total_screened = 0

        for video_id, cg in c.groupby("VID", sort=True):
            ids = self._representatives(cg, instance)
            if not ids:
                groups.append([_VersionSet((), 0.0, 0.0)])
                continue
            total_screened += len(ids)
            rpos = positions.get(video_id, np.empty(0, dtype=np.int64))
            if not len(rpos):
                groups.append([_VersionSet((), 0.0, 0.0)])
                continue

            source_local = src[rpos]
            w_local = weights[rpos]
            local_idx = {int(r): i for i, r in enumerate(rpos)}
            contributions: list[np.ndarray] = []
            for candidate_id in ids:
                d = float(decode[candidate_id])
                saving = np.zeros(len(rpos), dtype=float)
                for r in instance.compatibility.coverage.get(candidate_id, np.empty(0, np.int64)):
                    loc = local_idx.get(int(r))
                    if loc is not None:
                        saving[loc] = max(float(source_local[loc]) - d, 0.0)
                contributions.append(saving)
            # A subset's service saving is the pointwise maximum of its
            # constituting physical representations, NOT the sum of singletons.
            opts = [_VersionSet((), 0.0, 0.0)]
            for mask in range(1, 1 << len(ids)):
                items = tuple(ids[i] for i in range(len(ids)) if mask & (1 << i))
                amount = sum(float(storage[candidate_id]) for candidate_id in items)
                if not math.isfinite(amount) or amount <= EPS:
                    continue
                saving = np.maximum.reduce([
                    contributions[i] for i in range(len(ids)) if mask & (1 << i)
                ])
                benefit = float(np.dot(w_local, saving))
                if benefit > EPS:
                    opts.append(_VersionSet(items, amount, benefit))

            # Safe option dominance: if a cheaper-or-equal version SET saves
            # at least as much, the other set is never optimal for the DP.
            opts.sort(key=lambda o: (o.storage_mb, -o.benefit_s, o.ids))
            nondominated: list[_VersionSet] = []
            best_so_far = -1.0
            for opt in opts:
                if opt.benefit_s > best_so_far + EPS or not opt.ids:
                    nondominated.append(opt)
                    best_so_far = max(best_so_far, opt.benefit_s)
            groups.append(nondominated)

        return groups, total_screened

    def optimize(self, instance: OptimizationInstance,
                 method: str = "prior_vsa_dp") -> PlanResult:
        started = time.perf_counter()
        if instance.storage_col == "true_storage_mb" or instance.decode_col == "true_decode_s":
            raise ValueError("VSA adapter requires predicted planning costs")
        budget = float(instance.budget_mb)
        if budget <= EPS or instance.candidates.empty:
            return _final_plan(instance, [], method, started, {
                "baseline_family": "VSA-style multiple-choice knapsack",
                "vsa_dp_state_count": 0,
                "vsa_version_set_count": 0,
            })

        # Across budget sweeps the candidates and workload are identical.
        # Weak references guarantee that a new workload cannot accidentally
        # reuse stale options, while not retaining the large candidate table.
        cache_hit = (self._cached_candidate_ref is not None
                and self._cached_request_ref is not None
                and self._cached_candidate_ref() is instance.candidates
                and self._cached_request_ref() is instance.requests
                and self._cached_version_sets is not None)
        if cache_hit:
            video_opts, screened = self._cached_version_sets
        else:
            video_opts, screened = self._version_sets(instance)
            self._cached_candidate_ref = weakref.ref(instance.candidates)
            self._cached_request_ref = weakref.ref(instance.requests)
            self._cached_version_sets = (video_opts, screened)
        slots = self.capacity_bins
        quantum = budget / float(slots)
        values = np.full(slots + 1, -np.inf, dtype=np.float64)
        values[0] = 0.0
        # Every video has <=2^max_versions version-set options.
        back = np.full((len(video_opts), slots + 1), -1, dtype=np.int16)
        integer_costs: list[list[int]] = []

        for vid_pos, options in enumerate(video_opts):
            next_values = np.full(slots + 1, -np.inf, dtype=np.float64)
            costs: list[int] = []
            choice = back[vid_pos]
            for option_index, option in enumerate(options):
                unit = (0 if not option.ids
                        else max(1, int(math.ceil(option.storage_mb / quantum - 1e-12))))
                costs.append(unit)
                if unit > slots or option.storage_mb > budget + EPS:
                    continue
                candidate_values = values[:slots + 1 - unit] + option.benefit_s
                window = next_values[unit:]
                improve = candidate_values > window + EPS
                if np.any(improve):
                    window[improve] = candidate_values[improve]
                    choice[unit:][improve] = option_index
            values = next_values
            integer_costs.append(costs)

        capacity = int(np.argmax(values))
        selected: list[int] = []
        for video_index in range(len(video_opts) - 1, -1, -1):
            which = int(back[video_index, capacity])
            if which < 0:
                raise AssertionError("VSA MCKP dynamic-programming traceback failed")
            option = video_opts[video_index][which]
            selected.extend(option.ids)
            capacity -= integer_costs[video_index][which]
            if capacity < 0:
                raise AssertionError("VSA MCKP generated an invalid capacity traceback")
        selected.reverse()

        result = _final_plan(instance, selected, method, started, {
            "baseline_family": "VSA-style multiple-choice knapsack",
            "adaptation": "bounded_version_sets+conservative_discretized_dp",
            "vsa_version_set_cache_hit": bool(cache_hit),
            "vsa_max_versions_per_video": self.max_versions,
            "vsa_capacity_bins": self.capacity_bins,
            "vsa_capacity_quantum_mb": quantum,
            "vsa_candidate_versions_screened": screened,
            "vsa_version_set_count": sum(map(len, video_opts)),
            "vsa_video_count": len(video_opts),
            "vsa_videos_with_selected_versions": len(
                set(instance.candidates.loc[
                    instance.candidates["cand_idx"].isin(selected), "VID"
                ]) if selected else set()
            ),
        })
        return result


class PriorVSSCachePlanner:
    """VSS-style read-driven *whole-video* representation cache.

    Original VSS stores/evicts GOP pages and incorporates position,
    redundancy and a PSNR baseline-quality guarantee in LRU_VSS. Those
    quantities are not observable in the four metadata CSVs. This adapter
    caches whole-video representations, holds the always-present original
    outside the additional budget, uses request-format coverage as a
    documented redundancy proxy, and retains both per-video and global
    (fair-evaluation) capacity checks.

    A historical warm-up trace is sampled independently of the future serving
    trace from workload weights. No future query sequence and no measured
    candidate costs are used in caching decisions.
    """

    def __init__(self, *, history_queries: int = 20_000,
                 per_video_budget_multiplier: float = 10.0,
                 redundancy_weight: float = 1.0,
                 trace_seed: int = 1217):
        if history_queries < 1:
            raise ValueError("VSS history_queries must be positive")
        if per_video_budget_multiplier <= 0:
            raise ValueError("VSS per-video storage multiplier must be positive")
        if redundancy_weight < 0:
            raise ValueError("VSS redundancy weight must be nonnegative")
        self.history_queries = int(history_queries)
        self.per_video_budget_multiplier = float(per_video_budget_multiplier)
        self.redundancy_weight = float(redundancy_weight)
        self.trace_seed = int(trace_seed)
        self._proposal_signature = None
        self._proposal_candidate_ref = None
        self._proposal_request_ref = None
        self._proposal_coverage_ref = None
        self._proposal_vector: np.ndarray | None = None

    def _closest_requested_representations(self, inst: OptimizationInstance) -> np.ndarray:
        """Find a physical version close to each historically requested format.

        When a VSS read misses the materialized cache, it creates the format
        requested by that read (or the closest feasible available configuration).
        The closest-format decision uses dimensions and predicted decoding
        costs rather than a global workload-dependent optimization.
        """
        cand = inst.candidates
        req = inst.requests
        signature = (
            id(cand), id(inst.compatibility),
            len(cand), len(req),
            str(inst.decode_col), str(inst.storage_col),
        )
        if (self._proposal_signature == signature
                and self._proposal_candidate_ref is not None
                and self._proposal_request_ref is not None
                and self._proposal_coverage_ref is not None
                and self._proposal_candidate_ref() is cand
                and self._proposal_request_ref() is req
                and self._proposal_coverage_ref() is inst.compatibility
                and self._proposal_vector is not None):
            return self._proposal_vector

        n = len(req)
        proposal = np.full(n, -1, dtype=np.int64)
        best_surplus = np.full(n, np.inf, dtype=float)
        best_decode = np.full(n, np.inf, dtype=float)
        best_storage = np.full(n, np.inf, dtype=float)

        rw = req["req_w"].to_numpy(float)
        rh = req["req_h"].to_numpy(float)
        rf = req["req_f"].to_numpy(float)
        source = req["source_decode_cost"].to_numpy(float)
        cidx = cand["cand_idx"].to_numpy(np.int64)
        cw = cand["weight_replica"].to_numpy(float)
        ch = cand["height_replica"].to_numpy(float)
        cf = cand["frame_rate_replica"].to_numpy(float)
        cd = cand[inst.decode_col].to_numpy(float)
        cs = cand[inst.storage_col].to_numpy(float)

        for i, candidate_id in enumerate(cidx):
            if not (math.isfinite(float(cs[i])) and cs[i] > EPS
                    and math.isfinite(float(cd[i])) and cd[i] > 0):
                continue
            rr = inst.compatibility.coverage.get(int(candidate_id))
            if rr is None or not len(rr):
                continue
            rr = np.asarray(rr, dtype=np.int64)
            useful = cd[i] < source[rr] - EPS
            if not np.any(useful):
                continue
            rr = rr[useful]
            surplus = (
                np.log(np.maximum(cw[i] / np.maximum(rw[rr], EPS), 1.0))
                + np.log(np.maximum(ch[i] / np.maximum(rh[rr], EPS), 1.0))
                + np.log(np.maximum(cf[i] / np.maximum(rf[rr], EPS), 1.0))
            )
            better = surplus < best_surplus[rr] - 1e-10
            equal = np.abs(surplus - best_surplus[rr]) <= 1e-10
            better |= equal & (cd[i] < best_decode[rr] - EPS)
            equal2 = equal & (np.abs(cd[i] - best_decode[rr]) <= EPS)
            better |= equal2 & (cs[i] < best_storage[rr] - EPS)
            equal3 = equal2 & (np.abs(cs[i] - best_storage[rr]) <= EPS)
            better |= equal3 & (candidate_id < proposal[rr])
            if np.any(better):
                r = rr[better]
                best_surplus[r] = surplus[better]
                best_decode[r] = cd[i]
                best_storage[r] = cs[i]
                proposal[r] = int(candidate_id)

        self._proposal_signature = signature
        self._proposal_candidate_ref = weakref.ref(cand)
        self._proposal_request_ref = weakref.ref(req)
        self._proposal_coverage_ref = weakref.ref(inst.compatibility)
        self._proposal_vector = proposal
        return proposal

    def optimize(self, instance: OptimizationInstance,
                 method: str = "prior_vss_cache") -> PlanResult:
        started = time.perf_counter()
        if instance.storage_col == "true_storage_mb" or instance.decode_col == "true_decode_s":
            raise ValueError("VSS adapter requires predicted planning costs")
        req = instance.requests
        cand = instance.candidates
        if req.empty or cand.empty or instance.budget_mb <= EPS:
            return _final_plan(instance, [], method, started, {
                "baseline_family": "VSS-style whole-video read-driven cache",
                "vss_history_reads": 0,
            })
        source_weights = req["weight"].to_numpy(float)
        weights = np.maximum(source_weights, 0.0)
        weight_sum = float(weights.sum())
        weights = (weights / weight_sum if weight_sum > EPS
                   else np.full(len(req), 1.0 / len(req)))

        # Hash the historical *distribution* to obtain reproducible access
        # order across runs/budgets and different order for each workload seed.
        seed_bytes = hashlib.blake2b(
            source_weights.tobytes(), digest_size=8
        ).digest()
        entropy = int.from_bytes(seed_bytes, "little")
        effective_seed = (entropy ^ self.trace_seed) % (2**63 - 1)
        rng = np.random.default_rng(effective_seed)
        history = rng.choice(
            len(req), size=self.history_queries, replace=True, p=weights
        )
        proposal = self._closest_requested_representations(instance)

        ids = cand["cand_idx"].to_numpy(np.int64)
        sizes = cand[instance.storage_col].to_numpy(float)
        costs = cand[instance.decode_col].to_numpy(float)
        videos = cand["VID"].to_numpy()
        src_sizes = cand["storage_cost_source"].to_numpy(float) / 1024.0
        row_for_id = {int(candidate_id): i for i, candidate_id in enumerate(ids)}
        src_sizes_per_video: dict[object, float] = {}
        for i, video_id in enumerate(videos):
            if video_id not in src_sizes_per_video:
                src_sizes_per_video[video_id] = float(src_sizes[i])
        caps = {
            video_id: max(0.0, source_size * self.per_video_budget_multiplier)
            for video_id, source_size in src_sizes_per_video.items()
        }

        cache: dict[int, dict] = {}
        by_video: dict[object, set[int]] = defaultdict(set)
        used_by_video: dict[object, float] = defaultdict(float)
        total_used = 0.0
        event_order: list[int] = []
        heap: list[tuple[float, int, int, int]] = []
        counter = 0
        coverage_bits = instance.compatibility.coverage_bits

        def redundancy(candidate_id: int) -> int:
            """Only a logical-coverage proxy is available, not VSS PSNR."""
            row = row_for_id[candidate_id]
            video_id = videos[row]
            bit = int(coverage_bits.get(candidate_id, 0))
            if bit == 0:
                return 0
            score = 0
            for other in by_video.get(video_id, ()):
                if other == candidate_id:
                    continue
                obit = int(coverage_bits.get(other, 0))
                if (obit | bit) == obit:
                    score += 1
            return score

        def eviction_priority(candidate_id: int) -> float:
            return cache[candidate_id]["last"] - self.redundancy_weight * redundancy(candidate_id)

        def push(candidate_id: int) -> None:
            nonlocal counter
            counter += 1
            cache[candidate_id]["version"] = counter
            heappush(
                heap, (eviction_priority(candidate_id), cache[candidate_id]["last"], counter, candidate_id)
            )

        def refresh_video(video_id) -> None:
            for candidate_id in sorted(by_video.get(video_id, ())):
                push(candidate_id)

        def evict(candidate_id: int) -> None:
            nonlocal total_used
            state = cache.pop(candidate_id)
            video_id = state["VID"]
            by_video[video_id].remove(candidate_id)
            used_by_video[video_id] -= state["storage"]
            total_used -= state["storage"]
            refresh_video(video_id)

        def evict_lru(candidate_vid=None) -> bool:
            if candidate_vid is not None:
                choices = by_video.get(candidate_vid, set())
                if not choices:
                    return False
                victim = min(
                    choices, key=lambda c: (
                        eviction_priority(c), cache[c]["last"], c
                    )
                )
                evict(victim)
                return True
            while heap:
                _, _, version, candidate_id = heappop(heap)
                if candidate_id in cache and cache[candidate_id]["version"] == version:
                    evict(candidate_id)
                    return True
            return False

        hits = admissions = evictions = oversize = cold_misses = 0

        for tick, request_idx in enumerate(history, start=1):
            ri = int(request_idx)
            video_id = req.iloc[ri]["VID"]
            present = by_video.get(video_id, set())
            # Prefer an already cached version that can answer the read.
            best_cached = -1
            best_cost = float(req.iloc[ri]["source_decode_cost"])
            for candidate_id in present:
                if int(instance.compatibility.coverage_bits.get(candidate_id, 0)) == 0:
                    continue
                covered = instance.compatibility.coverage.get(candidate_id)
                if covered is not None and np.any(covered == ri):
                    d = float(costs[row_for_id[candidate_id]])
                    if d < best_cost - EPS:
                        best_cost = d
                        best_cached = candidate_id
            if best_cached >= 0:
                hits += 1
                cache[best_cached]["last"] = tick
                push(best_cached)
                continue

            cold_misses += 1
            candidate_id = int(proposal[ri])
            if candidate_id < 0 or candidate_id in cache:
                continue
            row = row_for_id.get(candidate_id)
            if row is None:
                continue
            s = float(sizes[row])
            if not math.isfinite(s) or s <= EPS:
                continue
            if s > float(instance.budget_mb) + EPS or s > caps.get(video_id, 0.0) + EPS:
                oversize += 1
                continue

            while used_by_video[video_id] + s > caps[video_id] + EPS:
                if not evict_lru(video_id):
                    break
                evictions += 1
            while total_used + s > instance.budget_mb + EPS:
                if not evict_lru():
                    break
                evictions += 1

            if (used_by_video[video_id] + s > caps[video_id] + EPS
                    or total_used + s > instance.budget_mb + EPS):
                continue
            cache[candidate_id] = {
                "last": tick, "VID": video_id, "storage": s, "version": 0,
            }
            by_video[video_id].add(candidate_id)
            used_by_video[video_id] += s
            total_used += s
            event_order.append(candidate_id)
            refresh_video(video_id)
            admissions += 1

            # Bound stale heap records: never retain an in-memory log of all
            # historical accesses or LRU changes.
            if len(heap) > 8 * max(len(cache), 1) + 2048:
                heap.clear()
                for current_cid in sorted(cache):
                    push(current_cid)

        # Cache snapshot at end of historical warmup. Evaluated on a separate
        # future trace by the shared true-cost serving simulation.
        # True-budget commit guard attempts the freshest cache entries first,
        # matching VSS's retention preference if prediction errors cause an
        # unexpected physical-size overrun.
        ordered = sorted(cache, key=lambda candidate_id: (-cache[candidate_id]["last"], candidate_id))
        return _final_plan(instance, ordered, method, started, {
            "baseline_family": "VSS-style whole-video read-driven cache",
            "adaptation": "whole_video_LRU_VSS_redundancy_proxy_no_GOP_or_PSNR",
            "vss_history_reads": int(self.history_queries),
            "vss_trace_seed": int(effective_seed),
            "vss_cache_hits": int(hits),
            "vss_cold_misses": int(cold_misses),
            "vss_admissions": int(admissions),
            "vss_evictions": int(evictions),
            "vss_oversize_rejects": int(oversize),
            "vss_cache_entries": int(len(ordered)),
            "vss_per_video_budget_multiplier": self.per_video_budget_multiplier,
            "vss_global_budget_mb": float(instance.budget_mb),
            "vss_surrogate_redundancy_weight": self.redundancy_weight,
            "vss_gop_position_offset_available": False,
            "vss_psnr_data_available": False,
            "vss_joint_compression_available": False,
        })
