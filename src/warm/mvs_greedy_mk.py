"""MVS Greedy(2,k)-inspired video physical-design adapter.

Agrawal, Chaudhuri & Narasayya (VLDB 2000) combine candidate screening with
limited configuration enumeration and greedy completion. This is an adaptation
to a *video* representation catalog, not a reproduction of SQL view merging
or hypothetical SQL index optimization.

Planning uses ONLY the columns named by OptimizationInstance.storage_col and
decode_col. The final true-cost assessment and budget guard are handled by the
existing experiments pipeline.
"""
from __future__ import annotations

import heapq
import time
import weakref

import numpy as np
import pandas as pd

from .coupling import OptimizationInstance
from .optimization import PlanResult

EPS = 1.0e-12


class MVSGreedyMKPlanner:
    """Screen candidates, enumerate promising configurations up to m=2,
    then greedily extend the best starting configuration to the budget.

    This intentionally differs from WARM:
      * own query-/video-oriented candidate screening, not dominance pruning;
      * exhaustive one- and two-view configurations in a bounded shortlist;
      * best total-benefit initial configuration (not density-first);
      * remaining candidates come from this screened physical-design catalog.

    Search is lazy but exact *for its candidate pool and initial seed*.
    Candidate screening and seed enumeration are heuristics, as in a
    practical MVS-style physical-design advisor.
    """

    def __init__(
        self,
        pool_cap: int = 12000,
        per_video_density: int = 2,
        per_video_gain: int = 1,
        pair_top_density: int = 24,
        pair_top_gain: int = 24,
        min_gain: float = EPS,
    ):
        if pool_cap < 2:
            raise ValueError("pool_cap must be >=2")
        if per_video_density < 0 or per_video_gain < 0:
            raise ValueError("per-video candidate counts must be nonnegative")
        if pair_top_density < 0 or pair_top_gain < 0:
            raise ValueError("pair shortlist sizes must be nonnegative")
        self.pool_cap = int(pool_cap)
        self.per_video_density = int(per_video_density)
        self.per_video_gain = int(per_video_gain)
        self.pair_top_density = int(pair_top_density)
        self.pair_top_gain = int(pair_top_gain)
        self.min_gain = float(min_gain)
        # Weak cache prevents a planner from retaining one 438K-row DataFrame
        # across the end-to-end suite's per-seed memory release.
        self._cached_dataframe_ref: weakref.ReferenceType[pd.DataFrame] | None = None
        self._cached_ids: tuple[int, ...] = ()
        self._cached_meta: dict[str, int] = {}

    @staticmethod
    def _ranked(df: pd.DataFrame, metric: str) -> list[int]:
        """Deterministic descending ranking with stable id tie-break."""
        if df.empty:
            return []
        x = df.sort_values(
            [metric, "cand_idx"],
            ascending=[False, True],
            kind="stable",
        )
        return x["cand_idx"].astype(int).tolist()

    def _screen(self, instance: OptimizationInstance) -> tuple[list[int], dict[str, int]]:
        cand = instance.candidates
        if self._cached_dataframe_ref is not None and self._cached_dataframe_ref() is cand:
            return list(self._cached_ids), dict(self._cached_meta)

        t0 = time.perf_counter()
        s = cand[instance.storage_col].to_numpy(dtype=float)
        t = cand[instance.decode_col].to_numpy(dtype=float)
        gains = cand["singleton_gain"].to_numpy(dtype=float)
        ids = cand["cand_idx"].to_numpy(dtype=np.int64)
        valid_mask = (
            np.isfinite(s) & (s > 0) & np.isfinite(t) & (t > 0)
            & np.isfinite(gains) & (gains > self.min_gain)
        )
        valid = cand.loc[
            valid_mask,
            ["cand_idx", "VID", instance.storage_col, instance.decode_col,
             "singleton_gain", "singleton_density"],
        ].copy()
        if valid.empty:
            meta = {"query_representatives": 0, "valid_candidates": 0,
                    "pool_candidates": 0}
            self._cached_dataframe_ref = weakref.ref(cand)
            self._cached_ids = ()
            self._cached_meta = meta
            return [], meta

        # Candidate selection, motivated by MVS workload-derived candidates:
        # one fastest useful predicted representation per logical query.
        req_source = instance.requests["source_decode_cost"].to_numpy(float)
        query_reps = np.full(len(req_source), -1, dtype=np.int64)
        order = np.lexsort((ids[valid_mask], s[valid_mask], t[valid_mask]))
        good_ids = ids[valid_mask]
        good_t = t[valid_mask]
        for pos in order:
            cidx = int(good_ids[pos])
            rr = instance.compatibility.coverage.get(cidx)
            if rr is None or len(rr) == 0:
                continue
            rr = np.asarray(rr, dtype=np.int64)
            need = (query_reps[rr] < 0) & (good_t[pos] < req_source[rr] - self.min_gain)
            if np.any(need):
                query_reps[rr[need]] = cidx

        # Preserve per-video diversity before global admission. All rankings
        # use predicted costs; there is no access to true_* fields here.
        density_rank = valid.sort_values(
            ["singleton_density", "cand_idx"],
            ascending=[False, True], kind="stable",
        )
        gain_rank = valid.sort_values(
            ["singleton_gain", "cand_idx"],
            ascending=[False, True], kind="stable",
        )
        by_video_density = (
            density_rank.groupby("VID", sort=False)
            .head(self.per_video_density)["cand_idx"].astype(int).tolist()
            if self.per_video_density else []
        )
        by_video_gain = (
            gain_rank.groupby("VID", sort=False)
            .head(self.per_video_gain)["cand_idx"].astype(int).tolist()
            if self.per_video_gain else []
        )
        request_choice = [int(x) for x in query_reps if x >= 0]
        if request_choice:
            request_set = set(request_choice)
            request_choice = self._ranked(
                valid[valid["cand_idx"].isin(request_set)], "singleton_density"
            )

        pool: list[int] = []
        seen: set[int] = set()

        def take(cidx: int) -> None:
            if len(pool) < self.pool_cap and cidx not in seen:
                seen.add(cidx)
                pool.append(cidx)

        # Representation candidates for individual workload queries,
        # with video diversity retained under the capacity cap.
        for cidx in by_video_density:
            take(cidx)
        for cidx in by_video_gain:
            take(cidx)
        for cidx in request_choice:
            take(cidx)

        # Fill the remaining catalog by alternating globally strongest
        # singleton density and absolute benefit, rather than applying the
        # WARM dominance-pruning procedure.
        density_ids = density_rank["cand_idx"].to_numpy(np.int64)
        gain_ids = gain_rank["cand_idx"].to_numpy(np.int64)
        for d, g in zip(density_ids, gain_ids):
            if len(pool) >= self.pool_cap:
                break
            take(int(d))
            take(int(g))
        if len(pool) < self.pool_cap:
            for cidx in density_ids:
                if len(pool) >= self.pool_cap:
                    break
                take(int(cidx))

        meta = {
            "raw_candidate_count": int(len(cand)),
            "valid_candidates": int(len(valid)),
            "query_representatives": int(len(set(request_choice))),
            "pool_candidates": int(len(pool)),
            "pool_cap": int(self.pool_cap),
            "screening_runtime_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }
        self._cached_dataframe_ref = weakref.ref(cand)
        self._cached_ids = tuple(pool)
        self._cached_meta = dict(meta)
        return pool, meta

    def _enumerate_seed(
        self,
        instance: OptimizationInstance,
        pool: pd.DataFrame,
    ) -> tuple[list[int], float, int, int]:
        """Exhaustively evaluate <=2-view configurations in a bounded shortlist."""
        if pool.empty:
            return [], 0.0, 0, 0
        feasible = pool[
            (pool[instance.storage_col] <= instance.budget_mb + EPS)
            & (pool["singleton_gain"] > self.min_gain)
        ]
        if feasible.empty:
            return [], 0.0, 0, 0

        def top(df: pd.DataFrame, key: str, cap: int) -> list[int]:
            return self._ranked(df, key)[:cap]

        pool_ids = top(feasible, "singleton_density", self.pair_top_density)
        pool_ids += top(feasible, "singleton_gain", self.pair_top_gain)
        shortlist_ids = list(dict.fromkeys(pool_ids))
        # With pair shortlist sizes zero, retain the best individual structure.
        if not shortlist_ids:
            shortlist_ids = top(feasible, "singleton_gain", 1)

        selected = feasible.set_index("cand_idx").loc[shortlist_ids]
        cids = selected.index.to_numpy(np.int64)
        storage = selected[instance.storage_col].to_numpy(float)
        decode = selected[instance.decode_col].to_numpy(float)
        gain = selected["singleton_gain"].to_numpy(float)
        weights = instance.requests["weight"].to_numpy(float)
        source = instance.requests["source_decode_cost"].to_numpy(float)
        cover = instance.compatibility.coverage
        # Configurations up to m=2: compare empty, singleton and feasible pair.
        best_ids: list[int] = []
        best_gain, best_storage = 0.0, float("inf")
        count = 0
        best_key: tuple[int, ...] = ()
        for a in range(len(cids)):
            candidate_id = int(cids[a])
            count += 1
            key = (candidate_id,)
            if (gain[a] > best_gain + EPS or
                (abs(gain[a] - best_gain) <= EPS and
                 (storage[a], key) < (best_storage, best_key))):
                best_ids = [candidate_id]
                best_gain = float(gain[a])
                best_storage = float(storage[a])
                best_key = key

        for a in range(len(cids)):
            ra = cover.get(int(cids[a]), np.empty(0, dtype=np.int32))
            for b in range(a + 1, len(cids)):
                joint_storage = float(storage[a] + storage[b])
                if joint_storage > instance.budget_mb + EPS:
                    continue
                count += 1
                rb = cover.get(int(cids[b]), np.empty(0, dtype=np.int32))
                overlap = np.intersect1d(ra, rb, assume_unique=True)
                duplicate = 0.0
                if len(overlap):
                    duplicate = float(np.dot(
                        weights[overlap],
                        np.minimum(
                            np.maximum(source[overlap] - decode[a], 0.0),
                            np.maximum(source[overlap] - decode[b], 0.0),
                        ),
                    ))
                val = float(gain[a] + gain[b] - duplicate)
                key = tuple(sorted((int(cids[a]), int(cids[b]))))
                if (val > best_gain + EPS or
                    (abs(val - best_gain) <= EPS and
                     (joint_storage, key) < (best_storage, best_key))):
                    best_ids = list(key)
                    best_gain = val
                    best_storage = joint_storage
                    best_key = key

        return best_ids, best_gain, count, len(shortlist_ids)

    def optimize(
        self,
        instance: OptimizationInstance,
        method: str = "prior_mvs_greedy",
    ) -> PlanResult:
        start = time.perf_counter()
        if (
            instance.storage_col in {"true_storage_mb", "true_storage"}
            or instance.decode_col in {"true_decode_s", "true_decode"}
        ):
            raise ValueError("MVS adapter requires predicted planning costs")

        pool_ids, screen_meta = self._screen(instance)
        candidates = instance.candidates
        if not pool_ids:
            empty = candidates.iloc[0:0].copy()
            return PlanResult(method, empty, 0.0, 0.0, time.perf_counter() - start,
                              metadata={"baseline_family": "MVS Greedy(2,k) adaptation",
                                        **screen_meta})

        pool = candidates[candidates["cand_idx"].isin(pool_ids)].copy()
        pool = pool[pool[instance.storage_col] <= instance.budget_mb + EPS].copy()
        seeds, seed_gain, pair_trials, shortlist_count = self._enumerate_seed(
            instance, pool
        )

        req = instance.requests
        weights = req["weight"].to_numpy(float)
        current = req["source_decode_cost"].to_numpy(float).copy()
        storage = pool.set_index("cand_idx")[instance.storage_col].astype(float).to_dict()
        decode = pool.set_index("cand_idx")[instance.decode_col].astype(float).to_dict()
        singleton = pool.set_index("cand_idx")["singleton_gain"].astype(float).to_dict()
        cover = instance.compatibility.coverage

        selected: list[int] = []
        selected_set: set[int] = set()
        remaining = float(instance.budget_mb)
        objective = 0.0
        evals, pops = 0, 0

        for cidx in seeds:
            if cidx in selected_set or storage[cidx] > remaining + EPS:
                continue
            ridx = cover.get(cidx, np.empty(0, dtype=np.int32))
            val = float(np.dot(
                weights[ridx],
                np.maximum(current[ridx] - decode[cidx], 0.0),
            )) if len(ridx) else 0.0
            if val <= self.min_gain:
                continue
            selected.append(cidx)
            selected_set.add(cidx)
            remaining -= storage[cidx]
            objective += val
            if len(ridx):
                current[ridx] = np.minimum(current[ridx], decode[cidx])

        # Greedy(m,k) completion: the first (up to) two structures are chosen
        # by limited configuration enumeration; the remaining ones are chosen
        # incrementally. An exact lazy heap is only an acceleration of the
        # completion greedy, not the WARM optimizer call.
        heap: list[tuple[float, float, int]] = []
        for cidx in pool["cand_idx"].astype(int):
            if cidx in selected_set:
                continue
            s = storage[cidx]
            g = singleton[cidx]
            if s <= 0 or s > remaining + EPS or g <= self.min_gain:
                continue
            heapq.heappush(heap, (-g / s, -g, cidx))

        while heap:
            _, _, cidx = heapq.heappop(heap)
            pops += 1
            s = storage[cidx]
            if s > remaining + EPS:
                continue
            ridx = cover.get(cidx, np.empty(0, dtype=np.int32))
            g = float(np.dot(
                weights[ridx], np.maximum(current[ridx] - decode[cidx], 0.0),
            )) if len(ridx) else 0.0
            evals += 1
            if g <= self.min_gain:
                continue
            new_key = (-g / s, -g, cidx)
            if not heap or new_key <= heap[0]:
                selected.append(cidx)
                selected_set.add(cidx)
                remaining -= s
                objective += g
                if len(ridx):
                    current[ridx] = np.minimum(current[ridx], decode[cidx])
            else:
                heapq.heappush(heap, new_key)

        plan = pool[pool["cand_idx"].isin(selected_set)].copy()
        if len(plan):
            order = {c: i for i, c in enumerate(selected)}
            plan["selected_order"] = plan["cand_idx"].map(order)
            plan.sort_values("selected_order", kind="stable", inplace=True)
            plan.reset_index(drop=True, inplace=True)

        predicted_used = float(plan[instance.storage_col].sum()) if len(plan) else 0.0
        metadata = {
            "baseline_family": "MVS Greedy(2,k)-inspired physical design",
            "algorithm": "screen+enumerate_m2+lazy_greedy_completion",
            "candidate_pruning": False,
            "candidate_screening": "query+per_video+global_predicted_cost",
            "enumeration_m": 2,
            "enumerated_configurations": pair_trials,
            "pair_shortlist_size": shortlist_count,
            "initial_config_count": len(seeds),
            "initial_config_predicted_gain": float(seed_gain),
            "actual_selected_count": len(selected),
            "selection_cost_basis": f"{instance.storage_col}+{instance.decode_col}",
            "best_singleton_safeguard": False,
            **screen_meta,
        }
        return PlanResult(
            method, plan, float(objective), predicted_used,
            time.perf_counter() - start,
            marginal_evaluations=evals,
            heap_pops=pops,
            metadata=metadata,
        )
