# Algorithms and Command-Line Reference

## WARM

WARM chooses candidate replicas from a video repository under a shared additional-storage budget. A query uses the least costly compatible serving option, including its source video as a fallback. The objective is the workload-weighted reduction in expected minimum serving time.

WARM uses dynamic marginal benefit per unit storage to rank candidates against the currently selected set. The optimizer includes a best-singleton comparison, safe dominance pruning, and lazy marginal evaluation.

The main implementation is in `src/warm/optimization.py`. Candidate-query serving relationships are implemented in `src/warm/coupling.py`; the planning entry point is in `src/warm/study.py`.

## Available Methods

| Name | Description |
|---|---|
| `WARM` | Global dynamic marginal-density selection |
| `VSS` | Adapted representation-caching comparison |
| `VStore` | Adapted video-format coalescing comparison |
| `VSA` | Adapted video-version allocation comparison |
| `VACTS` | Adapted popularity/storage–computation comparison |
| `Source-only` | Reference without selected replicas |
| `wm-pop` | Candidate ranking by popularity |
| `wm-smin` | Candidate ranking by storage size |
| `wm-ben` | Candidate ranking by standalone benefit |
| `wm-dens` | Fixed initial benefit-per-storage ranking |
| `wm-eqbd` | Equal per-video storage-budget allocation |
| `wm-pwbd` | Popularity-weighted per-video budget allocation |
| `wm-eagr` | Eager marginal-benefit recomputation |
| `wm-unpr` | Selection without dominance pruning |
| `wm-rbst` | Conservative candidate-cost adjustment |

The four prior-work comparisons are adaptations to the common WARM candidate and workload interface, not the original systems' software releases.

## Command-Line Interface

Show available methods:

```bash
warm methods
```

Check input data:

```bash
warm check-data --data-dir data --config config/paper_full.yaml
```

Run a selection plan:

```bash
warm plan --data-dir data --config config/experiment.yaml --methods WARM --budget-ratio 0.10 --output-dir outputs/planning
```

To compare methods, provide a comma-separated list, for example:

```bash
warm plan --data-dir data --config config/experiment.yaml --methods WARM,VSS,VStore,VSA,VACTS --budget-ratio 0.10 --output-dir outputs/planning
```

The `--candidate-source` option accepts `replica-json` (default) or `enumerated`. With `replica-json`, the planner uses candidate configurations from the per-video JSON files. With `enumerated`, it constructs candidate configurations from the values in the YAML configuration.

A request log can be supplied using `--historical-log path/to/queries.json`. The log must be a JSON array of request records containing `VID`, `req_w`, `req_h`, and `req_f`. If no log is supplied, workload weights follow the selected configuration.

## Planning Results

Each run writes to a directory identified by candidate source, storage-budget percentage, and seed. Its contents are:

```text
planning_summary.json
selected_replicas/
    WARM.json
    VSS.json
    VStore.json
    VSA.json
    VACTS.json
```

The selected-replica files are created for the requested methods; `Source-only` appears only in the summary.

`planning_summary.json` reports the method name, number of selected candidates, storage budget and estimated usage, estimated workload-weighted serving-time saving, and planning statistics.

Each selected-replica file lists `VID`, `CID`, replica configuration, selection order, and estimated storage/serving costs. These are **selection plans**, not encoded or persisted video files. Estimated planning costs should not be interpreted as measured end-to-end service latency or speedup.
