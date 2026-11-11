# WARM: Workload-Aware Replica Materialization for Video Query Serving

WARM studies workload-aware physical design for video repositories. Given source-video characteristics, a query workload, and a shared storage budget, it determines which video replicas should be selected for advance materialization to reduce expected query-serving time.

<img src=".\warm_pipeline.png" style="zoom:50%;" />

The repository provides an implementation of WARM's replica-selection algorithm, supporting components for workload and candidate modeling, and comparison methods evaluated within a common planning interface. It is intended for researchers interested in workload-aware storage design, video representation management, and budget-constrained selection algorithms.

## 1. Problem and Approach

A video repository may receive requests with different resolution and frame-rate requirements. Each request can be served by its original source video or by a compatible replica, and these alternatives may have different serving costs. Generating more replicas can improve future access, but consumes additional storage. Moreover, the benefit of any one replica depends on the other replicas already selected: two replicas may serve overlapping sets of requests, so their individual benefits cannot simply be added together.

WARM models these interactions using the minimum serving cost available to each query. For a query $q$ and a selected replica set $A$, let $t_0(q)$ denote its source-only serving cost and $t(q,c)$ the cost of serving it using candidate replica \(c\). An incompatible replica is treated as having infinite serving cost. The resulting serving cost is

$$
T(q,A)=\min\left\{t_0(q),\min_{c\in A}t(q,c)\right\}.
$$

Given query probabilities $p(q)$, WARM maximizes the expected serving-time saving

$$
F(A)=\sum_{q\in\mathcal Q}p(q)\bigl[t_0(q)-T(q,A)\bigr]
$$

subject to a repository-wide storage constraint \(\sum_{c\in A}s(c)\leq B\). Here, $s(c)$ is the storage required by a candidate and $B$ is the additional-storage budget shared by all videos. Source videos remain available as fallback serving options.

This objective is monotone submodular: as more replicas are selected, an additional compatible replica cannot provide greater marginal saving than it would have provided before those alternatives became available. WARM uses this structure to perform global selection based on current **marginal saving per unit storage**, rather than a fixed ranking determined independently for each video. Its planning algorithm incorporates a best-singleton comparison, safe dominance pruning, and lazy marginal evaluation to improve selection quality and reduce redundant computations. The accompanying paper provides the formal analysis.

## 2. Repository Layout

```text
WARM/
├── src/
│   └── warm/                    # Planner and supporting algorithms
├── config/
│   └── experiment.yaml          # Planning and workload configuration
├── data/
│   ├── README.md                # Data organization and access
│   ├── prediciton/
│   │   ├── train.json
│   │   └── test.json
│   └── experiment/
│       ├── source_videos.json
│       └── replicas/
│           └── replica_V_E_XXXXXXXXXX.json
├── docs/
│   ├── DATA_SCHEMA.md           # JSON fields and identifier conventions
│   └── METHODS.md               # Algorithm and CLI reference
├── pyproject.toml
└── README.md
```

The Python implementation is located in `src/warm/`. The repository's command-line interface provides data validation and planning commands without requiring users to invoke individual Python modules. Configuration values, including supported candidate configurations and workload settings, are stored in `config/paper_full.yaml`.

The data and methods documentation is available separately so that field definitions, algorithm descriptions, and command examples can be consulted without searching through source code.

## 3. Statistical Data

WARM operates on JSON statistical records describing source videos, candidate replicas, and physical-cost characteristics. The input directory is organized as follows:

| Input                                         | Description                                                  |
| --------------------------------------------- | ------------------------------------------------------------ |
| `data/prediciton/train.json`                  | Statistical records used by the replica-cost estimator       |
| `data/prediciton/test.json`                   | Held-out records supported by the cost-model data interface  |
| `data/experiment/source_videos.json`          | Source-video attributes and associated storage and serving costs |
| `data/experiment/replicas/replica_V_E_*.json` | Candidate replica configurations and statistics grouped by source video |

The experiment configuration is designed for a collection of **1,500 source videos**. Each source video has a string identifier (`VID`), while each candidate replica is identified by a `CID` associated with its owning `VID`. The replica files provide the candidate configurations from which the planner can construct a selection space.

Public JSON field names are descriptive. For example, `height_source`, `frame_rate_source`, and `storage_cost_source` describe source-video properties; `height_replica`, `gop_length_replica`, and `encoding_protocol_replica` describe candidate configurations. The fields `serving_speedup_value` and `storage_overhead_radio` represent measured ratios in the statistical records. See [Data Schema](docs/DATA_SCHEMA.md) for definitions, units, and identifier requirements, and [Data](data/README.md) for examples of reading the files.

Cost estimates used for planning are obtained through the cost-model interface. A candidate's selection is evaluated using these estimated costs and its contribution to the current query workload. Replica-catalog statistics are not treated as advance knowledge of a candidate's realized physical cost.

## 4. Installation

WARM requires **Python 3.10 or later**. Install the package from the repository root:

```bash
python -m pip install -e .
```

The package declares its Python dependencies in `pyproject.toml`, including NumPy, pandas, PyYAML, PyTorch, and scikit-learn. After installation, the `warm` command can be used from a terminal in the project environment.

The examples below assume that the `data/` directory follows the structure shown above and that the configuration file is available at `config/paper_full.yaml`.

## 5. Running the Planner

List the supported algorithms:

```bash
warm methods
```

Check the expected JSON input files and their basic consistency:

```bash
warm check-data --data-dir data --config config/experiment.yaml
```

Generate a WARM replica-selection plan with an additional-storage budget equal to **10% of the total source-video storage**:

```bash
warm plan --data-dir data --config config/experiment.yaml --methods WARM --budget-ratio 0.10 --output-dir outputs/planning
```

The `--budget-ratio` option defines the additional space available for selected replicas relative to the total source-video storage. The same setting can be used to study how selected replica sets change under different budget constraints.

To compare several supported algorithms under the same planning inputs, pass their names as a comma-separated list:

```bash
warm plan --data-dir data --config config/experiment.yaml --methods WARM,VSS,VStore,VSA,VACTS,Source-only --budget-ratio 0.10 --output-dir outputs/planning
```

By default, the planner obtains candidate configurations from the per-video replica JSON catalog. Alternatively, it can enumerate candidate configurations using the values in the YAML configuration:

```bash
warm plan --data-dir data --config config/experiment.yaml --methods WARM --budget-ratio 0.10 --candidate-source enumerated --output-dir outputs/planning
```

Workload weights may follow the configured workload distributions or be derived from a compatible request log provided through `--historical-log`. A JSON request log uses `VID`, `req_w`, `req_h`, and `req_f` to identify the requested video, width, height, and frame rate. The request specifications must correspond to entries in the supported request catalog.

## 6. Supported Methods

The public method names follow those used in the WARM paper:

| Method          | Description                                                  |
| --------------- | ------------------------------------------------------------ |
| **WARM**        | Workload-aware global dynamic marginal-density selection     |
| **VSS**         | Adapted representation-caching method                        |
| **VStore**      | Adapted video-format coalescing method                       |
| **VSA**         | Adapted video-version allocation method                      |
| **VACTS**       | Adapted popularity and storage–computation trade-off method  |
| **Source-only** | Reference serving option without additional selected replicas |

The interface also provides the paper's algorithm variants: `wm-pop` (popularity ranking), `wm-smin` (minimum-storage ranking), `wm-ben` (standalone-benefit ranking), `wm-dens` (fixed initial marginal density), `wm-eqbd` (equal local budgets), `wm-pwbd` (popularity-weighted local budgets), `wm-eagr` (eager marginal recomputation), `wm-unpr` (without dominance pruning), and `wm-rbst` (conservative cost adjustment).

All methods use the same public input and output conventions. The four prior-work methods are adaptations for a common candidate space and storage constraint; they should not be confused with distributions of the original systems. Additional algorithm details are available in [Methods](docs/METHODS.md).

## 7. Planning Outputs

Each planning command writes a directory identified by candidate source, storage budget, and random seed. For example, a comparison at the 10% budget using the replica JSON catalog and seed 42 produces:

```text
outputs/planning/
└── replica-json_budget_10pct_seed_42/
    ├── planning_summary.json
    └── selected_replicas/
        ├── WARM.json
        ├── VSS.json
        ├── VStore.json
        ├── VSA.json
        └── VACTS.json
```

`planning_summary.json` is a JSON array with one entry per requested method. It reports the method name, number of candidate specifications, number of selected replicas, configured budget, estimated storage use, estimated workload-weighted saving, planner runtime, and marginal-evaluation count.

Each `selected_replicas/<method>.json` file contains the chosen replica identifiers and configurations, selection order, and estimated storage and serving costs. Only requested methods receive selection files. `Source-only` appears in the planning summary and has no selected-replica file.

The results describe **selection decisions**, not video replicas that have already been generated or persisted. Estimated workload saving is an optimization metric and should not be interpreted as a measured end-to-end query latency or service speedup.

## 8. Availability and Intended Use

The repository is intended for understanding, evaluating, and extending workload-aware video replica materialization algorithms. Its code and statistical interfaces support investigations of replica valuation, overlapping request coverage, shared-budget allocation, and planning efficiency.

Raw video bitstreams, encoded replica files, and a complete video-serving runtime are not included. Accordingly, the planning interface does not directly reproduce the paper's measured end-to-end serving-performance results. Researchers can use the provided algorithms and JSON interfaces to inspect replica-selection behavior or integrate the planner into other video data management environments.

For further reference, see [Data](data/README.md), [Data Schema](docs/DATA_SCHEMA.md), and [Methods](docs/METHODS.md).
