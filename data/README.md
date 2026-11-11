# Data

The `data/` directory contains JSON statistical records used by WARM. These records describe source videos, candidate replica configurations and costs, and data for cost estimation. All input files are UTF-8 JSON arrays of objects.

## Files

| Location | Contents |
|---|---|
| `prediciton/train.json` | Video and replica statistics used to fit the candidate-cost estimator |
| `prediciton/test.json` | Held-out records for the cost-model data interface |
| `experiment/source_videos.json` | Source-video attributes, storage costs, and serving costs |
| `experiment/replicas/replica_V_E_*.json` | Candidate replica records grouped by their owning source video |

The `experiment/` collection is configured for 1,500 source videos. Each source video is identified by `VID`; each replica has a `CID` and references the corresponding source `VID`. One replica JSON file is associated with each source-video identifier. Files may contain empty arrays when no candidate records are present.

Refer to [Data Schema](../docs/DATA_SCHEMA.md) for required fields, units, and identifier formats.

## Accessing the Data

Read the source-video and corresponding replica records with Python's standard JSON library:

```python
import json
from pathlib import Path

data_dir = Path("data/experiment")
with (data_dir / "source_videos.json").open(encoding="utf-8") as f:
    videos = json.load(f)

video_id = videos[0]["VID"]
replica_path = data_dir / "replicas" / f"replica_{video_id}.json"
with replica_path.open(encoding="utf-8") as f:
    replicas = json.load(f)

print(video_id, len(replicas))
```

The `VID` and `CID` fields are string identifiers and should not be converted to numeric indices. To check file consistency and required fields from the project root, run:

```bash
warm check-data --data-dir data --config config/experiment.yaml
```

For replica planning, the command-line interface can use catalog configurations directly or enumerate configurations specified in `config/paper_full.yaml`. See [Methods](../docs/METHODS.md) for the relevant commands.

The records provide statistical information and candidate specifications; they are not video bitstreams.
