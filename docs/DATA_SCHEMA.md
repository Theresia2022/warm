# JSON Data Schema

WARM reads UTF-8 JSON arrays of objects from `data/`. Public field names are used directly in the code. Each file type has the required fields listed below.

## Identifiers

| Field | Format | Meaning |
|---|---|---|
| `VID` in prediction data | `V_T_` followed by 8–10 Base32 characters | Identifies a video in the cost-model records |
| `VID` in experiment data | `V_E_` followed by 8–10 Base32 characters | Identifies a source video in the repository |
| `CID` | `C_E_<video-token>_<replica-token>` | Identifies a candidate replica |

The video token in a replica `CID` matches its source video's `VID` after removing `V_E_`. Identifiers are strings; preserve their contents when reading or writing JSON.

## Source-Video Fields

These fields appear in `data/experiment/source_videos.json`:

| Field | Meaning |
|---|---|
| `VID` | Source-video identifier |
| `height_source` | Source height (pixels) |
| `weight_source` | Source width (pixels) |
| `frame_rate_source` | Source frame rate (frames/s) |
| `gop_length_source` | Source GOP parameter |
| `bit_rate_key_frame_source` | Key-frame bitrate descriptor |
| `bit_rate_non-key_frame_source` | Non-key-frame bitrate descriptor |
| `storage_cost_source` | Source storage cost (KB) |
| `serving_cost_source` | Source serving cost (microseconds) |
| `frame_nums_source` | Number of source frames |
| `encoding_protocol_source` | Source encoding protocol |

The two bitrate descriptors are numeric; their measurement units should be interpreted according to the associated dataset rather than assumed from the field names.

## Replica Fields

Every nonempty `data/experiment/replicas/replica_<VID>.json` contains `VID` and the source-video descriptors listed above, together with:

| Field | Meaning |
|---|---|
| `CID` | Replica identifier |
| `height_replica` | Replica height (pixels) |
| `weight_replica` | Replica width (pixels) |
| `frame_rate_replica` | Replica frame rate (frames/s) |
| `gop_length_replica` | Replica GOP parameter |
| `encoding_protocol_replica` | Replica encoding protocol |
| `encoding_params` | Encoding parameter or preset |
| `serving_speedup_value` | Ratio of source serving cost to replica serving cost |
| `storage_overhead_radio` | Ratio of replica storage cost to source storage cost |

The spelling of `weight_source`, `weight_replica`, and `storage_overhead_radio` is part of the JSON interface.

The ratios have the following meanings:

\[
\mathrm{serving\_speedup\_value}
=\frac{\mathrm{serving\_cost\_source}}{\mathrm{serving\_cost\_replica}},
\qquad
\mathrm{storage\_overhead\_radio}
=\frac{\mathrm{storage\_cost\_replica}}{\mathrm{storage\_cost\_source}}.
\]

The replica catalog provides configuration records and associated statistics. The `warm plan` command obtains planning-time candidate costs from the cost estimator rather than treating catalog measurements as known future costs.

## Prediction Records

`data/prediciton/train.json` and `data/prediciton/test.json` use the same schema:

**Video identifier and source features**

`VID`, `height_source`, `weight_source`, `frame_rate_source`, `gop_length_source`, `bit_rate_key_frame_source`, `bit_rate_non-key_frame_source`, `storage_cost_source`, `encoding_protocol_source`

**Replica configuration**

`height_replica`, `weight_replica`, `frame_rate_replica`, `gop_length_replica`, `encoding_protocol_replica`, `encoding_params`

**Target values and measured costs**

`serving_speedup_value`, `storage_overhead_radio`, `storage_cost_replica`, `serving_cost_replica`

`storage_cost_replica` is measured in KB; `serving_cost_replica` is measured in microseconds. The predictor uses video and replica configuration features, not measured target/cost fields, as model inputs.

## File and Consistency Requirements

- Each input file is a top-level JSON array; each record is an object.
- `VID` and `CID` use the string formats above; source IDs and replica IDs must be unique where appropriate.
- Each replica record's `VID` must match the `VID` encoded in its filename and `CID`.
- Numerical dimensions, storage costs, and serving costs must contain valid numbers; measurements used as positive costs or ratios must be positive.
- With the supplied configuration, `warm check-data` expects 1,500 source-video records and one replica JSON file for each source video.

Run `warm check-data --data-dir data --config config/paper_full.yaml` to check the supported schema and data consistency.
