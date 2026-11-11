"""Small synthetic JSON fixtures for tests and demos only; NOT paper data."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .identifiers import source_id, predictor_video_id, replica_id

def _record(d: dict, keys) -> dict:
    return {key: d[key] for key in keys}


def _dump(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def generate_synthetic_dataset(out_dir: str | Path, videos: int = 30, seed: int = 7) -> Path:
    """Generate fake source, replica, train and test JSON under out_dir/data/.

    Outputs match the same public schema and deterministic ID construction as
    captions/method.py.  They are for algorithm invariants, NOT measured videos.
    """
    root = Path(out_dir)
    data = root / "data"
    pred_dir, experiment_dir = data / "prediciton", data / "experiment"
    replicas_dir = experiment_dir / "replicas"
    rng = np.random.default_rng(seed)
    source_specs = [(1920, 1080, 60.0), (1280, 720, 30.0), (3840, 2160, 60.0)]
    codecs = ["h264", "hevc", "wmv"]
    sources, train, test = [], [], []
    source_keys = ("VID", "height_source", "weight_source", "frame_rate_source", "gop_length_source", "bit_rate_key_frame_source", "bit_rate_non-key_frame_source", "encoding_protocol_source", "storage_cost_source", "serving_cost_source", "frame_nums_source")
    replica_keys = source_keys + ("CID", "height_replica", "weight_replica", "frame_rate_replica", "gop_length_replica", "encoding_protocol_replica", "encoding_params", "serving_speedup_value", "storage_overhead_radio")
    pred_keys = ("VID", "height_source", "weight_source", "frame_rate_source", "gop_length_source", "bit_rate_key_frame_source", "bit_rate_non-key_frame_source", "storage_cost_source",
                 "encoding_protocol_source", "height_replica", "weight_replica", "frame_rate_replica", "gop_length_replica", "encoding_protocol_replica", "encoding_params",
                 "serving_speedup_value", "storage_overhead_radio", "storage_cost_replica", "serving_cost_replica")

    for i in range(videos):
        video_id = source_id(i + 1)
        w, h, fps = source_specs[i % len(source_specs)]
        motion = float(rng.uniform(0.1, 1.0))
        texture = float(rng.uniform(0.1, 1.0))
        duration = float(rng.uniform(5.0, 10.0))
        proto = codecs[i % 2]
        s1 = float((w * h * fps * duration / 1e5) * (0.7 + 0.8 * texture))
        t1 = float((w * h * duration / 800.0) * (0.7 + motion))
        src = {
            "VID": video_id, "height_source": h, "weight_source": w,
            "frame_rate_source": fps, "gop_length_source": 1.0,
            "bit_rate_key_frame_source": motion,
            "bit_rate_non-key_frame_source": texture,
            "encoding_protocol_source": proto,
            "frame_nums_source": int(round(duration * fps)),
            "storage_cost_source": s1, "serving_cost_source": t1,
        }
        sources.append(_record(src, source_keys))
        per_video = []
        widths = sorted({x for x in (360, 540, 720, 1280, 1920, 3840, w) if x <= w})
        frs = sorted({x for x in (15.0, 20.0, 30.0, 45.0, 60.0, fps) if x <= fps})
        for w2 in widths:
            h2 = max(4, int((w2 / w) * h / 4) * 4)
            for f2 in frs:
                for g2 in (0.25, 0.5, 1.0):
                    for proto2 in codecs:
                        for encoding_params in (1, 3, 5):
                            pixel_ratio = (w2 * h2) / (w * h)
                            fps_ratio = f2 / fps
                            codec_factor = {"h264": 1.0, "hevc": 0.75, "wmv": 1.25}[proto2]
                            preset_factor = {1: 0.92, 3: 1.0, 5: 1.08}[encoding_params]
                            quality_factor = 0.55 + 0.45 * g2
                            storage_ratio = max(0.02, pixel_ratio * fps_ratio * quality_factor
                                      * codec_factor * (0.85 + 0.2 * texture) * preset_factor)
                            decode_ratio = max(0.03, (pixel_ratio ** 0.72)
                                * (fps_ratio ** 0.80) * (0.75 + 0.45 * motion)
                                * {"h264": 1.0, "hevc": 1.12, "wmv": 0.95}[proto2]
                                * (1.04 - 0.025 * encoding_params))
                            serving_speedup = (1.0 / decode_ratio) * float(np.exp(rng.normal(0, 0.035)))
                            storage_ratio *= float(np.exp(rng.normal(0, 0.035)))
                            candidate_id = replica_id(video_id, h2, w2, int(f2),
                                             int(g2) if float(g2).is_integer() else g2,
                                             proto2, encoding_params)
                            row = dict(src, CID=candidate_id, height_replica=h2, weight_replica=w2, frame_rate_replica=f2,
                                       gop_length_replica=g2, encoding_protocol_replica=proto2, encoding_params=encoding_params,
                                       serving_speedup_value=serving_speedup, storage_overhead_radio=storage_ratio, storage_cost_replica=s1*storage_ratio, serving_cost_replica=t1/serving_speedup)
                            per_video.append(_record(row, replica_keys))
                            # Independent train/test namespace, no leakage of
                            # experimental-video V_E identifiers to prediction rows.
                            part = "train" if rng.random() < 0.75 else "test"
                            pred = dict(row)
                            pred["VID"] = predictor_video_id(part, i)
                            (train if part == "train" else test).append(_record(pred, pred_keys))
        _dump(replicas_dir / f"replica_{video_id}.json", per_video)
    _dump(experiment_dir / "source_videos.json", sources)
    _dump(pred_dir / "train.json", train)
    _dump(pred_dir / "test.json", test)
    return data
