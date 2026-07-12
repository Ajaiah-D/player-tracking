"""Multi-object tracking with ByteTrack (via the supervision library).

Track structure used throughout the pipeline:

    tracks = {
        "players": [ {track_id: {"bbox": [...], "cls": "player"|"goalkeeper", ...}}, ... one dict per frame ],
        "referees": [ {track_id: {"bbox": [...]}}, ... ],
        "ball":     [ {1: {"bbox": [...]}} or {}, ... ],
    }

Goalkeepers are tracked together with players (they behave the same for
tracking purposes) but keep their class label so downstream stages can
exclude them from team clustering and formation matching.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import supervision as sv


def _foot_position(bbox):
    return ((bbox[0] + bbox[2]) / 2, bbox[3])


class PlayerTracker:
    def __init__(self, detector):
        self.detector = detector
        self.byte_track = sv.ByteTrack()

    def track(self, frames, cache_path=None, progress=False):
        if cache_path is not None and Path(cache_path).exists():
            with open(cache_path, "rb") as f:
                print(f"loaded cached tracks from {cache_path}")
                return pickle.load(f)

        detections = self.detector.detect(frames, progress=progress)
        names = self.detector.class_names
        ids = {v: k for k, v in names.items()}

        tracks = {"players": [], "referees": [], "ball": []}

        for result in detections:
            det = sv.Detections.from_ultralytics(result)

            # remember which detections were goalkeepers, then fold them
            # into the player class so ByteTrack treats them uniformly
            gk_mask = det.class_id == ids.get("goalkeeper", -1)
            det.class_id[gk_mask] = ids["player"]

            tracked = self.byte_track.update_with_detections(det)

            players, referees, ball = {}, {}, {}
            for bbox, _, conf, cls_id, track_id, _ in tracked:
                bbox = bbox.tolist()
                if cls_id == ids["player"]:
                    players[int(track_id)] = {"bbox": bbox, "cls": "player"}
                elif cls_id == ids.get("referee"):
                    referees[int(track_id)] = {"bbox": bbox}

            # mark goalkeepers by IoU against the pre-track gk detections
            for gk_bbox in det.xyxy[gk_mask]:
                best_id, best_iou = None, 0.3
                for tid, info in players.items():
                    iou = _iou(gk_bbox, info["bbox"])
                    if iou > best_iou:
                        best_id, best_iou = tid, iou
                if best_id is not None:
                    players[best_id]["cls"] = "goalkeeper"

            # single ball: keep the highest-confidence detection, untracked
            ball_mask = det.class_id == ids.get("ball", -1)
            if ball_mask.any():
                idx = np.argmax(det.confidence[ball_mask])
                ball[1] = {"bbox": det.xyxy[ball_mask][idx].tolist()}

            tracks["players"].append(players)
            tracks["referees"].append(referees)
            tracks["ball"].append(ball)

        add_positions(tracks)
        tracks["ball"] = interpolate_ball(tracks["ball"])

        if cache_path is not None:
            with open(cache_path, "wb") as f:
                pickle.dump(tracks, f)
        return tracks


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def add_positions(tracks):
    """Anchor point per object: foot point for people, center for the ball."""
    for kind, per_frame in tracks.items():
        for frame_tracks in per_frame:
            for info in frame_tracks.values():
                b = info["bbox"]
                if kind == "ball":
                    info["position"] = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
                else:
                    info["position"] = _foot_position(b)


def interpolate_ball(ball_frames):
    """Fill gaps in ball detection by linear interpolation between hits.

    Frames that were interpolated are flagged so possession logic can
    weight them accordingly.
    """
    boxes = [f.get(1, {}).get("bbox", [np.nan] * 4) for f in ball_frames]
    df = pd.DataFrame(boxes, columns=["x1", "y1", "x2", "y2"])
    detected = ~df["x1"].isna()
    df = df.interpolate().bfill().ffill()

    out = []
    for i, row in enumerate(df.to_numpy()):
        if np.isnan(row).any():
            out.append({})
            continue
        b = row.tolist()
        out.append({1: {
            "bbox": b,
            "position": ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2),
            "interpolated": not bool(detected.iloc[i]),
        }})
    return out


def detection_stats(tracks):
    """Simple per-run numbers for sanity checking, printed by the CLI."""
    n = len(tracks["players"])
    players_per_frame = [len(f) for f in tracks["players"]]
    ball_detected = sum(
        1 for f in tracks["ball"] if f and not f[1].get("interpolated", False)
    )
    unique_ids = set()
    for f in tracks["players"]:
        unique_ids.update(f.keys())
    return {
        "frames": n,
        "players_per_frame_mean": float(np.mean(players_per_frame)) if n else 0,
        "players_per_frame_min": int(np.min(players_per_frame)) if n else 0,
        "players_per_frame_max": int(np.max(players_per_frame)) if n else 0,
        "unique_player_track_ids": len(unique_ids),
        "ball_detected_frames": ball_detected,
        "ball_detection_rate": ball_detected / n if n else 0,
    }
