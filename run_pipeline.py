"""End-to-end analysis of a soccer clip.

    python run_pipeline.py data/videos/clip.mp4 --calibration calibration/clip.json

Stages: YOLOv8 detection -> ByteTrack tracking -> team assignment ->
camera motion compensation -> pitch homography -> possession phases ->
formation classification. Detection and camera motion results are cached
under output/tracking/<clip>/ so re-runs after tweaking downstream
parameters are fast; pass --no-cache to force recompute.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from soccervision.annotate import render
from soccervision.camera_motion import CameraMotionEstimator
from soccervision.detection import Detector
from soccervision.formation import analyze_phases
from soccervision.pitch import PitchHomography, inside_pitch
from soccervision.possession import possession_by_frame, possession_segments
from soccervision.teams import TeamClassifier
from soccervision.tracking import PlayerTracker, detection_stats
from soccervision.video_io import read_video, save_video


def add_pitch_positions(tracks, homography, to_frame0):
    """Pixel position -> calibration reference frame pixels -> pitch
    meters. Positions that land well outside the pitch rectangle are
    treated as invalid (usually a sign the homography doesn't extend to
    that part of the frame, or that camera-motion compensation has
    drifted)."""
    # calibration may reference any frame, not just frame 0
    ref = homography.frame_index
    if ref >= len(to_frame0):
        raise SystemExit(
            f"calibration references frame {ref} but only {len(to_frame0)} frames "
            "were processed; raise --max-frames or recalibrate on an earlier frame")
    to_ref = [np.linalg.inv(to_frame0[ref]) @ m for m in to_frame0]
    dropped = total = 0
    for per_frame in tracks.values():
        for frame_idx, frame_tracks in enumerate(per_frame):
            m = to_ref[frame_idx]
            for info in frame_tracks.values():
                stabilized = CameraMotionEstimator.transform_point(info["position"], m)
                pitch = homography.to_pitch([stabilized])[0]
                total += 1
                if inside_pitch(pitch.reshape(1, 2))[0]:
                    info["position_pitch"] = pitch.tolist()
                else:
                    info["position_pitch"] = None
                    dropped += 1
    return dropped, total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--model", type=Path, default=Path("models/yolov8m-640-football-players.pt"))
    parser.add_argument("--calibration", type=Path, default=None,
                        help="homography calibration JSON (see calibrate.py); "
                             "without it, pitch coordinates and formations are skipped")
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-video", action="store_true", help="skip writing the annotated video")
    args = parser.parse_args()

    stem = args.video.stem
    track_dir = args.output_dir / "tracking" / stem
    formation_dir = args.output_dir / "formations"
    track_dir.mkdir(parents=True, exist_ok=True)
    formation_dir.mkdir(parents=True, exist_ok=True)

    print(f"reading {args.video}")
    frames, fps = read_video(args.video, max_frames=args.max_frames)
    print(f"{len(frames)} frames @ {fps:.1f} fps")

    cache = None if args.no_cache else track_dir / "tracks.pkl"
    detector = Detector(args.model, conf=args.conf)
    tracker = PlayerTracker(detector)
    print("running detection + tracking (slow on CPU)")
    tracks = tracker.track(frames, cache_path=cache, progress=True)

    stats = detection_stats(tracks)
    print("tracking stats:", json.dumps(stats, indent=2))

    print("assigning teams from jersey colors")
    classifier = TeamClassifier()
    assignment = classifier.fit_and_assign(frames, tracks["players"])
    classifier.apply(tracks["players"], assignment)

    print("estimating camera motion")
    motion = CameraMotionEstimator()
    motion_cache = None if args.no_cache else track_dir / "camera_motion.pkl"
    to_frame0 = motion.estimate(frames, cache_path=motion_cache)
    zoom = motion.zoom_factor(to_frame0)
    print(f"camera zoom relative to frame 0: min {zoom.min():.2f}x max {zoom.max():.2f}x "
          f"(1.0 = no zoom; far from 1.0 means the static-calibration shortcut is doing real work)")

    report = None
    if args.calibration is not None:
        homography = PitchHomography.from_file(args.calibration)
        err = homography.reprojection_error()
        print(f"homography reprojection error on calibration points: {err:.2f} m")
        if err > 2.0:
            print("WARNING: calibration error above 2 m -- pitch coordinates will be poor")

        dropped, total = add_pitch_positions(tracks, homography, to_frame0)
        print(f"pitch mapping: {dropped}/{total} positions fell outside the pitch and were dropped")

        possession, _ = possession_by_frame(tracks, smooth_window=int(fps))
        segments = possession_segments(possession, min_frames=int(fps))
        counts = {t: int((possession == t).sum()) for t in (0, 1, 2)}
        print(f"possession frames: team1={counts[1]} team2={counts[2]} unknown={counts[0]}")
        print(f"{len(segments)} possession segments of at least 1 s")

        report = analyze_phases(tracks, segments)
        for entry in report:
            seg = entry["segment"]
            label = entry["formation"] or f"unclassified ({entry.get('reason')})"
            print(f"  frames {seg['start']}-{seg['end']} team {entry['team']} "
                  f"{entry['phase']}: {label}"
                  + (f" cost={entry['match_cost']}" if entry.get("formation") else ""))

        out_path = formation_dir / f"{stem}.json"
        with open(out_path, "w") as f:
            json.dump({"video": str(args.video), "stats": stats,
                       "segments": segments, "phases": report}, f, indent=2)
        print(f"formation report -> {out_path}")
    else:
        possession = None
        print("no calibration file given: skipping pitch coordinates, "
              "possession and formation analysis")

    if not args.no_video:
        print("rendering annotated video")
        annotated = render(frames, tracks, possession_team=possession)
        video_out = track_dir / f"{stem}_annotated.mp4"
        save_video(annotated, video_out, fps)
        print(f"annotated video -> {video_out}")


if __name__ == "__main__":
    main()
