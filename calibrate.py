"""Manual homography calibration for a video.

This is the one step of the pipeline that cannot be automated here and
must be redone for every new video (and after every hard camera cut):

    python calibrate.py data/videos/clip.mp4 --frame 0

A window shows the chosen frame. For each visible pitch landmark:
  1. left-click its exact location in the image
  2. the terminal lists the known landmark names -- type the matching
     one (or raw pitch coordinates as "x,y" in meters)
Keys in the window: u = undo last point, s = save and quit, q = quit
without saving.

Use at least 4 points, ideally 6+ spread across the visible area (line
intersections work best: penalty area corners, the halfway line ends,
center circle tangent points). Points clustered in one small region
give a homography that extrapolates badly to the rest of the frame.

The result is saved to calibration/<video-name>.json and passed to
run_pipeline.py via --calibration.
"""

import argparse
import json
from pathlib import Path

import cv2

from soccervision.pitch import LANDMARKS, PitchHomography

clicks = []


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicks.append((x, y))


def ask_landmark():
    names = sorted(LANDMARKS)
    print("landmarks:")
    for i, name in enumerate(names):
        print(f"  {i:2d}  {name}  {LANDMARKS[name]}")
    while True:
        raw = input("landmark name/number (or 'x,y' meters, empty to discard click): ").strip()
        if not raw:
            return None
        if "," in raw:
            try:
                x, y = (float(v) for v in raw.split(","))
                return [x, y]
            except ValueError:
                continue
        if raw.isdigit() and int(raw) < len(names):
            return names[int(raw)]
        if raw in LANDMARKS:
            return raw
        print("not recognized, try again")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--frame", type=int, default=0,
                        help="frame to calibrate on; pick the one where the most pitch lines "
                             "are visible (wide/zoomed-out moment). The pipeline maps other "
                             "frames onto it via the camera motion estimate.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"could not read frame {args.frame} from {args.video}")

    points = []
    window = "calibrate (click landmark, then name it in the terminal)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        display = frame.copy()
        for p in points:
            x, y = (int(v) for v in p["image"])
            cv2.drawMarker(display, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            label = p["pitch"] if isinstance(p["pitch"], str) else str(p["pitch"])
            cv2.putText(display, label, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow(window, display)
        key = cv2.waitKey(30) & 0xFF

        if clicks:
            click = clicks.pop(0)
            print(f"\nclicked image point {click}")
            target = ask_landmark()
            if target is not None:
                points.append({"image": list(click), "pitch": target})
        if key == ord("u") and points:
            points.pop()
        elif key == ord("q"):
            return
        elif key == ord("s"):
            break
    cv2.destroyAllWindows()

    if len(points) < 4:
        raise SystemExit("need at least 4 points, nothing saved")

    calib = {"video": args.video.name, "frame_index": args.frame, "points": points}
    out = args.out or Path("calibration") / f"{args.video.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(calib, f, indent=2)

    homography = PitchHomography.from_file(out)
    print(f"saved {len(points)} points -> {out}")
    print(f"reprojection error: {homography.reprojection_error():.2f} m "
          "(aim for < 1 m; if it's higher, re-click the worst points)")


if __name__ == "__main__":
    main()
