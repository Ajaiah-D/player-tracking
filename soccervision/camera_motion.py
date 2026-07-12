"""Camera motion estimation from tracked background features.

The reference implementation (soccer_tracker) models camera motion as
pure translation via Lucas-Kanade optical flow on border features. That
breaks down on real broadcast footage: even the 30 s sample clip zooms
out noticeably mid-clip, and under zoom a translation model is simply
wrong.

Here, features are detected in the frame border regions (stands,
hoardings -- static scenery) and tracked frame-to-frame with LK optical
flow; each consecutive frame pair then gets a full homography estimated
with RANSAC, and the pair homographies are composed so that any pixel in
frame t can be mapped back into frame 0 coordinates. The pitch
calibration homography only ever needs to exist for frame 0.

Limitations, honestly:
- errors compound over time (each pair homography is slightly off, and
  the composition drifts). Over ~30 s it stays usable; over minutes it
  will not. The pipeline prints a drift check so this is measurable.
- fails across hard camera cuts (features don't match at all).
  RANSAC inlier counts drop sharply at a cut; a warning is emitted, but
  positions after an unhandled cut are garbage until recalibration.
"""

import pickle
from pathlib import Path

import cv2
import numpy as np


class CameraMotionEstimator:
    def __init__(self, border_top=0.20, border_bottom=0.12, border_sides=0.06,
                 min_inliers=15):
        self.border_top = border_top
        self.border_bottom = border_bottom
        self.border_sides = border_sides
        self.min_inliers = min_inliers
        self.feature_params = dict(maxCorners=400, qualityLevel=0.01, minDistance=8, blockSize=7)
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )

    def _border_mask(self, shape):
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[: int(h * self.border_top), :] = 1
        mask[h - int(h * self.border_bottom) :, :] = 1
        mask[:, : int(w * self.border_sides)] = 1
        mask[:, w - int(w * self.border_sides) :] = 1
        return mask

    def estimate(self, frames, cache_path=None):
        """Returns a list of 3x3 matrices, one per frame: H[t] maps pixel
        coordinates in frame t to pixel coordinates in frame 0."""
        if cache_path is not None and Path(cache_path).exists():
            with open(cache_path, "rb") as f:
                print(f"loaded cached camera motion from {cache_path}")
                return pickle.load(f)

        to_frame0 = [np.eye(3)]
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        mask = self._border_mask(prev_gray.shape)
        low_confidence_frames = []

        for t in range(1, len(frames)):
            gray = cv2.cvtColor(frames[t], cv2.COLOR_BGR2GRAY)
            pair = np.eye(3)

            pts = cv2.goodFeaturesToTrack(prev_gray, mask=mask, **self.feature_params)
            if pts is not None and len(pts) >= 8:
                new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray, gray, pts, None, **self.lk_params
                )
                good = status.ravel() == 1
                if good.sum() >= 8:
                    src = new_pts[good].reshape(-1, 2)
                    dst = pts[good].reshape(-1, 2)  # frame t -> frame t-1
                    H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
                    if H is not None and inliers.sum() >= self.min_inliers:
                        pair = H
                    else:
                        low_confidence_frames.append(t)
                else:
                    low_confidence_frames.append(t)
            else:
                low_confidence_frames.append(t)

            to_frame0.append(to_frame0[-1] @ pair)
            prev_gray = gray

        if low_confidence_frames:
            print(f"WARNING: camera motion unreliable on {len(low_confidence_frames)} "
                  f"frames (first few: {low_confidence_frames[:5]}) -- possible camera "
                  "cut or too few background features; positions there inherit the "
                  "last good estimate")

        if cache_path is not None:
            with open(cache_path, "wb") as f:
                pickle.dump(to_frame0, f)
        return to_frame0

    @staticmethod
    def transform_point(point, matrix):
        p = np.array([point[0], point[1], 1.0])
        q = matrix @ p
        return (q[0] / q[2], q[1] / q[2])

    @classmethod
    def apply_to_tracks(cls, tracks, to_frame0):
        """Add 'position_stabilized': the position mapped into frame 0's
        pixel coordinates."""
        for per_frame in tracks.values():
            for frame_idx, frame_tracks in enumerate(per_frame):
                m = to_frame0[frame_idx]
                for info in frame_tracks.values():
                    info["position_stabilized"] = cls.transform_point(info["position"], m)

    @staticmethod
    def zoom_factor(to_frame0):
        """Approximate scale of each frame relative to frame 0 (>1 means
        the camera has zoomed in relative to frame 0). Useful diagnostics."""
        return np.array([np.sqrt(abs(np.linalg.det(m[:2, :2]))) for m in to_frame0])
