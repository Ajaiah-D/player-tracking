"""Pitch model and image-to-pitch homography.

Coordinate system: top-down view of a standard 105m x 68m pitch.
Origin is the top-left corner (left goal line, "top" touchline as seen
from the main broadcast camera). x runs along the length of the pitch
(0 = left goal line, 105 = right goal line), y across the width.
"""

import json

import cv2
import numpy as np

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

# Penalty area: 16.5m deep, 40.32m wide. Goal area: 5.5m deep, 18.32m wide.
_PEN_D = 16.5
_PEN_W = 40.32
_GOAL_D = 5.5
_GOAL_W = 18.32
_CIRCLE_R = 9.15

# Named landmarks a calibration file can reference. These are the points
# that are actually visible as line intersections in broadcast footage.
LANDMARKS = {
    "corner_left_top": (0.0, 0.0),
    "corner_left_bottom": (0.0, PITCH_WIDTH),
    "corner_right_top": (PITCH_LENGTH, 0.0),
    "corner_right_bottom": (PITCH_LENGTH, PITCH_WIDTH),
    "halfway_top": (PITCH_LENGTH / 2, 0.0),
    "halfway_bottom": (PITCH_LENGTH / 2, PITCH_WIDTH),
    "center_spot": (PITCH_LENGTH / 2, PITCH_WIDTH / 2),
    "center_circle_top": (PITCH_LENGTH / 2, PITCH_WIDTH / 2 - _CIRCLE_R),
    "center_circle_bottom": (PITCH_LENGTH / 2, PITCH_WIDTH / 2 + _CIRCLE_R),
    "center_circle_left": (PITCH_LENGTH / 2 - _CIRCLE_R, PITCH_WIDTH / 2),
    "center_circle_right": (PITCH_LENGTH / 2 + _CIRCLE_R, PITCH_WIDTH / 2),
    # left penalty area (defending the x=0 goal)
    "left_pen_top_corner": (_PEN_D, (PITCH_WIDTH - _PEN_W) / 2),
    "left_pen_bottom_corner": (_PEN_D, (PITCH_WIDTH + _PEN_W) / 2),
    "left_pen_top_goal_line": (0.0, (PITCH_WIDTH - _PEN_W) / 2),
    "left_pen_bottom_goal_line": (0.0, (PITCH_WIDTH + _PEN_W) / 2),
    "left_goal_area_top_corner": (_GOAL_D, (PITCH_WIDTH - _GOAL_W) / 2),
    "left_goal_area_bottom_corner": (_GOAL_D, (PITCH_WIDTH + _GOAL_W) / 2),
    "left_penalty_spot": (11.0, PITCH_WIDTH / 2),
    # right penalty area
    "right_pen_top_corner": (PITCH_LENGTH - _PEN_D, (PITCH_WIDTH - _PEN_W) / 2),
    "right_pen_bottom_corner": (PITCH_LENGTH - _PEN_D, (PITCH_WIDTH + _PEN_W) / 2),
    "right_pen_top_goal_line": (PITCH_LENGTH, (PITCH_WIDTH - _PEN_W) / 2),
    "right_pen_bottom_goal_line": (PITCH_LENGTH, (PITCH_WIDTH + _PEN_W) / 2),
    "right_goal_area_top_corner": (PITCH_LENGTH - _GOAL_D, (PITCH_WIDTH - _GOAL_W) / 2),
    "right_goal_area_bottom_corner": (PITCH_LENGTH - _GOAL_D, (PITCH_WIDTH + _GOAL_W) / 2),
    "right_penalty_spot": (PITCH_LENGTH - 11.0, PITCH_WIDTH / 2),
}


class PitchHomography:
    """Maps image pixel coordinates to pitch coordinates (meters).

    Built from a calibration file of point correspondences that must be
    produced manually per video (see calibrate.py). The homography is
    estimated for one reference frame; positions from other frames must
    first be shifted back into that frame's coordinates using the camera
    motion estimate. This only holds up while the camera translation
    model is a reasonable fit -- see README for the caveats.
    """

    def __init__(self, image_points, pitch_points, frame_index=0):
        image_points = np.asarray(image_points, dtype=np.float32)
        pitch_points = np.asarray(pitch_points, dtype=np.float32)
        if len(image_points) < 4:
            raise ValueError("homography needs at least 4 point pairs")
        self.frame_index = frame_index
        self.image_points = image_points
        self.pitch_points = pitch_points
        if len(image_points) == 4:
            self.matrix = cv2.getPerspectiveTransform(image_points, pitch_points)
        else:
            self.matrix, _ = cv2.findHomography(image_points, pitch_points, cv2.RANSAC, 3.0)
        if self.matrix is None:
            raise ValueError("homography estimation failed")

    @classmethod
    def from_file(cls, path):
        with open(path) as f:
            calib = json.load(f)
        pitch_points = []
        for ref in calib["points"]:
            target = ref["pitch"]
            if isinstance(target, str):
                target = LANDMARKS[target]
            pitch_points.append(target)
        image_points = [ref["image"] for ref in calib["points"]]
        return cls(image_points, pitch_points, calib.get("frame_index", 0))

    def to_pitch(self, points):
        """Transform an (N, 2) array of pixel coords to pitch meters."""
        points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(points, self.matrix)
        return out.reshape(-1, 2)

    def reprojection_error(self):
        """Mean error (in meters) of the calibration points themselves."""
        projected = self.to_pitch(self.image_points)
        return float(np.mean(np.linalg.norm(projected - self.pitch_points, axis=1)))


def inside_pitch(points, margin=3.0):
    """Boolean mask for pitch coords that land on (or near) the pitch."""
    points = np.asarray(points)
    return (
        (points[:, 0] >= -margin)
        & (points[:, 0] <= PITCH_LENGTH + margin)
        & (points[:, 1] >= -margin)
        & (points[:, 1] <= PITCH_WIDTH + margin)
    )
