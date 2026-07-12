"""Rendering of tracking results onto video frames, plus a top-down
minimap that makes homography problems visible at a glance: if the dots
drift off the pitch or players slide around while standing still, the
calibration or camera-motion compensation is off."""

import cv2
import numpy as np

from .pitch import PITCH_LENGTH, PITCH_WIDTH

REFEREE_COLOR = (0, 255, 255)
BALL_COLOR = (0, 255, 0)


def display_color(color):
    """Team colors from k-means are muddied by shadow and motion blur
    (a neon kit averages out to pale green). Saturate for drawing so the
    two teams are actually distinguishable on the green pitch."""
    pixel = np.uint8([[list(color)]])
    h, s, v = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0, 0]
    if s < 40:  # near-achromatic kit: draw as white
        return (255, 255, 255)
    vivid = cv2.cvtColor(np.uint8([[[h, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(c) for c in vivid)


def _ellipse(frame, bbox, color, track_id=None):
    x_center = int((bbox[0] + bbox[2]) / 2)
    y2 = int(bbox[3])
    width = int(bbox[2] - bbox[0])
    cv2.ellipse(
        frame,
        center=(x_center, y2),
        axes=(width, int(0.35 * width)),
        angle=0,
        startAngle=-45,
        endAngle=235,
        color=color,
        thickness=2,
        lineType=cv2.LINE_4,
    )
    if track_id is not None:
        cv2.putText(
            frame,
            str(track_id),
            (x_center - 10, y2 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )


def _ball_marker(frame, bbox):
    x = int((bbox[0] + bbox[2]) / 2)
    y = int(bbox[1])
    pts = np.array([[x, y], [x - 8, y - 16], [x + 8, y - 16]])
    cv2.drawContours(frame, [pts], 0, BALL_COLOR, cv2.FILLED)
    cv2.drawContours(frame, [pts], 0, (0, 0, 0), 2)


def _draw_minimap_base(width=300):
    """Pitch outline scaled to `width` pixels wide."""
    scale = width / PITCH_LENGTH
    h = int(PITCH_WIDTH * scale)
    img = np.full((h, width, 3), (60, 130, 60), dtype=np.uint8)
    white = (255, 255, 255)
    cv2.rectangle(img, (0, 0), (width - 1, h - 1), white, 1)
    cv2.line(img, (width // 2, 0), (width // 2, h - 1), white, 1)
    cv2.circle(img, (width // 2, h // 2), int(9.15 * scale), white, 1)
    for x0 in (0, PITCH_LENGTH - 16.5):
        p1 = (int(x0 * scale), int((PITCH_WIDTH - 40.32) / 2 * scale))
        p2 = (int((x0 + 16.5) * scale), int((PITCH_WIDTH + 40.32) / 2 * scale))
        cv2.rectangle(img, p1, p2, white, 1)
    return img, scale


def render(frames, tracks, possession_team=None, minimap=True):
    minimap_base, scale = _draw_minimap_base()
    out = []
    for i, frame in enumerate(frames):
        frame = frame.copy()

        for tid, info in tracks["players"][i].items():
            if "team_color" in info:
                color = display_color(info["team_color"])
            else:
                color = (100, 100, 100)  # unassigned (e.g. misdetected referee)
            _ellipse(frame, info["bbox"], color, tid)
        for info in tracks["referees"][i].values():
            _ellipse(frame, info["bbox"], REFEREE_COLOR)
        ball = tracks["ball"][i].get(1)
        if ball is not None:
            _ball_marker(frame, ball["bbox"])

        if possession_team is not None:
            label = {0: "possession: -", 1: "possession: team 1", 2: "possession: team 2"}
            cv2.putText(frame, label[int(possession_team[i])], (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        if minimap:
            mm = minimap_base.copy()
            for info in tracks["players"][i].values():
                pos = info.get("position_pitch")
                if pos is None or "team_color" not in info:
                    continue
                color = display_color(info["team_color"])
                cv2.circle(mm, (int(pos[0] * scale), int(pos[1] * scale)), 4, color, -1)
                cv2.circle(mm, (int(pos[0] * scale), int(pos[1] * scale)), 4, (30, 30, 30), 1)
            if ball is not None and ball.get("position_pitch") is not None:
                bp = ball["position_pitch"]
                cv2.circle(mm, (int(bp[0] * scale), int(bp[1] * scale)), 3, BALL_COLOR, -1)
            h, w = mm.shape[:2]
            fh, fw = frame.shape[:2]
            y0, x0 = fh - h - 20, (fw - w) // 2
            roi = frame[y0 : y0 + h, x0 : x0 + w]
            frame[y0 : y0 + h, x0 : x0 + w] = cv2.addWeighted(mm, 0.8, roi, 0.2, 0)

        out.append(frame)
    return out
