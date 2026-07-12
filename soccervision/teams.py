"""Team assignment from jersey color.

Approach follows the reference implementations: for each player crop,
k-means (k=2) on the pixels of the top half of the box separates jersey
from grass; the cluster not touching the crop corners is the jersey.
Player jersey colors are then clustered into two teams.

Differences from the references: colors are sampled across multiple
frames per track (not just frame 0) and team assignment is a majority
vote per track id, which is more robust to bad single-frame crops.
Goalkeepers are excluded from the team-color fit since their kit
matches neither outfield team.
"""

import numpy as np
from sklearn.cluster import KMeans


def jersey_color(frame, bbox):
    x1, y1, x2, y2 = (int(v) for v in bbox)
    x1, y1 = max(x1, 0), max(y1, 0)
    crop = frame[y1:y2, x1:x2]
    top = crop[: max(crop.shape[0] // 2, 1)]
    if top.shape[0] < 2 or top.shape[1] < 2:
        return None

    pixels = top.reshape(-1, 3).astype(np.float64)
    km = KMeans(n_clusters=2, n_init=1, random_state=0).fit(pixels)
    labels = km.labels_.reshape(top.shape[0], top.shape[1])

    corners = [labels[0, 0], labels[0, -1], labels[-1, 0], labels[-1, -1]]
    background = max(set(corners), key=corners.count)
    return km.cluster_centers_[1 - background]


class TeamClassifier:
    def __init__(self, sample_stride=30, max_samples_per_track=8):
        self.sample_stride = sample_stride
        self.max_samples = max_samples_per_track
        self.kmeans = None
        self.team_colors = {}

    def fit_and_assign(self, frames, player_tracks):
        """Returns {track_id: team} with team in {1, 2}. Goalkeepers get
        team 0 (unassigned) -- they are not used for formation analysis."""
        samples = {}  # track_id -> list of colors
        gk_ids = set()
        for frame_idx in range(0, len(frames), self.sample_stride):
            for tid, info in player_tracks[frame_idx].items():
                if info.get("cls") == "goalkeeper":
                    gk_ids.add(tid)
                    continue
                if len(samples.get(tid, [])) >= self.max_samples:
                    continue
                color = jersey_color(frames[frame_idx], info["bbox"])
                if color is not None:
                    samples.setdefault(tid, []).append(color)

        all_colors = [c for colors in samples.values() for c in colors]
        if len(all_colors) < 4:
            raise ValueError("not enough player crops to fit team colors")

        self.kmeans = KMeans(n_clusters=2, n_init=10, random_state=0).fit(all_colors)
        self.team_colors = {
            1: self.kmeans.cluster_centers_[0],
            2: self.kmeans.cluster_centers_[1],
        }

        # distance of each track's mean color to its nearest team color,
        # to catch kits that belong to neither team (referees that the
        # detector misclassified as players, staff, etc.)
        mean_colors = {tid: np.mean(colors, axis=0) for tid, colors in samples.items()}
        dists = {
            tid: min(np.linalg.norm(c - self.team_colors[1]),
                     np.linalg.norm(c - self.team_colors[2]))
            for tid, c in mean_colors.items()
        }
        cutoff = max(3.0 * np.median(list(dists.values())), 60.0)

        assignment = {}
        for tid, colors in samples.items():
            if dists[tid] > cutoff:
                assignment[tid] = 0  # neither team
                continue
            votes = self.kmeans.predict(np.array(colors)) + 1
            assignment[tid] = int(np.bincount(votes).argmax())
        for tid in gk_ids:
            assignment.setdefault(tid, 0)
        return assignment

    def apply(self, player_tracks, assignment):
        """Write team labels back onto the track dicts. Track ids never
        seen during sampling (short-lived tracks) stay unassigned."""
        for frame_tracks in player_tracks:
            for tid, info in frame_tracks.items():
                team = assignment.get(tid, 0)
                info["team"] = team
                if team in self.team_colors:
                    info["team_color"] = self.team_colors[team].tolist()
