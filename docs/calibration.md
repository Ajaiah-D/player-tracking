# Homography calibration guide

Every new video needs a manual calibration before pitch coordinates,
possession and formation analysis will work. This is the only manual
step in the pipeline, and it is also the least reliable stage overall --
read the caveats at the bottom before trusting the output.

## What the calibration is

A JSON file (one per video, stored in `calibration/`) listing point
correspondences between pixel locations in **one reference frame** of
the video and known positions on a standard 105 m x 68 m pitch:

```json
{
  "video": "myclip.mp4",
  "frame_index": 600,
  "points": [
    {"image": [1170.9, 348.8], "pitch": "right_pen_top_corner"},
    {"image": [196.5, 401.0],  "pitch": [52.5, 0.0]}
  ]
}
```

`pitch` is either a named landmark (full list printed by
`calibrate.py`, defined in `soccervision/pitch.py`) or raw
`[x, y]` coordinates in meters. Origin is the left goal line at the far
touchline as seen from the main camera; x runs toward the right goal.

## Producing one

```
python calibrate.py data/videos/myclip.mp4 --frame 600
```

Click a landmark in the window, then name it in the terminal. Save with
`s`. The tool prints the reprojection error; aim for under 1 m.

Practical rules that came out of calibrating the sample clip:

- **Pick the most zoomed-out frame** of the clip as the reference frame
  (scrub through the video first). More visible line intersections
  beats everything else. The camera-motion estimate maps all other
  frames onto the reference frame, so it does not need to be frame 0.
- **Use 6+ points, spread out.** With 4 points the homography is exact
  on those points and unchecked everywhere else. Errors only become
  visible with redundant points.
- **Only click line intersections** (penalty box corners, halfway line
  x touchline, goal posts on the goal line, penalty spot). Do not click
  along a lone line "somewhere" -- there is no way to know its pitch
  coordinate.
- **Avoid points behind advertising boards.** The far touchline is
  usually hidden behind the LED boards; the grass/board boundary is
  NOT the touchline and can be off by 1-3 m at that depth. If a far
  touchline point is unavoidable, expect the error and put more trust
  in points painted on grass.
- The center circle's leftmost/rightmost points are usable but
  approximate: the extreme point of the ellipse in the image is not
  exactly the projection of the circle's extreme point on the pitch.
  Prefer the two intersections of the circle with the halfway line,
  which are exact.

## Known failure modes (be honest with yourself about these)

- **Camera cuts.** The calibration is tied to one continuous camera
  shot. A hard cut (replay, close-up, reverse angle) breaks the
  camera-motion chain that links frames to the reference frame, and
  every position after the cut is garbage. This pipeline is built for
  single-shot clips; there is no cut detection. Trim the video to one
  continuous shot first.
- **Chain drift.** Frame-to-frame homographies absorb pan and zoom,
  but each pair estimate is slightly off and the errors compound.
  Measured on the 30 s sample clip (which pans and zooms ~1.6x):
  landmarks carried across the full 600-frame chain land 38-227 px
  from their true positions, i.e. roughly 4-15 m depending on pitch
  region. Positions are accurate near the calibration frame and
  degrade with temporal distance from it. Mitigations: calibrate the
  frame in the middle of the clip rather than the first frame, keep
  clips short, or calibrate multiple keyframes for longer footage.
  Check the annotated video's minimap: if stationary players slide
  across the minimap, the chain has drifted.
- **Lens distortion.** Broadcast lenses bend straight lines: on the
  sample clip the near touchline is visibly curved by several pixels
  across the frame. A homography cannot represent this; it puts a
  floor of roughly 1-2 m on position accuracy far from the calibration
  points. No undistortion step is implemented.
- **Areas without nearby calibration points are extrapolated.** If all
  your points are on the right half of the pitch, positions on the left
  half can be off by many meters. The pipeline drops positions that
  land outside the pitch rectangle, but positions that are wrong yet
  still on the pitch cannot be detected automatically.
