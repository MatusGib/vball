# Phase 2b: overlays, ball model v2, court calibration, players — design

Date: 2026-10-03
Parent: `2026-10-02-vball-design.md`; follows phase 2a (`docs/results/phase2a-ball.md`).
Decisions (user): include ball overlay, court calibration, players + tracking; pose later. Check whether the external sources in `docs/references/external_data.md` can be used for training. Downloaded (with approval): `asigatchov-fast-volleyball-tracking-inference` (MIT) and `vballnet-dataset` into `data/external/`.

Two implementation plans, executed in order:
1. `2026-10-03-phase2b1-overlay-ball-v2.md`: ball overlay, external ball-model benchmark, fine-tune v2 with VballNet labels.
2. `2026-10-03-phase2b2-court-players.md`: court calibration with camera-bump compensation, player detection + tracking, overlays.

## Findings that shape the design

| Source | Finding | Use |
|---|---|---|
| `vballnet-dataset` | 89 labelled clips (+1 predictions-only, skipped), 29,342 hand-labelled frames, ball visible in 27,551; TrackNetV3 layout `{train,test}/<match>/video/*.mp4` + `csv/<stem>_ball.csv` (`Frame,Visibility,X,Y`); mostly 1080p 30 fps; indoor amateur fixed cameras (behind baseline / corner, higher than ours) + 3 beach clips; licence not stated (personal use) | Real ball labels for fine-tune v2 |
| `fast-volleyball-tracking-inference` | MIT; ONNX ball models (VballNetV4c, GridV3, GridV2b, FastV1; 9 grayscale frames at 288×512), CSV in source pixels with extra `Radius` column; RAVEL-VB-011/012 player+ball(+head) detector with track ids (OpenVINO, 9-frame clips at 640×360) writing `ravel-vb-predictions-v1` JSON (`predictions[]` with `frame_index`, `class_name`, `score`, `bbox_xyxy` in pixels, `track_id`); own uv project (Python ≥ 3.14) | Ball benchmark; player-detector candidate |
| `Court-Keypoint-Detection` | No weights, data or licence | Not used (manual calibration) |
| Our footage | Camera mostly static, but bumped in 5 of 11 sets (shifts 10–119 px at 1080p, e.g. Kent 1 +119 px mid-set, Brunel away 3 −45 px) | Calibration must follow camera moves |

## Part 1

### Ball overlay
`GET /api/matches/{id}/ball` returns the ball track (`x`, `y` arrays in work pixels, `null` when not detected, plus fps and size). The viewer draws on a `<canvas>` over the video's displayed content rectangle (respecting letterboxing), synced per frame with `requestVideoFrameCallback` (frame = round(mediaTime × fps)); a "Show ball" toggle draws the current position and an 8-frame fading trail. The canvas layer is reused by Part 2.

### External ball models
Run each VballNet ONNX model with its own script on the work videos of the held-out sets (Kent 2, Brunel away 3), score with `vball balleval --pred`. Comparison only; adoption of an external model would be a separate decision (it would need integration).

### Fine-tune v2
`vball finetune --vballnet <dir>` adds VballNet clips as **labelled** sources: Visibility 1 → positive (x, y scaled to 512×288), Visibility 0 → negative (hand-labelled "no ball", unlike pseudo-labels where unseen frames are ignored), frames without a row → ignored. 512×288 copies cached under `data/external/vballnet-dataset/_vball/`. Same training recipe as v1 (windows sampled in proportion to positives, now ~3,000 windows). Adoption rule unchanged from 2a: better held-out ball recall at no worse precision than v1 @ 0.3 (Kent 2 0.73 / 0.67, Brunel away 3 0.66 / 0.87), rally F1 within −0.02 after re-tuning.

## Part 2

### Court model
Coordinates in metres, x across the court 0–9 (left→right as seen from the camera), y along it 0–18 from the near (camera-side) baseline; net at y = 9. Landmarks: near/far left/right corners, near/far attack-line ends (y = 6, 12), centre-line ends (y = 9). Optional net-top points (antenna tops) for drawing the net.

### Calibration
In the viewer: pause, click ≥ 4 visible landmarks (choosing each from a list, skipping hidden ones), save. Server fits the floor homography (`cv2.findHomography`, least squares), reports per-point reprojection error, and detects camera moves: ORB features on the 512×288 copy every 2 s vs the reference frame, RANSAC image-to-image homographies, segments split where the background displacement changes by > 3 px (1080p). Stored as `court.json` with the reference homography and per-segment `ref_to_frame` and precomputed `court_to_image` matrices. Overlay draws court lines for the current segment.

### Players
Two backends producing the same `players.csv` (`frame,track_id,x1,y1,x2,y2,score`, work pixels):
- **YOLO11** (Ultralytics, COCO person, AGPL — fine for personal use) with ByteTrack, every frame.
- **RAVEL-VB** via its own script/env, converted from its JSON.

Court enrichment at read time (calibration can change): foot point = bottom-centre → court (x, y), side = near if y < 9, on-court if x ∈ [−1.5, 10.5] and y ∈ [−4, 22] (includes the serve zone).

Evaluation without labelling (`vball playereval`): over rally frames, share with 5–7 on-court players on each side; mean number of distinct track ids per side per rally (ideal 6); runtime. Plus visual check: boxes with ids and a top-down mini court map in the viewer. The better backend becomes the default and runs on all sets.

## Success criteria

| Item | Criterion |
|---|---|
| Ball overlay | Marker sits on the ball when paused on clicked test frames (visual); toggle works while playing |
| Ball v2 | Adoption rule above (else keep v1 and record why) |
| Calibration | Reprojection error ≤ 10 px at 1080p on clicked points; after camera bumps (Kent 1, Brunel away 3) the drawn lines stay on the court (visual) |
| Players | ≥ 80 % of rally frames have 5–7 on-court players per side; ≤ 10 distinct track ids per side per rally on average; ≤ 40 min per 30 fps set on the RTX 3050 |

## Risks

| Risk | Mitigation |
|---|---|
| External code needs Python 3.14 + OpenVINO | Run in its own uv environment as a subprocess, never imported |
| RAVEL-VB on CPU too slow | Timed on a 2-minute excerpt first; YOLO11 on the GPU is the fallback |
| ORB matches on moving players | RANSAC; background (walls, lights) dominates; segments need ≥ 2 consistent samples |
| Near-camera landmarks out of frame (low camera) | Any 4 of 10 landmarks suffice; far corners, attack lines and centre line are usually visible |
| VballNet "no ball" labels may include occluded balls | Intended: the tracker should not fire where the ball cannot be seen |
