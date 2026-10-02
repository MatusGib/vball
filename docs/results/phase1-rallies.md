# Phase 1 results: rally detection

Match: Imperial M2 vs Kent, set 1 (23.3 min, 1920x1080, 30 fps; phone behind the baseline at roughly standing height).
Ground truth: 42 rallies labelled in the web app (serve toss → ball dead), 5.7 min of play in total.

| Run | Params changed | Precision | Recall | F1 | Start err | End err | Dead time removed |
|---|---|---|---|---|---|---|---|
| baseline | defaults from the plan | 0.88 | 0.67 | 0.76 | 0.75 s | 1.77 s | 82 % |
| tuned | `min_active_frac` 0.3→0.2, `max_gap_s` 2→3, `min_rally_s` 2→1, `pre_pad_s` 1→0.5, `post_pad_s` 1→1.5 | **0.97** | **0.90** | **0.94** | 0.66 s | 0.80 s | 77 % |

Tuning: 2,304-combination grid search (6 s, rules only, cached ball track). The chosen values sit in the middle of a broad plateau (15 combinations within 0.02 F1 of the best; `max_gap_s` 2–5 all score well), not on a single best point. **Tuned and evaluated on the same set — must be validated on another set.**

Phase 1 target (P ≥ 0.85, R ≥ 0.90): met on this set.

## Validation: Kent set 2 (params unchanged)

25.7 min, 40 labelled rallies (labels started from detections, then false ones deleted and missed ones added).

| Set | Precision | Recall | F1 |
|---|---|---|---|
| Kent set 1 (tuned on) | 0.97 | 0.90 | 0.94 |
| Kent set 2 (held out) | 0.86 | 0.93 | 0.89 |

Target still met on the held-out set. 6 false detections (no overlapping label, consistent with between-rally ball returns), 3 missed short rallies (2.7–3.4 s). Boundary errors on set 2 are not meaningful: 36 of 40 labels kept the detected edges unchanged, which is why the web app now has an explicit approve step.

## Ball tracker

Ball visible in 49 % of labelled rally frames, but in only 1.2 % of dead-time frames. The tracker rarely fires outside play, which is why visibility-based rules work despite the low hit rate inside rallies. Losses are mostly the far court, high balls against the ceiling and occlusion by near players.

## Processing time (RTX 3050 Laptop 4 GB, libx264 because NVENC needs driver ≥ 610)

23.3-min set: 13.2 min end to end (ingest ~5 min, downscale ~1 min, TrackNet ~7 min). Rallies-only export: 5 s (stream copy).

## Remaining errors

- Missed (4): short rallies of 1.3–3.4 s (aces / serve errors / quick points) where the tracker saw the ball in under 20 % of frames.
- False (1): 2:56.6–3:01.4, the ball being thrown back across the court to the server after a point.
- Earlier false positives were all long rallies split in two where the ball vanished > 2 s; `max_gap_s` 3 fixed these.

## Decision for phase 2

1. Validate these params on a second labelled set (Kent set 2) before trusting them.
2. Ball tracker fine-tune on own footage is the biggest lever (49 % rally-frame visibility caps recall on short rallies and will limit touch detection). Use model-assisted labelling of a few hundred frames.
3. Court-region mask from court calibration to drop between-rally ball returns.
4. Portrait videos (Royal Holloway) need letterboxing before TrackNet instead of squashing to 16:9.
