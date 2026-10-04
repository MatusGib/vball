# Phase 2d — 3D ball flights from one camera (interim)

Date: 2026-10-04. Spec: `docs/superpowers/specs/2026-10-04-ball3d-design.md`. Status: **interim**. Net-tape clicks added on Kent 1, Kent 2 and Brunel away 3 (see below).

## Camera fits (`vball camera <id>`, floor points only)

| Match | Focal (px) | Camera position (x, y, z m) | Floor error mean / max (px) | Off-screen clicks left out |
|---|---|---|---|---|
| Kent 1 | 1655 | 4.63, −6.78, 1.58 | 5.2 / 15.1 | near right corner |
| Brunel away 3 | 1433 | 4.67, −3.13, 0.51 | 6.8 / 18.8 | both near corners |

Near corners clicked outside the picture are guesses; including them moved the Kent 1 camera by ~1 m and raised
the error to 13 px mean / 28 px max, so the camera fit now ignores clicks outside the frame (they still count in
the floor homography used for players — not changed). Brunel away 3 is filmed from ~0.5 m up: its far half-court
is ~25 px tall.

**With the net clicks** (2026-10-04, later): the net tops reproject within 2.8 / 1.7 px (Kent 1), 8.0 / 4.6 px
(Kent 2) and 15.8 / 15.3 px (Brunel away 3) of the clicks; Kent 1's camera moved by under 1 cm and 3 px of focal
length. Re-running `vball ball3d` changed serve medians by 1–4 km/h and the plausible shares by a few points
(Kent 1 64%, Kent 2 69%, Brunel away 3 60% in 40–100 km/h). The camera is therefore not what limits real serves.

## Simulation (`vball ball3dsim <id> --count 100`)

Ideal camera (the fitted one generates and fits the data), real dropout patterns, Gaussian jitter σ per axis.
Errors are median / 90th percentile; "fit" is the share of flights with ≥ 8 visible frames that were fitted.

Kent 1 (30 fps):

| Kind | σ px | Fit | Speed (km/h) | Apex (m) | Net crossing (m) | Landing (m) |
|---|---|---|---|---|---|---|
| serve | 3 | 96% | 1.2 / 8.4 | 0.02 / 0.08 | 0.03 / 0.10 | 0.10 / 0.37 |
| serve | 6 | 94% | 2.0 / 12.6 | 0.04 / 0.14 | 0.05 / 0.17 | 0.22 / 0.59 |
| serve | 9 | 96% | 4.2 / 26.1 | 0.06 / 0.27 | 0.08 / 0.33 | 0.34 / 1.26 |
| attack | 6 | 73% | 12.7 / 57.3 | 0.26 / 1.35 | 0.42 / 1.75 | 2.2 / 12.9 |
| set | 6 | 88% | 0.3 / 2.9 | 0.04 / 0.37 | — | 0.42 / 1.17 |

Brunel away 3 (60 fps, very low camera): serves 2.0 / 13.1 km/h, net 0.05 / 0.87 m, landing 0.18 / 2.8 m at
σ 6; attacks 13.5 / 48.6 km/h, landing 1.3 / 10.4 m; sets as Kent.

Camera sensitivity (Kent 1, serves, σ 6): focal ±5% → speed 4.3 / 11.4 km/h, net 0.14 / 0.29 m, landing
0.5 / 1.2 m; pitch +0.5° → 1.8 / 11.2 km/h; camera height +0.2 m → net 0.20 / 0.26 m.

## Real rallies (`vball ball3d <id>`)

85% of flights fit on all three sets. Serve = first flight in the rally that crosses the net plane (the first
flight is often the toss):

| Match | Serves found | Speed p10 / median / p90 (km/h) | In 40–100 km/h | Net crossing median | Above 2.43 m | Landing within 2 m of court |
|---|---|---|---|---|---|---|
| Kent 1 | 36 / 42 | 32 / 63 / 125 | 64% | 3.06 m | 67% | 90% |
| Kent 2 | 35 / 40 | 41 / 68 / 108 | 71% | 2.96 m | 77% | 88% |
| Brunel away 3 | 40 / 40 | 34 / 50 / 83 | 62% | 3.20 m | 78% | 100% |

Reprojected arcs overlay the tracked ball closely for most serves (contact sheets), so failures are depth, not
fit: several far-side serves pin the start at the fit's limit (6 m behind the far baseline) with 125 km/h and
1.4 m net crossings; some "serves" are far-court passes whose wrong depth makes them cross the net plane; one is a
0.3 s fragment.

## Reading so far

- **Serves (speed, net height, landing):** promising — accurate in simulation and plausible in ~2/3 of real serves.
  The real-data gap is larger than camera error alone explains (small camera errors cost ~2–3 km/h), so
  segmentation (toss + hit merged, fragments) and depth ambiguity on serves aimed at the camera are next.
- **Sets:** accurate in simulation (speed < 1 km/h, apex 4 cm).
- **Attacks:** not with one camera as built — short, steep flights along the line of sight (12 km/h median,
  2 m landing error even in simulation).

## Next

1. Reject fits that end on a bound; split the toss from the serve.
2. If serves stay plausible: a landing check page (click where serves land) for real ground truth.
