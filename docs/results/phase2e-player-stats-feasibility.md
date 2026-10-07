# Phase 2e — Can we get Balltime-style stats? (feasibility, 2026-10-06)

Question from the owner: after exploring the unused sources, can vball produce per-player stats like Balltime
(hitting percentage, serve in/out, aces, receive rating, ...)? This page records what was measured, not plans.

## 3D flights: all 11 sets current

Every set has `flights.csv`. Brunel away 3 (match 11) was stale (court re-saved after the run) and was re-run:
469 flights, 395 fitted (84%), serve median 44 km/h, 55% in 40–100 km/h.

## What Balltime reports (academy.balltime.com, box score)

| Group | Stats | Formula |
|---|---|---|
| Attack | kills, errors, attempts, hitting % | (K − E) / TA |
| Serve | aces, errors, attempts, serving % | (TA − SE) / TA |
| Serve receive | 0–3 pass rating per reception, pass % | sum of ratings / TA |
| Setting | assists, set attempts | set to a teammate who kills |
| Defence | digs | playing an opponent's attack |

Every one of these needs four things: **contacts** (when the ball was touched), **the action** of each contact
(serve, pass, set, attack, block, dig), **who** touched it, and **the rally outcome** (who won the point and how).

## Unused sources, read (no downloads)

| Source | What it actually offers (from the repo) | Verdict for us |
|---|---|---|
| VolleyVision | YOLOv8m **action detector** (block, defence, serve, set, spike; mAP 92% on its own mostly front-view images), 52 MB `best.pt` on GitHub; also YOLOv8m players, YOLOv7-tiny ball. CC BY-NC-ND. | Worth a test: an action box gives the action *and* the player at once. Unknown how it does on our back-view near side. Needs a download. |
| ddecks beach-volley-vision | TrackNet + RF-DETR/ByteTrack players + court homography + rally state machine + serve detection by "depth-on-side + trajectory fusion"; score keeper unfinished; no per-player stats. MIT. | Confirms our approach; its serve-side idea is what was tested below. RF-DETR is the candidate player detector for low cameras. |
| passform | Contact = kink in the ball path; the toucher = the person whose pose skeleton overlaps the ball at that frame (YOLOv8-pose + MediaPipe). No accuracy figures. | The attribution rule to copy: our flight splits are the kinks; nearest player box (or pose) at the split frame. No download needed. |
| VNL-STES | 6,137 frame-accurate events (serve 17%, receive 25%, set 23%, spike 22%, block 9%, score 4%) in 1,028 VNL broadcast rallies; spotting 74–77% mAP within 1–4 frames. 13 GB; weights not published. | Broadcast VNL is filmed from behind a baseline, closest public view to ours. Too heavy to train on the laptop now; keep for touch spotting later. |
| VREN | One 136 KB CSV of rally notation: per touch locations, pass rating (in/out), set type, hit type, blockers, serve type, win/lose reason, winning team. | Useful priors for the touch grammar and how ratings are defined; not training data for vision. |
| WASB | Ball detection baselines across 5 sports including volleyball, MIT, weights on Google Drive. | Skip: ball tracking is not the bottleneck. |
| Ibrahim volleyball, MultiSports, TU Graz | Broadcast / side views, gated access. | Not useful for this camera. |
| openvolley ovscout2 | R app for scouting from a fixed camera into DataVolley files, court registration by corners. | The way to make **ground truth** for one set if we want to measure per-player stats; our own viewer could do a lighter version. |

## Experiment 1: which end served each rally (Kent 3, 45 labelled rallies)

The serving end is the cheapest route to rally outcomes: whoever serves rally *i + 1* won rally *i*. Reference:
36 rallies where the serving end was clear by eye on a contact sheet (ball path over the frame 0.3 s after the
rally start); the other 9 were ambiguous.

| Method | Agrees with eye |
|---|---|
| 3D fit (side of the first flight / direction of the first net-crossing flight) | 25 / 36 (69%) |
| Two 3D fits per opening flight, near-server vs far-server bounds, lower cost wins | 26 / 36 (72%) |
| Ball leaves a player standing behind a baseline (serve check), decided on 26 | 25 / 26 when decided |
| Players standing behind each baseline (count over −1 s … +1.5 s) | 32 / 36 (89%) |
| **Ball-leaves-player when decided, else the count** | **34 / 36 (94%)** |

## Experiment 2: does the serving end give legal set scores?

Points per end from the combined method (winner of rally *i* = server of rally *i + 1*; the last rally is
unknown). A set must end 25–x (x ≤ 23) or by two after 24–24.

| Set | Rallies | Near / far / unknown | Legal? |
|---|---|---|---|
| Kent 1 (labels) | 42 | 15 / 24 / 2 | yes (25–17 if the unknowns go far) |
| Kent 2 (labels) | 40 | 15 / 24 / 0 | yes (25–15) |
| Kent 3 (labels) | 45 | 19 / 24 / 1 | yes (25–20) |
| Kent 4 (detected) | 53 | 20 / 31 / 1 | no: 31 > 25, detected rallies include non-rallies |
| Brunel home 1–4 (detected) | 40–53 | e.g. 12 / 27 / 8 | no / too many unknowns |
| Brunel away 1, 3 (low camera) | 38, 41 | 5 / 32, 39 / 1 | no: method fails |
| Brunel away 2 | 45 | 27 / 17 / 0 | no |

Reading: on the Kent camera (about 1.6 m up) with approved rallies, rally winners look right. With detected
(unapproved) rallies, false rallies add points. With the 0.5 m Brunel away camera, the far baseline is too
compressed for the foot positions to tell who is behind it.

## Experiment 3: pin the serve's start behind the server's baseline

Refitting the first net-crossing flight with its start bounded to 1.5–4 m up behind the known serving
baseline: Kent 1 13 → 17 plausible serves, Kent 2 22 → 22, Kent 3 23 → 20; p90 speed still 106–130 km/h. **No
clear gain**: the remaining bad serves come from how flights are cut (toss and hit merged, fragments), not from
depth. Not adopted.

## Stat by stat

| Stat | Needs | Today | Verdict |
|---|---|---|---|
| Points won / lost, side-out %, run lengths (team) | serving end per rally + approved rallies | experiment 1–2 | **Feasible** on the Kent camera; needs a one-key correction in the viewer for the rest |
| Serve attempts (team) | serving end | yes | **Feasible** |
| Aces / serve errors (team) | serving end + outcome + whether the serve was touched | rally outcome yes; touch count rough | **Feasible approx.**: 1-flight rallies won by the server = ace, lost = error; needs checking |
| Serve in / out | where an *untouched* serve lands, ±0.3 m near lines | 3D landing 0.2–0.6 m error in simulation; only ~55% of real serves plausible | **Experimental**; only clear-in / clear-out, line calls no |
| Serve speed | 3D serve fit | yes (plausible half) | **Have** |
| Receive rating 0–3 | where the pass goes vs the setter's target | set flights accurate (sim 4 cm apex); pass flights unmeasured | **Possible** as a proxy (pass apex and distance from the target zone); needs ground truth |
| Attack attempts / kills / errors, hitting % (team) | attack contacts + outcome | 3D attacks poor (12 km/h, 2 m landing error in sim); outcome from experiment 2 | **Approx.** attempts from the touch sequence (third contact crossing the net); kill/error from who won |
| Assists, digs, blocks | action type of each contact | no touch classification yet | Needs touch detection + action labels (phase 2c) |
| **Any of the above per player** | who touched the ball | YOLO tracks with many id switches; no names | **Not yet.** Three routes, below |

## Routes to per-player stats

1. **Servers from the rotation (cheapest).** Teams serve in a fixed order and rotate on each side-out. With the
   lineup in serving order and the first server of each set, the serving end sequence names every server, so
   serve speed, aces, errors and in/out become per player. Breaks on substitutions (manual fix in the viewer).
2. **Jersey numbers on the near side.** On the Kent contact sheets the near players' back numbers (14, 25, 33,
   36, 41) are large and readable, so number recognition on the near-side team looks realistic (phase 3). The far
   side is too small.
3. **Contact attribution** (passform rule): at each flight split, the nearest player box to the ball. Gives
   per-track touches, but needs route 2 or a one-click name per track to turn tracks into people, and fewer id
   switches (SportsMOT / RF-DETR).

## Scripts

Experiment scripts were throwaway (scratchpad): `serve_side.py`, `hyp.py`, `server.py`, `constrained.py`,
`sheet.py` (rally-start contact sheet). The serving-end method is ~40 lines on top of `vball.serve`.

## Built: `vball serving` on all 11 sets (2026-10-06)

Design: `docs/superpowers/specs/2026-10-06-serving-score-design.md`. Serving end as in experiment 1; serve outcomes by
the owner's rule (ended right after the serve: the next server says ace or error; played: in); the VolleyVision action
detector looks for the receivers playing the serve. About 5 minutes per set on the RTX 3050 (action detector).

| Set | Rallies | Ends: contact / count / unknown | Score near–far | Legal? | Aces | Errors | To check | Receive seen |
|---|---|---|---|---|---|---|---|---|
| Kent set 1 (labels) | 42 | 20 / 20 / 2 | 15–25 | yes | 1 | 3 | 7 | 33 |
| Kent set 2 (labels) | 40 | 20 / 20 / 0 | 15–25 | yes | 0 | 0 | 4 | 29 |
| Kent set 3 (labels) | 45 | 30 / 14 / 1 | 19–25 | yes | 1 | 1 | 9 | 21 |
| Kent set 4 (detected) | 53 | 23 / 29 / 1 | 20–31 | no | 0 | 0 | 6 | 34 |
| Brunel home set 1 (detected) | 40 | 19 / 17 / 4 | 19–17 | no | 0 | 0 | 10 | 14 |
| Brunel home set 2 (detected) | 46 | 15 / 27 / 4 | 17–25 | yes | 0 | 0 | 10 | 17 |
| Brunel home set 3 (detected) | 48 | 18 / 22 / 8 | 12–27 | no | 0 | 0 | 16 | 17 |
| Brunel home set 4 (detected) | 53 | 16 / 31 / 6 | 17–29 | no | 0 | 0 | 15 | 20 |
| Brunel away set 1 (detected) | 38 | 9 / 29 / 0 | 5–32 | no | 0 | 0 | 4 | 0 |
| Brunel away set 2 (detected) | 45 | 12 / 33 / 0 | 27–17 | no | 0 | 0 | 10 | 2 |
| Brunel away set 3 (labels) | 41 | 4 / 37 / 0 | 39–1 | no | 3 | 0 | 3 | 3 |

"Legal" = the final score is a finished set (25 with the loser on 23 or less, or two clear past 25).

Reading:

- **Kent 1–3 (labelled):** legal scores (15–25, 15–25, 19–25), the camera height the method was built on.
- **Detected rallies:** false rallies add points (Kent 4 20–31, Brunel home 3 12–27). Approving labels fixes this.
- **Brunel away (0.5 m camera):** serving end fails (39–1); needs the V key or a better server finder.
- **Aces and errors are under-counted:** VREN says a 45-point set has about 7 serve errors and 2 aces; the automatic
  rule finds 0–3 per set. Most serve-ending rallies land in the 2.5–4.5 s band (*check*), and a receiver catching or
  bumping an out ball looks like a receive to the action detector. The viewer suggests ace or error for each one to
  check (from who served next), one O press each.
- **Receive seen** is frequent on the Kent camera (21–34 rallies per set) and rare on the low Brunel away camera (0–3).

## Using every source (2026-10-06, later)

The owner gave the real Kent scores (us = near end): 20–25, 22–25, 26–24, 22–25, and marked the outcome of 40 serves
on Kent 1. Downloads approved: model weights (VolleyVision players / ball / actions, RF-DETR Medium, YOLO11m-pose) and
the VNL-STES dataset (not fetched: 13 GB with 16 GB free). Code repos were not approved (WASB, T-DEED, STES code). Every
source's work is logged on the Sources page.

**Serve outcome thresholds from the owner's Kent 1 results.** Aces / errors ended 2.1–3.4 s after the serve, played
serves 4.2 s or later; the next server matched the owner's ace / error every time; the action detector "saw" a
receive on 2 of 7 serve errors (out balls caught), so it only hints. New thresholds 3.6 / 4.0 s: 38 / 40 right, 2 to
check, none wrong.

**Spectators past the floor's horizon.** On the 0.5 m Brunel away camera, feet on the far balcony mapped behind the
camera (y −18 to −124 m), i.e. behind the near baseline. Such feet now get no court position: Brunel away 3 serving
ends 16 / 30 → 25 / 30 (eye-checked), Kent 3 unchanged.

**Player detectors for the serving end** (rally start −1 s … +1.5 s, eye-checked rallies):

| Detector | Kent 3 (36) | Brunel away 3 (30) |
|---|---|---|
| RF-DETR Medium (now the default) | 35 | 29 |
| YOLO11s tracks (after the horizon fix) | 34 | 25 |
| YOLO11m-pose, server's wrist at the ball | 32 | 23 |
| VolleyVision YOLOv8m players | 19 | – |

**Ball detectors on the clicked frames** (precision / recall at 15 px): VolleyVision YOLOv7-tiny 0.94 / 0.85 (Kent 2),
0.88 / 0.43 (Brunel away 3); RF-DETR "sports ball" 0.93 / 0.62, 0.91 / 0.52; our tracker 0.67 / 0.73, 0.87 / 0.66.

**Touch attribution (passform's rule, YOLO11m-pose) on Kent 3:** 226 contacts (flight splits); a wrist within 0.3 body
heights of the ball at 42%; those give legal visits (≤ 3 touches per side) 97% of the time.

**Real scores vs labelled rallies:** Kent 1 has 42 labelled rallies for 45 points played, Kent 2 40 for 47, Kent 3 45
for 50: some rallies are missing from the labels (or the recording), so the inferred scores cannot be exact there.

**Low cameras and jump serves.** Drawing every set's calibration showed them right; the failures were in the reading:
benches and walls 20–90 m "behind" the far baseline counted as far servers (now: within 9 m of a baseline), jump
servers' feet in the air map to the far end on a 0.5 m camera (now: where they stood in the second before the contact),
and on Brunel away 1 the far receivers themselves map behind the far baseline (a foot 19 px above it is 93 m away). On
cameras under 1 m the ball decides first (ddecks' trajectory fusion): a near serve is first seen more than 200 px above
the net tape. Brunel away 3 28 / 30 eye-checked; Brunel away 1 4–33 → 20–17, matching all 14 serves judged by eye; on
the 1.5 m Kent camera the cue is wrong (19 / 36), so there the players decide.

## The remaining sources (2026-10-07)

The owner freed disk space and approved everything; the low Brunel away camera is to be treated as bad video.

**WASB (ball, MIT, volleyball weights).** On the clicked frames (precision / recall at 15 px): Kent 1 0.92 / 0.74
(ours 0.72 / 0.68), Kent 2 1.00 / 0.85 (0.67 / 0.73), Kent 3 0.93 / 0.81 (0.66 / 0.76); Brunel away 3 0.70 / 0.24.
Like VballNet it follows the ball between points, so rally F1 drops (0.92 → 0.86, 0.90 → 0.80, 0.87 → 0.86), but inside
rallies it fits more 3D flights (75–81% → 85–89%) and gives far more plausible serves. Now `vball wasb <id>` writes
`ball_wasb.csv` and `vball ball3d` uses it; rallies are still cut from our tracker. Serves in 40–100 km/h after the
switch: Kent 1–4 70 / 86 / 94 / 70%, Brunel home 1–4 78 / 76 / 86 / 86% (before: about 55–70%).

**TU Graz VB14 (actions).** 36,178 player boxes in 6 Austrian league videos, camera behind a baseline. A ResNet-18 crop
classifier, trained on 5 videos and tested on the sixth: 70% (stand 96, serve 93, reception 97, setting 36, attack 93,
block 62, defence 24%). On our Kent 3 contacts it does not transfer: most overhead touches come out as "block"
(touch-order agreement 8/58, 1/17, 1/5; 1 of 5 serves). It needs some of our own labelled touches.

**VNL-STES + STES (event spotting).** Dataset downloaded (811 / 102 / 115 rallies). The STES code runs on Windows with
three local patches; RegNet-Y 800MF trained at ~4 s per step while sharing the GPU (~4 h per epoch), so a RegNet-Y
200MF run of 12 epochs is the laptop-sized attempt.

**T-DEED** code downloaded; no volleyball weights, and below STES on VNL in the STES paper, so not trained.
**SportsMOT** and **MultiSports** still need the owner (CodaLab Competitions sign-up; request form).
