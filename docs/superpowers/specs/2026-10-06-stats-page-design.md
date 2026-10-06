# Stats page — design

Date: 2026-10-06. Branch: `phase2b`. A separate web page with match and cross-match statistics from what vball
already produces (rallies, 3D ball flights, player tracks). Every leaderboard row opens the viewer at that moment
so the number can be checked against the video.

## Sources of each statistic (and their honesty rules)

| Stat | From | Rule |
|---|---|---|
| Rallies | `gt_rallies.csv` if present (source "labels"), else detected rallies ("detected") | count, total rally time, dead-time share of the video, mean / median length, longest rallies |
| Serves | `flights.csv` | per rally, the first fitted flight that crosses the net plane. *Plausible* if 40–100 km/h, net crossing > 2.43 m and, when it lands, within 2 m of the court. Only plausible serves enter leaderboards; the number left out is shown |
| Sets | `flights.csv` | fitted, not the serve, starts within 4 m of the net, rising (v0 z > 0), does not cross the net. Plausible if apex ≤ 8 m and speed ≤ 40 km/h. Ranked by apex |
| Attacks (experimental) | `flights.csv` | fitted, not the serve, starts within 3 m of the net and ≥ 2.3 m up, falling (v0 z < 0), crosses the net. Plausible if 30–130 km/h. Shown with an "experimental, ~13 km/h typical error" badge (simulation, `docs/results/phase2d-ball3d.md`) |
| Players (per side) | `players.csv` + `court.json` | rally frames only, on-court feet (`players.with_court`): mean players on court per side, mean distance from the net per side, heat map of foot positions in 1 m cells over x −2…11, y −4…22 |

Side of a flight = near if it starts at y < 9 m, else far. A per-match `meta.json` (`{"our_side": "near" | "far" |
null}`) turns near/far into "us"/"them" (teams change ends between sets, so it is per match).

## Units

- `src/vball/stats.py` — pure functions: `rally_stats(intervals_s, duration_s)`, `classify_flights(rows, fps)`
  (serves / sets / attacks with plausibility), `player_stats(players, court_xy, on_court, intervals_frames)` (per
  side numbers + heat maps), `match_stats(paths, conn, match_id)` (assembles what exists; a `missing` dict says why a
  section is absent and what to run), `leaderboards(all_match_stats, n=10)`.
- `Paths.meta_json(id)`, `load_meta` / `save_meta` in `stats.py`.
- API in `web/app.py`: `GET /api/stats` (every match's summary + cross-match leaderboards), `GET
  /api/matches/{id}/stats` (full detail incl. heat maps), `PUT /api/matches/{id}/meta`. Player numbers are cached
  per match keyed by the mtimes of `players.csv` and `court.json` (loading a set's tracks takes seconds); the rest
  is cheap and computed per request.
- `web/static/stats.html` + `stats.js`: match picker ("All matches" default); headline tiles; leaderboard tables
  (hardest serves, highest sets, fastest attacks — experimental, longest rallies) with match, time, value, side;
  for one match: the "we are near / far" switch and the two side heat maps on a court drawing. Empty states name
  the command to run. Linked from the viewer header ("Stats →").
- Viewer deep links: `/?match=<id>&t=<seconds>&show=speed,ball,court,players` selects the match, seeks, and ticks
  the overlays. Leaderboard rows link there (serves/sets/attacks with `show=speed`).

## Testing

Unit tests on synthetic flights (a serve, an implausible serve, a set, an attack, a mid-court pass that is none of
them), synthetic player rows and intervals; API tests on a temp data dir; a browser check of the page, the
"us/them" switch and a click-through to the viewer.

## Out of scope

Per-player stats (waiting on better tracking), the Sources page (separate piece), touch-based action detection.
