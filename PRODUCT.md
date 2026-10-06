# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

- **Primary: the owner**, a player on Imperial M2 (university men's volleyball). Reviews his team's match recordings after the game: cuts rallies, checks and corrects automatic labels, calibrates each video's court, and looks at ball and player statistics.
- **Secondary: teammates and the coach**, shown stats, clips and heat maps by the owner (e.g. after a match). They do not operate the tool themselves.

## Product Purpose

vball is a personal clone of Balltime's "Volleyball AI": it turns a phone recording of an indoor match into rally clips, ball tracks, 3D ball flights (serve speed, set height), player positions and statistics. Success: the owner can review a set in minutes instead of scrubbing raw video, and every number can be checked against the moment in the video it came from.

## Positioning

Built around one specific setup and team: a phone on a tripod behind the baseline at the team's own matches, with every metric honest about its accuracy (plausibility rules, "experimental" badges, measured error) and one click from the video moment that produced it.

## Operating Context

- Runs locally on the owner's Windows laptop (RTX 3050, 4 GB) as `uv run vball serve`, opened at `http://127.0.0.1:8000`. Laptop screen only.
- Workflow per set: `vball process` → label/approve rallies in the viewer → Calibrate court (with net clicks) → `vball players` / `vball ball3d` → Stats page.
- Pages: Viewer (`/`: video, rally timeline, labels, overlays, calibration), Ball check (`/ball.html`), Stats (`/stats.html`), Sources (`/sources.html`).
- Footage: 11 sets from 3 matches (Kent, Brunel home, Brunel away), 1080p, 30/60 fps; gyms are bright wooden-floor sports halls.

## Capabilities and Constraints

- FastAPI backend, vanilla HTML/CSS/JS front end, no build step, no external network needed at run time.
- Keyboard-driven labelling in the viewer (S/E/N/P/U/A/Esc); element ids are relied on by the JS and the test suite.
- Light and dark appearance both supported (system preference).
- Terminology: rally, serve, set (of a match) vs set (the pass), near end / far end (camera end = near), us / them once the team's end is set.
- Undecided: per-player stats (needs better tracking), touch/action tagging (phase 2c).

## Brand Commitments

- Name: **vball**.
- Carries the **Imperial M2** team identity (explicitly requested). Specific colours and marks are not yet confirmed by the owner; evidence in footage: black match shirts with white "IMPERIAL" lettering, red libero shirt. Do not use Imperial College logos or crests.

## Evidence on Hand

- Real match data in `data/matches/<id>/` (rallies, labels, ball tracks, flights, player tracks, court calibrations).
- Measured results in `docs/results/` and `docs/PROJECT_SUMMARY.md`.
- No testimonials, users, or external claims exist; none may be invented.

## Product Principles

1. Every number links back to the video moment that produced it.
2. Be honest about accuracy: show uncertainty, plausibility filters and what is missing, and say which command fixes it.
3. The review loop (watch, correct, approve) must stay fast and keyboard-first.
4. Readable for a teammate or coach glancing over the owner's shoulder, without explanation.
