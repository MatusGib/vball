import argparse
import sys
from pathlib import Path

from vball import pipeline, store
from vball import players as pl_mod
from vball.ball.metrics import ball_metrics, tolerance_px
from vball.ball.testset import load_ball_test
from vball.ball.track import load_tracknet_csv
from vball.ball.tracknet import run_tracknet
from vball.config import ACTIONS_WEIGHTS, BASE_TRACKNET_WEIGHTS, TRACKNET_THRESHOLD, TRACKNET_WEIGHTS, default_paths
from vball.court import load_calibration
from vball.serve import ServeParams
from vball.evaluate import rally_metrics, restrict_to_span, visible_fraction
from vball.export import export_rallies
from vball.labels import load_labels
from vball.rallies import RallyParams
from vball.tuning import GRID, SERVE_GRID, TuneCase, grid_search


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vball", description="Volleyball video analysis")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("process", help="ingest a match video, track the ball, detect rallies")
    p.add_argument("video", type=Path)
    p.add_argument("--name")
    p.add_argument("--encoder", default="libx264", help="ffmpeg encoder; h264_nvenc needs NVIDIA driver >= 610")
    p.add_argument("--weights", type=Path, default=TRACKNET_WEIGHTS, help="TrackNet checkpoint for ball tracking")
    p.add_argument("--threshold", type=float, default=TRACKNET_THRESHOLD)

    r = sub.add_parser("redetect", help="re-run rally detection from the cached ball track")
    r.add_argument("match_id", type=int)
    r.add_argument("--serve", action="store_true", help="experimental: require a serve (needs players + court)")

    e = sub.add_parser("export", help="write per-rally clips and a rallies-only video")
    e.add_argument("match_id", type=int)
    e.add_argument("--out", type=Path)

    v = sub.add_parser("eval", help="compare detected rallies with hand labels (start_s,end_s CSV)")
    v.add_argument("match_id", type=int)
    v.add_argument("gt_csv", type=Path, nargs="?", help="defaults to the labels saved in the web app")

    t = sub.add_parser("track", help="re-run ball tracking on a processed match")
    t.add_argument("match_id", type=int)
    t.add_argument("--weights", type=Path, default=TRACKNET_WEIGHTS)
    t.add_argument("--threshold", type=float, default=TRACKNET_THRESHOLD)
    t.add_argument("--out", type=Path, help="write here instead of the match's ball.csv (rallies are left alone)")

    b = sub.add_parser("balleval", help="score a ball track against the frames clicked on the Ball check page")
    b.add_argument("match_id", type=int)
    b.add_argument("--pred", type=Path, help="ball CSV to score (default: the match's ball.csv)")

    f = sub.add_parser("finetune", help="fine-tune TrackNet on pseudo-labels from processed matches")
    f.add_argument("--matches", type=int, nargs="+", required=True, help="training match ids (never the held-out ones)")
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--init", type=Path, default=BASE_TRACKNET_WEIGHTS)
    f.add_argument("--windows", type=int, default=2000)
    f.add_argument("--epochs", type=int, default=4)
    f.add_argument("--batch-size", type=int, default=2)
    f.add_argument("--rebuild-cache", action="store_true", help="needed whenever --matches changes")
    f.add_argument(
        "--vballnet", type=Path, help="VballNet dataset root (.../vballnet-dataset/data) to add as labelled sources"
    )

    g = sub.add_parser("tunerallies", help="grid-search rally parameters against hand-labelled matches")
    g.add_argument("match_ids", type=int, nargs="+")
    g.add_argument("--top", type=int, default=8)
    g.add_argument("--serve", action="store_true", help="experimental: tune the serve check (needs players + court)")

    pl = sub.add_parser("players", help="detect and track players")
    pl.add_argument("match_id", type=int)
    pl.add_argument("--backend", choices=["yolo", "ravel"], default="yolo")
    pl.add_argument("--out", type=Path)
    pl.add_argument(
        "--ravel-repo", type=Path, default=Path("data/external/asigatchov-fast-volleyball-tracking-inference")
    )

    pe = sub.add_parser("playereval", help="label-free player tracking quality on rally frames")
    pe.add_argument("match_id", type=int)
    pe.add_argument("--players", type=Path)

    cc = sub.add_parser("courtcopy", help="draft a court calibration from another set filmed from the same spot")
    cc.add_argument("src", type=int, help="calibrated set to copy from")
    cc.add_argument("dst", type=int, help="set to draft a calibration for")
    cc.add_argument("--force", action="store_true", help="replace an existing calibration")

    c3 = sub.add_parser("camera", help="fit the 3D camera to the court calibration and report it")
    c3.add_argument("match_id", type=int)

    bs = sub.add_parser("ball3dsim", help="simulate flights through this match's camera; report 3D metric errors")
    bs.add_argument("match_id", type=int)
    bs.add_argument("--count", type=int, default=100)
    bs.add_argument("--noise", type=float, nargs="+", default=[3.0, 6.0, 9.0])
    bs.add_argument("--seed", type=int, default=0)

    b3 = sub.add_parser("ball3d", help="fit 3D ball flights in every rally; writes flights.csv")
    b3.add_argument("match_id", type=int)
    b3.add_argument("--noise-px", type=float, default=6.0)

    sv = sub.add_parser("serving", help="which end served each rally (+ receive check); writes serving.csv")
    sv.add_argument("match_id", type=int)
    sv.add_argument("--no-actions", action="store_true", help="skip the action detector (receive check)")

    s = sub.add_parser("serve", help="start the web app")
    s.add_argument("--port", type=int, default=8000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = default_paths()

    if args.command == "process":
        match_id = pipeline.process_match(
            args.video,
            paths,
            name=args.name,
            encoder=args.encoder,
            ball_runner=lambda video, out_csv: run_tracknet(
                video, out_csv, weights=args.weights, threshold=args.threshold
            ),
        )
        print(f"match {match_id} processed")
        return 0

    if args.command == "serve":
        import uvicorn

        from vball.web.app import create_app

        print(f"open http://127.0.0.1:{args.port}")
        uvicorn.run(create_app(paths), host="127.0.0.1", port=args.port)
        return 0

    if args.command == "finetune":
        from vball.ball import external
        from vball.ball import finetune as ft  # torch-heavy; imported only when needed
        from vball.ball.train_data import build_cache

        conn = store.connect(paths.db_path)
        try:
            sources = ft.prepare_sources(conn, paths, args.matches)
        finally:
            conn.close()
        if args.vballnet:
            sources += external.clip_sources(args.vballnet, paths.data_dir / "external_cache" / "vballnet")
            print(f"{len(sources)} training sources including VballNet clips")
        cache = paths.ball_train_dir
        if args.rebuild_cache or not (cache / "meta.npz").exists():
            build_cache(sources, cache, n_windows=args.windows)
        ft.finetune(cache, args.init, args.out, epochs=args.epochs, batch_size=args.batch_size)
        print(f"wrote {args.out}")
        return 0

    if args.command == "courtcopy":
        import numpy as np

        from vball.camera import MIN_RESPONSE, camera_segments, image_shift, read_gray
        from vball.court import save_calibration, shifted_copy

        conn = store.connect(paths.db_path)
        try:
            src, dst = store.get_match(conn, args.src), store.get_match(conn, args.dst)
        finally:
            conn.close()
        if src is None or dst is None:
            print("no such match", file=sys.stderr)
            return 1
        if not paths.court_json(args.src).exists():
            print(f"calibrate #{args.src} first (viewer: Calibrate court)", file=sys.stderr)
            return 1
        if paths.court_json(args.dst).exists() and not args.force:
            print(f"#{args.dst} is already calibrated; add --force to replace it", file=sys.stderr)
            return 1
        cal = load_calibration(paths.court_json(args.src))
        ref = read_gray(paths.track_video(args.src), cal.ref_frame)
        scale = dst["width"] / 512
        samples = []  # (frame, dx, dy, response) in work-video pixels
        for frac in (0.15, 0.3, 0.45, 0.6, 0.75, 0.9):
            frame = int(dst["n_frames"] * frac)
            dx, dy, response = image_shift(ref, read_gray(paths.track_video(args.dst), frame))
            if response >= MIN_RESPONSE:
                samples.append((frame, dx * scale, dy * scale, response))
        median = np.median([s[1:3] for s in samples], axis=0) if samples else None
        agree = [s for s in samples if np.hypot(s[1] - median[0], s[2] - median[1]) <= 10] if samples else []
        if len(agree) < 3:
            print(f"#{args.src} and #{args.dst} don't line up reliably ({len(agree)} of 6 samples agree): "
                  "calibrate this set by hand", file=sys.stderr)
            return 1
        frame, dx, dy, _ = max(agree, key=lambda s: s[3])
        segments = camera_segments(paths.track_video(args.dst), frame, dst["n_frames"], scale)
        save_calibration(paths.court_json(args.dst), shifted_copy(cal, (dx, dy), frame, segments, args.src))
        print(f"draft court for #{args.dst} from #{args.src}: picture shift {dx:+.0f}, {dy:+.0f} px "
              f"({len(agree)} of 6 samples agree), {len(segments)} camera segment(s). Check it in the viewer.")
        return 0

    if args.command == "tunerallies":
        conn = store.connect(paths.db_path)
        try:
            cases = []
            for match_id in args.match_ids:
                m = store.get_match(conn, match_id)
                track = load_tracknet_csv(paths.ball_csv(match_id), m["n_frames"])
                gt = [(label.start_s, label.end_s) for label in load_labels(paths.gt_csv(match_id))]
                serve = None
                if args.serve and pipeline.serve_status(paths, match_id) == "serve check on":
                    players = pl_mod.load_players(paths.players_csv(match_id))
                    court_xy, _, _ = pl_mod.with_court(players, load_calibration(paths.court_json(match_id)))
                    serve = (players, court_xy)
                cases.append(TuneCase(track, m["fps"], m["width"], m["height"], gt, serve))
        finally:
            conn.close()
        grid = SERVE_GRID if args.serve and all(c.serve is not None for c in cases) else GRID
        if args.serve:
            print("serve check on" if grid is SERVE_GRID else "serve check off (a match lacks player tracks or a court)")
        defaults = {**vars(RallyParams()), **vars(ServeParams())}
        current = grid_search(cases, grid={k: [defaults[k]] for k in grid})[0]
        print(f"current defaults: mean F1 {current.mean_f1:.3f} per match {[round(f, 3) for f in current.f1s]}")
        for r in grid_search(cases, grid)[: args.top]:
            print(f"mean F1 {r.mean_f1:.3f} per match {[round(f, 3) for f in r.f1s]} {r.params}")
        return 0

    conn = store.connect(paths.db_path)
    try:
        match = store.get_match(conn, args.match_id)
        if match is None:
            print(f"no match with id {args.match_id}", file=sys.stderr)
            return 1

        if args.command == "redetect":
            rallies = pipeline.redetect(conn, paths, args.match_id, serve=args.serve)
            print(f"{len(rallies)} rallies")
            if args.serve:
                print(pipeline.serve_status(paths, args.match_id))

        elif args.command == "track":
            out = args.out or paths.ball_csv(args.match_id)
            run_tracknet(
                paths.work_video(args.match_id),
                out,
                weights=args.weights,
                threshold=args.threshold,
                small_video=paths.track_video(args.match_id),
            )
            print(f"wrote {out}")
            if args.out is None:
                print(f"{len(pipeline.redetect(conn, paths, args.match_id))} rallies")

        elif args.command == "players":
            out = args.out or paths.players_csv(args.match_id)
            if args.backend == "yolo":
                pl_mod.run_yolo(paths.work_video(args.match_id), out)
            else:
                pl_mod.run_ravel(paths.work_video(args.match_id), out, args.ravel_repo)
            print(f"wrote {out}")

        elif args.command == "playereval":
            if not paths.court_json(args.match_id).exists():
                print("calibrate the court first (viewer: Calibrate court)", file=sys.stderr)
                return 1
            cal = load_calibration(paths.court_json(args.match_id))
            players = pl_mod.load_players(args.players or paths.players_csv(args.match_id))
            fps = match["fps"]
            gt_path = paths.gt_csv(args.match_id)
            if gt_path.exists():
                rallies = [(round(l.start_s * fps), round(l.end_s * fps)) for l in load_labels(gt_path)]
            else:
                rallies = [(r["start_frame"], r["end_frame"]) for r in store.get_rallies(conn, args.match_id)]
            print(pl_mod.player_stats(players, cal, rallies).summary())

        elif args.command == "serving":
            from vball import serving
            from vball.stats import rally_intervals

            for need, what in ((paths.court_json(args.match_id), "calibrate the court first (viewer: Calibrate court)"),
                               (paths.players_csv(args.match_id), f"run first: uv run vball players {args.match_id}")):
                if not need.exists():
                    print(what, file=sys.stderr)
                    return 1
            fps = match["fps"]
            cal = load_calibration(paths.court_json(args.match_id))
            players = pl_mod.load_players(paths.players_csv(args.match_id))
            order = players.frame.argsort(kind="stable")
            players = pl_mod.Players(players.frame[order], players.track_id[order], players.box[order],
                                     players.score[order])
            court_xy = pl_mod.feet_to_court(players.frame, players.box, cal)
            track = load_tracknet_csv(paths.ball_csv(args.match_id), match["n_frames"])
            intervals, source = rally_intervals(paths, conn, args.match_id)
            rows = serving.match_serving(intervals, track, players, court_xy, fps)
            if not args.no_actions and ACTIONS_WEIGHTS.exists():
                print(f"action detector on {len(rows)} serves...")
                actions = serving.detect_actions(paths.work_video(args.match_id), serving.action_windows(rows, fps),
                                                 cal, ACTIONS_WEIGHTS, every=max(1, round(fps / 15)))
                serving.save_actions(paths.actions_csv(args.match_id), actions)
                serving.mark_defense(rows, actions, fps)
            elif not args.no_actions:
                print(f"no action detector at {ACTIONS_WEIGHTS}: receive check skipped")
            serving.save_serving(paths.serving_csv(args.match_id), rows)
            how = [r["how"] for r in rows]
            ends = "".join({"near": "N", "far": "F"}.get(r["end"], "?") for r in rows)
            print(f"{len(rows)} rallies ({source}): contact {how.count('contact')}, count {how.count('count')}, "
                  f"unknown {how.count('none')}")
            print(f"serving ends: {ends}")
            print(f"wrote {paths.serving_csv(args.match_id)}")

        elif args.command in ("camera", "ball3dsim", "ball3d"):
            from vball import ball3d, ball3d_sim  # scipy-heavy; imported only when needed
            from vball.camera3d import fit_camera

            if not paths.court_json(args.match_id).exists():
                print("calibrate the court first (viewer: Calibrate court)", file=sys.stderr)
                return 1
            cam, err = fit_camera(load_calibration(paths.court_json(args.match_id)), match["width"], match["height"])
            fps = match["fps"]
            gt_path = paths.gt_csv(args.match_id)
            if gt_path.exists():
                rallies = [(round(l.start_s * fps), round(l.end_s * fps)) for l in load_labels(gt_path)]
            else:
                rallies = [(r["start_frame"], r["end_frame"]) for r in store.get_rallies(conn, args.match_id)]
            if args.command == "camera":
                n_floor = len(cam.cal.points)
                x, y, z = cam.centre()
                print(f"focal {cam.K[0, 0]:.0f} px | camera at x {x:.2f} y {y:.2f} z {z:.2f} m")
                inside = [0 <= p["x"] < match["width"] and 0 <= p["y"] < match["height"] for p in cam.cal.points]
                used = [e for e, ok in zip(err[:n_floor], inside) if ok]
                print(f"floor error mean {sum(used) / len(used):.1f} px max {max(used):.1f} px over {len(used)} points")
                off = [p["landmark"] for p, ok in zip(cam.cal.points, inside) if not ok]
                if off:
                    print(f"not used (clicked outside the picture): {', '.join(off)}")
                if len(err) > n_floor:
                    print(f"net error {', '.join(f'{e:.1f}' for e in err[n_floor:])} px")
                else:
                    print("no net clicks: height comes from the floor alone")
            elif args.command == "ball3dsim":
                track = load_tracknet_csv(paths.ball_csv(args.match_id), match["n_frames"])
                rows = ball3d_sim.run_simulation(cam, track, rallies, fps, match["width"], match["height"],
                                                 args.count, args.noise, args.seed)
                for r in rows:
                    print(" | ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}" for k, v in r.items()))
            else:
                track = load_tracknet_csv(paths.ball_csv(args.match_id), match["n_frames"])
                rows = ball3d.match_flights(cam, track, rallies, fps, args.noise_px)
                ball3d.save_flights(paths.flights_csv(args.match_id), rows)
                print(ball3d.summary(rows))
                print(f"wrote {paths.flights_csv(args.match_id)}")

        elif args.command == "balleval":
            test_path = paths.ball_test_csv(args.match_id)
            if not test_path.exists():
                print("no ball test frames yet; click them on the Ball check page (/ball.html)", file=sys.stderr)
                return 1
            items = load_ball_test(test_path)
            track = load_tracknet_csv(args.pred or paths.ball_csv(args.match_id), match["n_frames"])
            print(ball_metrics(track, items, tolerance_px(match["width"])).summary())
            done = sum(item.status in ("ball", "none") for item in items)
            print(f"{done} of {len(items)} test frames labelled")

        elif args.command == "export":
            intervals = [(r["start_s"], r["end_s"]) for r in store.get_rallies(conn, args.match_id)]
            out = args.out or paths.match_dir(args.match_id) / "rallies_only.mp4"
            export_rallies(paths.work_video(args.match_id), intervals, out, paths.match_dir(args.match_id) / "clips")
            print(f"wrote {out}")

        elif args.command == "eval":
            gt_path = args.gt_csv or paths.gt_csv(args.match_id)
            if not gt_path.exists():
                print(f"no labels at {gt_path}; label rallies in the web app first", file=sys.stderr)
                return 1
            pred = [(r["start_s"], r["end_s"]) for r in store.get_rallies(conn, args.match_id)]
            labels = load_labels(gt_path)
            gt = [(label.start_s, label.end_s) for label in labels]
            unapproved = sum(not label.approved for label in labels)
            if unapproved:
                print(f"warning: {unapproved} of {len(labels)} labels not approved yet (copied detections); scores are optimistic")
            print(rally_metrics(restrict_to_span(pred, gt), gt).summary())
            duration_s = match["n_frames"] / match["fps"]
            kept_s = sum(end - start for start, end in pred)
            print(f"dead time removed: {1 - kept_s / duration_s:.0%} of {duration_s / 60:.1f} min")
            track = load_tracknet_csv(paths.ball_csv(args.match_id), match["n_frames"])
            print(f"ball visible in {visible_fraction(track.visible, gt, match['fps']):.0%} of labelled rally frames")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
