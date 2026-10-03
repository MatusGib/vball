import argparse
import sys
from pathlib import Path

from vball import pipeline, store
from vball.ball.metrics import ball_metrics, tolerance_px
from vball.ball.testset import load_ball_test
from vball.ball.track import load_tracknet_csv
from vball.ball.tracknet import run_tracknet
from vball.config import BASE_TRACKNET_WEIGHTS, TRACKNET_THRESHOLD, TRACKNET_WEIGHTS, default_paths
from vball.evaluate import rally_metrics, restrict_to_span, visible_fraction
from vball.export import export_rallies
from vball.labels import load_labels
from vball.rallies import RallyParams
from vball.tuning import GRID, TuneCase, grid_search


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

    if args.command == "tunerallies":
        conn = store.connect(paths.db_path)
        try:
            cases = []
            for match_id in args.match_ids:
                m = store.get_match(conn, match_id)
                track = load_tracknet_csv(paths.ball_csv(match_id), m["n_frames"])
                gt = [(label.start_s, label.end_s) for label in load_labels(paths.gt_csv(match_id))]
                cases.append(TuneCase(track, m["fps"], m["width"], m["height"], gt))
        finally:
            conn.close()
        current = grid_search(cases, grid={k: [v] for k, v in vars(RallyParams()).items() if k in GRID})[0]
        print(f"current defaults: mean F1 {current.mean_f1:.3f} per match {[round(f, 3) for f in current.f1s]}")
        for r in grid_search(cases)[: args.top]:
            print(f"mean F1 {r.mean_f1:.3f} per match {[round(f, 3) for f in r.f1s]} {r.params}")
        return 0

    conn = store.connect(paths.db_path)
    try:
        match = store.get_match(conn, args.match_id)
        if match is None:
            print(f"no match with id {args.match_id}", file=sys.stderr)
            return 1

        if args.command == "redetect":
            rallies = pipeline.redetect(conn, paths, args.match_id)
            print(f"{len(rallies)} rallies")

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
