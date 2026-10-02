import argparse
import sys
from pathlib import Path

from vball import pipeline, store
from vball.ball.track import load_tracknet_csv
from vball.config import default_paths
from vball.evaluate import rally_metrics, restrict_to_span, visible_fraction
from vball.export import export_rallies
from vball.labels import load_labels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vball", description="Volleyball video analysis")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("process", help="ingest a match video, track the ball, detect rallies")
    p.add_argument("video", type=Path)
    p.add_argument("--name")
    p.add_argument("--encoder", default="libx264", help="ffmpeg encoder; h264_nvenc needs NVIDIA driver >= 610")

    r = sub.add_parser("redetect", help="re-run rally detection from the cached ball track")
    r.add_argument("match_id", type=int)

    e = sub.add_parser("export", help="write per-rally clips and a rallies-only video")
    e.add_argument("match_id", type=int)
    e.add_argument("--out", type=Path)

    v = sub.add_parser("eval", help="compare detected rallies with hand labels (start_s,end_s CSV)")
    v.add_argument("match_id", type=int)
    v.add_argument("gt_csv", type=Path, nargs="?", help="defaults to the labels saved in the web app")

    s = sub.add_parser("serve", help="start the web app")
    s.add_argument("--port", type=int, default=8000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = default_paths()

    if args.command == "process":
        match_id = pipeline.process_match(args.video, paths, name=args.name, encoder=args.encoder)
        print(f"match {match_id} processed")
        return 0

    if args.command == "serve":
        import uvicorn

        from vball.web.app import create_app

        print(f"open http://127.0.0.1:{args.port}")
        uvicorn.run(create_app(paths), host="127.0.0.1", port=args.port)
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
