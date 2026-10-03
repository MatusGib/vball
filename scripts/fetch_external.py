"""Fetch auto-downloadable external sources into $VBALL_DATA/external (default data/external).

Source of truth for what each item is: docs/references/external_data.md
Manual-only sources (VNL-STES, Ibrahim Volleyball, TU Graz, SportsMOT) are printed, not fetched.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import pathlib
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile

ROOT = pathlib.Path(os.environ.get("VBALL_DATA", "data")) / "external"

GIT = {
    "ddecks-beach-volley-vision": "https://github.com/ddecks/beach-volley-vision",
    "asigatchov-fast-volleyball-tracking-inference": "https://github.com/asigatchov/fast-volleyball-tracking-inference",
    "asigatchov-vball-net-pytorch": "https://github.com/asigatchov/vball-net-pytorch",
    "asigatchov-court-keypoint-detection": "https://github.com/asigatchov/Court-Keypoint-Detection",
    "asigatchov-vaa": "https://github.com/asigatchov/VAA",
    "nttcom-wasb-sbdt": "https://github.com/nttcom/WASB-SBDT",
    "hoangqnguyen-spot": "https://github.com/hoangqnguyen/spot",
    "arturxe2-t-deed": "https://github.com/arturxe2/T-DEED",
    "shukkkur-volleyvision": "https://github.com/shukkkur/VolleyVision",
    "haotianxia-vren": "https://github.com/haotianxia/VREN",
    "mostafa-saad-deep-activity-rec": "https://github.com/mostafa-saad/deep-activity-rec",
    "mcg-nju-sportsmot": "https://github.com/MCG-NJU/SportsMOT",
    "mcg-nju-multisports": "https://github.com/MCG-NJU/MultiSports",
    "williamyen042-passform": "https://github.com/williamyen042/passform",
}

ARCHIVES = {
    "vballnet-dataset": "https://volleyball-orel.ru/static/system/docs/data_20250711_2330.tgz",
}

# id: (repo_id, repo_type, files or None for whole repo)
HF = {
    "deadfast-tracknet": ("deadfast/beach-volley-vision-models", "model", ["tracknet_best.pt"]),
}
HF_GATED = {
    "multisports": ("MCG-NJU/MultiSports", "dataset", None),
}

# id: (workspace, project, version, export_format). Versions marked ⚠ in the reference doc are guesses.
ROBOFLOW = {
    "volleyvision-ball": ("shukur-sabzaliev1", "volleyball_v2", 2, "yolov8"),
    "volleyvision-actions": ("shukur-sabzaliev-42xvj", "volleyball-actions", 5, "yolov8"),
    "volleyvision-players": ("shukur-sabzaliev-42xvj", "players-dataset", 1, "yolov8"),
    "volleyvision-court-seg": ("shukur-sabzaliev-bh7pq", "court-segmented", 1, "png-mask-semantic"),
    "tugraz-roboflow-mirror": ("shukur-sabzaliev-zc3en", "volleyball-activity-dataset", 3, "yolov8"),
    "daniel-ball": ("daniel-eqnnu", "volleyball_ball_object_detection_dataset-3jez2", 1, "yolov8"),
    "myarmy-ball": ("myarmy", "volleyball-2vfip", 2, "yolov8"),
    "aivolleyballref-ball": ("aivolleyballref", "volleyball_detection", 2, "yolov8"),
    "aimbotcod-ball": ("aimbotcod", "volley-ball-recognition", 5, "yolov8"),
    "protom-court-keypoints": ("protom", "volleyball-court-detection-tpvsi", 1, "yolov8"),
}

MANUAL = {
    "vnl-stes": "Resolve https://bit.ly/vnlvolley1 in a browser; if Google Drive: uvx gdown --folder <url> -O data/external/vnl-stes",
    "ibrahim-volleyball": "Download links in https://github.com/mostafa-saad/deep-activity-rec README (mirrors may hit quota)",
    "tugraz-volleyball-2014": "https://tugraz.at/index.php?id=17751 (or --roboflow mirror 'tugraz-roboflow-mirror', re-split by video)",
    "sportsmot": "Sign up on CodaLab via https://deeperaction.github.io/datasets/sportsmot.html, Participate > Get Data",
    "ovscout2": "https://github.com/openvolley/ovscout2/releases/download/v0.1.0/ovscout2-win-x64.zip",
}


def stamp(dest: pathlib.Path, url: str) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "_source.txt").write_text(
        f"url: {url}\nfetched: {dt.date.today().isoformat()}\nlicence: see docs/references/external_data.md\n",
        encoding="utf-8",
    )


def fetch_git(name: str, url: str) -> None:
    dest = ROOT / name
    if dest.exists():
        print(f"skip {name} (exists)")
        return
    subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=True)
    stamp(dest, url)


def fetch_archive(name: str, url: str) -> None:
    dest = ROOT / name
    if (dest / "_source.txt").exists():
        print(f"skip {name} (exists)")
        return
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / url.rsplit("/", 1)[-1]
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (vball fetch)"})
    print(f"download {url}")
    with urllib.request.urlopen(req) as r, open(archive, "wb") as f:
        shutil.copyfileobj(r, f)
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as t:
            if hasattr(tarfile, "data_filter"):
                t.extractall(dest, filter="data")
            else:
                t.extractall(dest)
    elif zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    stamp(dest, url)


def fetch_hf(name: str, repo_id: str, repo_type: str, files: list[str] | None) -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    dest = ROOT / name
    if files:
        for f in files:
            hf_hub_download(repo_id, f, repo_type=repo_type, local_dir=dest)
    else:
        snapshot_download(repo_id, repo_type=repo_type, local_dir=dest)
    prefix = "datasets/" if repo_type == "dataset" else ""
    stamp(dest, f"https://huggingface.co/{prefix}{repo_id}")


def fetch_roboflow(name: str, ws: str, project: str, version: int, fmt: str) -> None:
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise RuntimeError("ROBOFLOW_API_KEY not set")
    from roboflow import Roboflow

    dest = ROOT / "roboflow" / name
    if dest.exists():
        print(f"skip {name} (exists)")
        return
    Roboflow(api_key=key).workspace(ws).project(project).version(version).download(fmt, location=str(dest))
    stamp(dest, f"https://universe.roboflow.com/{ws}/{project}/dataset/{version}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="source ids to fetch")
    ap.add_argument("--roboflow", action="store_true", help="include Roboflow sets")
    ap.add_argument("--gated", action="store_true", help="include gated HF datasets")
    ap.add_argument("--list", action="store_true", help="list ids and exit")
    args = ap.parse_args()

    jobs = []
    jobs += [(n, fetch_git, (n, u)) for n, u in GIT.items()]
    jobs += [(n, fetch_archive, (n, u)) for n, u in ARCHIVES.items()]
    jobs += [(n, fetch_hf, (n, *spec)) for n, spec in HF.items()]
    if args.gated or args.only:
        jobs += [(n, fetch_hf, (n, *spec)) for n, spec in HF_GATED.items()]
    if args.roboflow or args.only:
        jobs += [(n, fetch_roboflow, (n, *spec)) for n, spec in ROBOFLOW.items()]

    if args.list:
        for n, fn, _ in jobs:
            print(f"{n:50s} {fn.__name__}")
        for n, hint in MANUAL.items():
            print(f"{n:50s} manual: {hint}")
        return

    if args.only:
        jobs = [j for j in jobs if j[0] in set(args.only)]

    ROOT.mkdir(parents=True, exist_ok=True)
    failed = []
    for name, fn, fargs in jobs:
        try:
            fn(*fargs)
        except Exception as e:  # keep going, report at end
            failed.append((name, repr(e)))
            print(f"FAILED {name}: {e}")

    print("\nManual sources:")
    for n, hint in MANUAL.items():
        print(f"  {n}: {hint}")
    if failed:
        print("\nFailures:")
        for n, e in failed:
            print(f"  {n}: {e}")


if __name__ == "__main__":
    main()
