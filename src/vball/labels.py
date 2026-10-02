"""Hand-labelled rallies saved by the web app (start_s, end_s, approved)."""

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Label:
    start_s: float
    end_s: float
    approved: bool = True  # False = copied from detections and not reviewed yet


def load_labels(path: Path) -> list[Label]:
    with open(path, newline="") as f:
        return [
            Label(float(row["start_s"]), float(row["end_s"]), row.get("approved", "1") != "0")
            for row in csv.DictReader(f)
        ]


def save_labels(path: Path, labels: list[Label]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["start_s", "end_s", "approved"])
        for label in sorted(labels, key=lambda l: l.start_s):
            writer.writerow([round(label.start_s, 3), round(label.end_s, 3), int(label.approved)])
