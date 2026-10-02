import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

Interval = tuple[float, float]


def interval_iou(a: Interval, b: Interval) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def match_intervals(pred: list[Interval], gt: list[Interval], min_iou: float = 0.5) -> list[tuple[int, int]]:
    """Greedy one-to-one matching by descending IoU. Returns sorted (pred_index, gt_index) pairs."""
    pairs = sorted(
        ((interval_iou(p, g), i, j) for i, p in enumerate(pred) for j, g in enumerate(gt)),
        reverse=True,
    )
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    matches = []
    for iou, i, j in pairs:
        if iou < min_iou:
            break
        if i in used_pred or j in used_gt:
            continue
        used_pred.add(i)
        used_gt.add(j)
        matches.append((i, j))
    return sorted(matches)


@dataclass(frozen=True)
class RallyMetrics:
    n_pred: int
    n_gt: int
    n_matched: int
    precision: float
    recall: float
    f1: float
    mean_abs_start_err_s: float
    mean_abs_end_err_s: float

    def summary(self) -> str:
        return (
            f"rallies: pred {self.n_pred} gt {self.n_gt} matched {self.n_matched} | "
            f"precision {self.precision:.2f} recall {self.recall:.2f} f1 {self.f1:.2f} | "
            f"start err {self.mean_abs_start_err_s:.2f}s end err {self.mean_abs_end_err_s:.2f}s"
        )


def rally_metrics(pred: list[Interval], gt: list[Interval], min_iou: float = 0.5) -> RallyMetrics:
    matches = match_intervals(pred, gt, min_iou)
    k = len(matches)
    precision = k / len(pred) if pred else 0.0
    recall = k / len(gt) if gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    start_err = float(np.mean([abs(pred[i][0] - gt[j][0]) for i, j in matches])) if matches else math.nan
    end_err = float(np.mean([abs(pred[i][1] - gt[j][1]) for i, j in matches])) if matches else math.nan
    return RallyMetrics(len(pred), len(gt), k, precision, recall, f1, start_err, end_err)


def restrict_to_span(pred: list[Interval], gt: list[Interval], margin_s: float = 5.0) -> list[Interval]:
    """Keep predictions overlapping the labelled time span, so partial labelling (one set) is fair."""
    if not gt:
        return pred
    lo = min(g[0] for g in gt) - margin_s
    hi = max(g[1] for g in gt) + margin_s
    return [p for p in pred if p[1] > lo and p[0] < hi]


def visible_fraction(visible: np.ndarray, intervals: list[Interval], fps: float) -> float:
    """Fraction of frames inside the intervals where the ball was detected."""
    total = seen = 0
    for start_s, end_s in intervals:
        a, b = max(0, int(start_s * fps)), min(len(visible), int(end_s * fps))
        total += max(0, b - a)
        seen += int(visible[a:b].sum())
    return seen / total if total else 0.0


def load_intervals_csv(path: Path) -> list[Interval]:
    with open(path, newline="") as f:
        return [(float(row["start_s"]), float(row["end_s"])) for row in csv.DictReader(f)]


def save_intervals_csv(path: Path, intervals: list[Interval]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["start_s", "end_s"])
        for start_s, end_s in sorted(intervals):
            writer.writerow([round(start_s, 3), round(end_s, 3)])
