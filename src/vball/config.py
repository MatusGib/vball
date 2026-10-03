import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
TRACKNET_DIR = REPO_ROOT / "third_party" / "TrackNetV3"
BASE_TRACKNET_WEIGHTS = MODELS_DIR / "tracknet_volleyball.pt"  # downloaded beach-volleyball weights
# Fine-tuned on our indoor footage (vball finetune, docs/results/phase2a-ball.md); not in git, rebuild with:
# vball finetune --matches 1 3 4 5 6 7 8 9 10 --out models/tracknet_vball_v1.pt --rebuild-cache
TRACKNET_WEIGHTS = MODELS_DIR / "tracknet_vball_v1.pt"
TRACKNET_THRESHOLD = 0.3  # heatmap value above which TrackNet reports a ball (chosen in phase 2a)


@dataclass(frozen=True)
class Paths:
    data_dir: Path

    @property
    def db_path(self) -> Path:
        return self.data_dir / "vball.db"

    @property
    def matches_dir(self) -> Path:
        return self.data_dir / "matches"

    def match_dir(self, match_id: int) -> Path:
        return self.matches_dir / str(match_id)

    def work_video(self, match_id: int) -> Path:
        return self.match_dir(match_id) / "work.mp4"

    def ball_csv(self, match_id: int) -> Path:
        return self.match_dir(match_id) / "ball.csv"

    def gt_csv(self, match_id: int) -> Path:
        """Hand-labelled rally intervals saved by the web app."""
        return self.match_dir(match_id) / "gt_rallies.csv"

    def track_video(self, match_id: int) -> Path:
        """512x288 copy of the work video that TrackNet reads (kept for re-tracking and training)."""
        return self.match_dir(match_id) / "track.mp4"

    def ball_test_csv(self, match_id: int) -> Path:
        """Hand-clicked ball positions used to score the ball tracker."""
        return self.match_dir(match_id) / "ball_test.csv"

    @property
    def ball_train_dir(self) -> Path:
        return self.data_dir / "ball_train"


def default_paths() -> Paths:
    return Paths(Path(os.environ.get("VBALL_DATA", REPO_ROOT / "data")))
