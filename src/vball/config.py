import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
TRACKNET_DIR = REPO_ROOT / "third_party" / "TrackNetV3"
TRACKNET_WEIGHTS = MODELS_DIR / "tracknet_volleyball.pt"


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


def default_paths() -> Paths:
    return Paths(Path(os.environ.get("VBALL_DATA", REPO_ROOT / "data")))
