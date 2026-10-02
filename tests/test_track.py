import numpy as np
from helpers import make_track, write_tracknet_csv

from vball.ball.track import load_tracknet_csv


def test_load_fills_missing_and_invisible_frames(tmp_path):
    csv_path = tmp_path / "ball.csv"
    csv_path.write_text(
        "Frame,Visibility,X,Y\n"
        "0,1,100,200\n"
        "1,0,0,0\n"
        "3,1,110,190\n"
        "9,1,999,999\n"  # beyond n_frames: ignored
    )
    track = load_tracknet_csv(csv_path, n_frames=5)
    assert len(track) == 5
    assert track.visible.tolist() == [True, False, False, True, False]
    assert track.x[0] == 100 and track.y[3] == 190
    assert np.isnan(track.x[1]) and np.isnan(track.x[2]) and np.isnan(track.y[4])


def test_load_header_only_file(tmp_path):
    csv_path = tmp_path / "ball.csv"
    csv_path.write_text("Frame,Visibility,X,Y\n")
    track = load_tracknet_csv(csv_path, n_frames=3)
    assert not track.visible.any()


def test_round_trip_with_helper(tmp_path):
    track = make_track(90, flights=[(1.0, 2.0)])
    loaded = load_tracknet_csv(write_tracknet_csv(tmp_path / "b.csv", track), n_frames=90)
    assert loaded.visible.tolist() == track.visible.tolist()
