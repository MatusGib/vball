from vball.ball.testset import BallTestItem, load_ball_test, sample_test_frames, save_ball_test


def test_samples_only_rally_frames_sorted_unique():
    frames = sample_test_frames([(1.0, 2.0), (10.0, 11.0)], fps=30.0, n=20, seed=1)
    assert len(frames) == 20 == len(set(frames))
    assert frames == sorted(frames)
    assert all(30 <= f < 60 or 300 <= f < 330 for f in frames)


def test_sampling_is_reproducible_and_capped():
    assert sample_test_frames([(0, 10)], 30.0, 5, seed=3) == sample_test_frames([(0, 10)], 30.0, 5, seed=3)
    assert sample_test_frames([(0.0, 0.2)], 30.0, 100, seed=3) == [0, 1, 2, 3, 4, 5]


def test_round_trip(tmp_path):
    items = [BallTestItem(30, "ball", 100.5, 200.0), BallTestItem(10), BallTestItem(20, "none")]
    path = tmp_path / "x" / "ball_test.csv"
    save_ball_test(path, items)
    assert load_ball_test(path) == sorted(items, key=lambda i: i.frame)
