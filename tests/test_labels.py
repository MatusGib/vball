from vball.labels import Label, load_labels, save_labels


def test_round_trip_sorted_with_approval(tmp_path):
    path = tmp_path / "sub" / "gt.csv"
    save_labels(path, [Label(20.0, 31.25, approved=False), Label(1.5, 9.0)])
    assert load_labels(path) == [Label(1.5, 9.0, True), Label(20.0, 31.25, False)]


def test_files_without_approved_column_count_as_approved(tmp_path):
    path = tmp_path / "gt.csv"
    path.write_text("start_s,end_s\n1.5,9.0\n")
    assert load_labels(path) == [Label(1.5, 9.0, True)]
