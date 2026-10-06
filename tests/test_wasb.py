import numpy as np

from vball.ball.wasb import IN_H, IN_W, peaks_to_rows


def test_heatmap_peaks_become_tracknet_rows_in_work_video_pixels():
    hm = np.zeros((2, IN_H, IN_W), dtype=np.float32)
    hm[0, 100, 200] = 0.9  # a ball
    hm[1, 50, 60] = 0.3  # below the threshold: no ball
    rows = peaks_to_rows(hm, first_frame=30, sx=1920 / IN_W, sy=1080 / IN_H)
    assert rows == ["30,1,750,375", "31,0,0,0"]
