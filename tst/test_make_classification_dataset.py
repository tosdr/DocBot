import bisect


def test_stretch_points():
    a = [0, 12, 24, 30]
    for bounds, n in [
        ((12, 24), 1),
        ((16, 24), 1),
        ((12, 16), 1),
        ((16, 20), 1),
        ((12, 30), 2),
        ((8, 24), 2),
        ((8, 30), 3),
    ]:
        sent_idx_start = bisect.bisect_right(a, bounds[0]) - 1
        sent_idx_end = bisect.bisect_left(a, bounds[1])
        assert sent_idx_end - sent_idx_start == n, bounds
