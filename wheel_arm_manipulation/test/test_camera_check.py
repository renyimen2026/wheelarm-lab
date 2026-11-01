import numpy as np
import pytest

from wheel_arm_manipulation.camera_check import depth_statistics


def test_invalid_depth_is_not_confused_with_display_clipping():
    stats = depth_statistics(np.array([[np.inf, -np.inf], [np.nan, 0.]]))
    assert stats["valid"] == 0 and stats["percentiles"] is None
    assert stats["nan"] == stats["posinf"] == stats["neginf"] == stats["zero"] == .25
    far = depth_statistics(np.full((3, 4), 5.))
    assert far["valid"] == 1 and far["display"] == 0
    assert far["percentiles"] == [5., 5., 5.]
    near = depth_statistics(np.full((3, 4), .8))
    assert near["display"] == 1
    with pytest.raises(ValueError):
        depth_statistics(np.empty((0, 0)))
