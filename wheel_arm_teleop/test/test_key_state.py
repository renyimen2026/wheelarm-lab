import pytest

from wheel_arm_teleop.key_state import KeyState


@pytest.mark.parametrize("keys,expected", [
    ([], (0, 0)), (["w"], (0.5, 0)), (["s"], (-0.5, 0)),
    (["a"], (0, 1)), (["d"], (0, -1)), (["w", "a"], (0.5, 1)),
    (["w", "d"], (0.5, -1)), (["s", "a"], (-0.5, 1)),
    (["s", "d"], (-0.5, -1)), (["w", "s", "a", "d"], (0, 0)),
])
def test_velocity_combinations(keys, expected):
    state = KeyState()
    for key in keys:
        state.press(key)
    assert state.velocity(0.5, 1) == expected


def test_release_independent_axes():
    state = KeyState()
    state.press("w")
    state.press("a")
    state.release("w")
    assert state.velocity(0.5, 1) == (0, 1)
    state.release("a")
    assert state.velocity(0.5, 1) == (0, 0)


def test_autorepeat_has_no_effect():
    state = KeyState()
    assert state.press("w")
    assert not state.press("w")
    assert state.velocity(0.5, 1) == (0.5, 0)


def test_space_requires_old_keys_released():
    state = KeyState()
    state.press("w")
    state.press("space")
    state.release("space")
    assert not state.press("w")
    assert state.velocity(0.5, 1) == (0, 0)
    state.release("w")
    state.press("w")
    assert state.velocity(0.5, 1) == (0.5, 0)


def test_focus_loss_clears_everything():
    state = KeyState()
    state.press("w")
    state.press("space")
    state.press("q")
    state.clear()
    assert not state.pressed and not state.blocked
    assert state.velocity(0.5, 1) == (0, 0)
