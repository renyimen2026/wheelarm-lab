from types import SimpleNamespace

from wheel_arm_teleop.keyboard_window import KeyboardWindow


class Scheduler:
    def __init__(self):
        self.callbacks = {}
        self.counter = 0

    def after_idle(self, callback):
        self.counter += 1
        self.callbacks[self.counter] = callback
        return self.counter

    def after_cancel(self, identifier):
        self.callbacks.pop(identifier, None)

    def idle(self):
        callbacks = list(self.callbacks.values())
        self.callbacks.clear()
        for callback in callbacks:
            callback()


def window():
    view = KeyboardWindow.__new__(KeyboardWindow)
    view.root = Scheduler()
    view.pending = {}
    view.focused = True
    view.events = []
    view.press = lambda key: view.events.append(("press", key))
    view.release = lambda key: view.events.append(("release", key))
    view.clear = lambda: view.events.append(("clear", None))
    view.close = lambda: view.events.append(("close", None))
    view.status = SimpleNamespace(set=lambda _: None)
    return view


def key(name, state=0):
    return SimpleNamespace(keysym=name, state=state)


def test_real_release_delivered_at_idle():
    view = window()
    view.on_release(key("w"))
    assert not view.events
    view.root.idle()
    assert view.events == [("release", "w")]


def test_autorepeat_release_press_is_coalesced():
    view = window()
    view.on_release(key("w"))
    view.on_press(key("w"))
    view.root.idle()
    assert view.events == [("press", "w")]


def test_focus_loss_cancels_pending_releases_and_clears():
    view = window()
    view.on_release(key("w"))
    view.on_focus_out(None)
    view.root.idle()
    view.on_press(key("w"))
    assert view.events == [("clear", None)]


def test_quit_shortcuts():
    view = window()
    view.on_press(key("Escape"))
    view.on_press(key("c", state=4))
    assert view.events == [("close", None), ("close", None)]
