"""Held-key state without dependence on terminal repeat timing."""


class KeyState:
    BASE_KEYS = {"w", "s", "a", "d"}

    def __init__(self):
        self.pressed = set()
        self.blocked = set()

    def press(self, key):
        if key in self.blocked or key in self.pressed:
            return False
        if key == "space":
            self.stop()
        self.pressed.add(key)
        return True

    def release(self, key):
        self.pressed.discard(key)
        self.blocked.discard(key)

    def clear(self):
        self.pressed.clear()
        self.blocked.clear()

    def stop(self):
        held = self.pressed & self.BASE_KEYS
        self.blocked.update(held)
        self.pressed.difference_update(held)

    def velocity(self, linear_speed, angular_speed):
        if "space" in self.pressed:
            return 0.0, 0.0
        linear = int("w" in self.pressed) - int("s" in self.pressed)
        angular = int("a" in self.pressed) - int("d" in self.pressed)
        return linear * linear_speed, angular * angular_speed
