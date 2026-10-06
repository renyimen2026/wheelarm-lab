"""Focused keyboard events with autorepeat release/press pairs coalesced."""

import tkinter as tk


class KeyboardWindow:
    def __init__(self, press, release, clear, close, stop):
        self.root = tk.Tk()
        self.root.title("WheelArm Teleop")
        self.root.geometry("420x220")
        self.root.minsize(360, 200)
        self.press = press
        self.release = release
        self.clear = clear
        self.close = close
        self.pending = {}
        self.focused = False
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        self.status = tk.StringVar(value="未激活")
        self.speed = tk.StringVar(value="v = 0.00 m/s    ω = 0.00 rad/s")
        self.keys = tk.StringVar(value="—")
        tk.Label(self.root, textvariable=self.status).grid(row=0, column=0, pady=(16, 4))
        tk.Label(self.root, textvariable=self.speed, font=("TkDefaultFont", 14)).grid(
            row=1, column=0, padx=12, sticky="nsew")
        tk.Label(self.root, textvariable=self.keys).grid(row=2, column=0, pady=4)
        tk.Button(self.root, text="停止", command=stop, takefocus=False).grid(
            row=3, column=0, pady=(4, 16), ipadx=20)
        self.root.bind("<KeyPress>", self.on_press)
        self.root.bind("<KeyRelease>", self.on_release)
        self.root.bind("<FocusIn>", self.on_focus_in)
        self.root.bind("<FocusOut>", self.on_focus_out)
        self.root.protocol("WM_DELETE_WINDOW", close)

    def on_press(self, event):
        key = event.keysym.lower()
        pending = self.pending.pop(key, None)
        if pending is not None:
            self.root.after_cancel(pending)
        if key == "escape" or (key == "c" and event.state & 0x4):
            self.close()
        elif self.focused:
            self.press(key)
        return "break"

    def on_release(self, event):
        key = event.keysym.lower()
        pending = self.pending.pop(key, None)
        if pending is not None:
            self.root.after_cancel(pending)
        # X11 autorepeat can emit release+press together; a real release survives idle.
        self.pending[key] = self.root.after_idle(lambda: self.finish_release(key))
        return "break"

    def finish_release(self, key):
        self.pending.pop(key, None)
        self.release(key)

    def on_focus_in(self, event):
        self.focused = True
        self.status.set("已激活")

    def on_focus_out(self, event):
        self.focused = False
        for callback in self.pending.values():
            self.root.after_cancel(callback)
        self.pending.clear()
        self.clear()
        self.status.set("未激活")

    def update(self):
        self.root.update()

    def display(self, linear, angular, pressed):
        self.speed.set(f"v = {linear:.2f} m/s    ω = {angular:.2f} rad/s")
        self.keys.set(" + ".join(sorted(key.upper() for key in pressed)) or "—")

    def destroy(self):
        self.root.destroy()
